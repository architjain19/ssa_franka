#!/usr/bin/env python3
"""
ROS1 Noetic service node for ArUco marker detection.

Service
-------
  /robot/perception/detect_markers   (robot_api_interfaces/RobotQuery)

Behaviour
---------
On each request the node grabs one fresh frame from:
  * /realsense/scene/color/image_raw
  * /realsense/scene/color/camera_info
  * /realsense/scene/aligned_depth_to_color/image_raw   (optional)

It detects ArUco markers from the DICT_6X6_250 dictionary, filters them by a
configurable target-id list, computes the 6-DoF pose of each marker centre,
and returns the result as a JSON string in the `data` field of the response.

Response JSON schema:
{
  "markers": [
    {
      "id": <int>,
      "pose_wrt_camera":    {"position": {x,y,z}, "orientation": {x,y,z,w}},
      "pose_wrt_base_link": {"position": {x,y,z}, "orientation": {x,y,z,w}}  // null if TF fails
    },
    ...
  ]
}

Parameters (all private, with sensible defaults)
------------------------------------------------
  ~service_name        (str)   service name to advertise
  ~color_topic         (str)   RGB image topic
  ~camera_info_topic   (str)   color camera_info topic
  ~depth_topic         (str)   aligned depth-to-color image topic
  ~marker_size         (float) marker edge length in metres (default 0.05)
  ~target_ids          (int[]) marker ids of interest (default [1,2,3,4,5])
  ~base_frame          (str)   robot base TF frame (default "base_link")
  ~msg_timeout         (float) seconds to wait for camera messages (default 2.0)
  ~tf_timeout          (float) seconds to wait for TF (default 1.0)
  ~use_depth_position  (bool)  refine translation with measured depth (default True)
"""

import json
import threading

import cv2
import numpy as np
import ros_numpy
import rospy
import tf2_ros
import tf2_geometry_msgs  # noqa: F401  -- registers PoseStamped transform handler
from geometry_msgs.msg import PoseStamped, TransformStamped
from sensor_msgs.msg import CameraInfo, Image
from tf.transformations import quaternion_from_matrix, quaternion_matrix

from robot_api_interfaces.srv import RobotQuery, RobotQueryResponse
from robot_api_interfaces.msg import ResultCode


class MarkerDetectionService:
    def __init__(self):
        # -------- Parameters --------
        self.service_name       = rospy.get_param("~service_name",       "/robot/perception/detect_markers")
        self.color_topic        = rospy.get_param("~color_topic",        "/realsense/scene/color/image_raw")
        self.camera_info_topic  = rospy.get_param("~camera_info_topic",  "/realsense/scene/color/camera_info")
        self.depth_topic        = rospy.get_param("~depth_topic",        "/realsense/scene/aligned_depth_to_color/image_raw")
        self.marker_size        = float(rospy.get_param("~marker_size",  0.08))
        self.target_ids         = list(rospy.get_param("~target_ids",    [1, 2, 3, 4, 5]))
        self.base_frame         = rospy.get_param("~base_frame",         "panda_link0")
        self.msg_timeout        = float(rospy.get_param("~msg_timeout",  2.0))
        self.tf_timeout         = float(rospy.get_param("~tf_timeout",   1.0))
        self.use_depth_position = bool(rospy.get_param("~use_depth_position", True))
        self.publish_tf         = bool(rospy.get_param("~publish_tf", True))
        self.tf_publish_rate    = float(rospy.get_param("~tf_publish_rate", 10.0))
        self.tf_publish_timeout = float(rospy.get_param("~tf_publish_timeout", 10.0))
        self.marker_frame_prefix = rospy.get_param("~marker_frame_prefix", "aruco_marker_")

        # -------- ArUco detector (OpenCV >= 4.7 API) --------
        self.aruco_dict   = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector     = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        h = self.marker_size / 2.0
        self.obj_pts = np.array([
            [-h,  h, 0.0],
            [ h,  h, 0.0],
            [ h, -h, 0.0],
            [-h, -h, 0.0],
        ], dtype=np.float32)

        # -------- ROS plumbing --------
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self._lock       = threading.Lock()

        self._tf_broadcaster = tf2_ros.TransformBroadcaster() if self.publish_tf else None
        self._tf_cache       = {}
        self._tf_cache_lock  = threading.Lock()
        self._tf_timer       = None
        self._tf_deadline    = None

        self.service = rospy.Service(self.service_name, RobotQuery, self._on_request)
        rospy.loginfo("Marker detection service ready: %s", self.service_name)
        rospy.loginfo("  target_ids=%s  marker_size=%.3fm  base_frame=%s",
                      self.target_ids, self.marker_size, self.base_frame)

    # ------------------------------------------------------------------
    def _on_request(self, _req):
        with self._lock:
            return self._detect()

    def _detect(self):
        resp = RobotQueryResponse()
        resp.result_code = ResultCode()

        try:
            cam_info  = rospy.wait_for_message(self.camera_info_topic, CameraInfo, timeout=self.msg_timeout)
            color_msg = rospy.wait_for_message(self.color_topic,        Image,      timeout=self.msg_timeout)
        except rospy.ROSException as e:
            return self._fail(resp, 2, "Camera topics unavailable: {}".format(e))

        depth_msg = None
        if self.use_depth_position:
            try:
                depth_msg = rospy.wait_for_message(self.depth_topic, Image, timeout=self.msg_timeout)
            except rospy.ROSException:
                rospy.logwarn_throttle(10.0, "Depth image unavailable; using PnP-only translation.")

        K = np.array(cam_info.K, dtype=np.float64).reshape(3, 3)
        D = np.array(cam_info.D, dtype=np.float64).ravel()
        camera_frame = cam_info.header.frame_id or color_msg.header.frame_id

        try:
            color = self._color_msg_to_bgr(color_msg)
        except Exception as e:
            return self._fail(resp, 3, "Color image decode failed: {}".format(e))

        depth = None
        if depth_msg is not None:
            try:
                depth = ros_numpy.numpify(depth_msg)
            except Exception as e:
                rospy.logwarn("Depth image decode failed: %s", e)
                depth = None

        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None or len(ids) == 0:
            resp.result_code.result_code = 0
            resp.result_code.message = "Successfully detected 0 marker(s)"
            resp.data = json.dumps({"markers": []})
            return resp

        ids        = ids.flatten().tolist()
        target_set = {int(x) for x in self.target_ids}
        markers_out = []

        if self.publish_tf:
            with self._tf_cache_lock:
                self._tf_cache.clear()

        for i, mid in enumerate(ids):
            if int(mid) not in target_set:
                continue

            img_pts = corners[i].reshape(-1, 2).astype(np.float32)

            ok, rvec, tvec = cv2.solvePnP(
                self.obj_pts, img_pts, K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            if not ok:
                rospy.logwarn("solvePnP failed for marker id=%d", mid)
                continue
            tvec = np.asarray(tvec, dtype=np.float64).reshape(3)

            # ---- Optional translation refinement using measured depth ----
            if depth is not None:
                cx_px = float(np.mean(img_pts[:, 0]))
                cy_px = float(np.mean(img_pts[:, 1]))
                z_meas = self._sample_depth(depth, depth_msg.encoding, cx_px, cy_px)
                if z_meas is not None:
                    fx, fy     = K[0, 0], K[1, 1]
                    cx_k, cy_k = K[0, 2], K[1, 2]
                    tvec = np.array([
                        (cx_px - cx_k) * z_meas / fx,
                        (cy_px - cy_k) * z_meas / fy,
                        z_meas,
                    ], dtype=np.float64)

            # --- Build rotation from corner geometry, not from solvePnP's rvec ---

            # 1. Get the plane normal (Z axis) from PnP — this part is stable
            R_pnp, _ = cv2.Rodrigues(rvec)
            z_axis_cam = R_pnp[:, 2]  # marker's Z in camera frame, from PnP

            # 2. Get the marker's "up" direction (X axis) from corner geometry.
            #    corners[i] is ordered TL, TR, BR, BL by the ArUco detector.
            #    "Top" of marker = midpoint of TL & TR = (corners[0] + corners[1]) / 2
            #    "Bottom" of marker = midpoint of BL & BR = (corners[2] + corners[3]) / 2
            #    We need this in 3D, in the camera frame.

            # Back-project the 4 corners to 3D using the PnP solution.
            # In the marker's own frame, the corners are self.obj_pts. Transform them
            # into the camera frame via (R_pnp, tvec).
            corners_cam = (R_pnp @ self.obj_pts.T).T + tvec.reshape(1, 3)
            # corners_cam[0]=TL, [1]=TR, [2]=BR, [3]=BL  (matching obj_pts order)

            top_mid_cam    = 0.5 * (corners_cam[0] + corners_cam[1])
            bottom_mid_cam = 0.5 * (corners_cam[2] + corners_cam[3])

            x_axis_cam = top_mid_cam - bottom_mid_cam
            x_axis_cam /= np.linalg.norm(x_axis_cam)

            # 3. Re-orthogonalize: project x onto the plane perpendicular to z
            x_axis_cam = x_axis_cam - np.dot(x_axis_cam, z_axis_cam) * z_axis_cam
            x_axis_cam /= np.linalg.norm(x_axis_cam)

            # 4. Y = Z × X for a right-handed frame
            y_axis_cam = np.cross(z_axis_cam, x_axis_cam)

            # 5. Assemble rotation matrix (columns = axes expressed in camera frame)
            # R_marker_cam = np.column_stack([x_axis_cam, y_axis_cam, z_axis_cam])
            # Z into the marker, X still along height (toward top), Y by right-hand rule
            z_axis_cam = -z_axis_cam
            y_axis_cam = np.cross(z_axis_cam, x_axis_cam)  # recompute Y so frame stays right-handed
            R_marker_cam = np.column_stack([x_axis_cam, y_axis_cam, z_axis_cam])

            T44 = np.eye(4)
            T44[:3, :3] = R_marker_cam
            qx, qy, qz, qw = quaternion_from_matrix(T44)


            # ---- PoseStamped in camera frame ----
            pose_cam = PoseStamped()
            pose_cam.header.frame_id = camera_frame
            pose_cam.header.stamp    = color_msg.header.stamp
            pose_cam.pose.position.x = float(tvec[0])
            pose_cam.pose.position.y = float(tvec[1])
            pose_cam.pose.position.z = float(tvec[2])
            pose_cam.pose.orientation.x = float(qx)
            pose_cam.pose.orientation.y = float(qy)
            pose_cam.pose.orientation.z = float(qz)
            pose_cam.pose.orientation.w = float(qw)

            pose_base_dict = None
            try:
                pose_base = self.tf_buffer.transform(
                    pose_cam, self.base_frame, rospy.Duration(self.tf_timeout)
                )
                pose_base_dict = self._pose_to_dict(pose_base.pose)

                # # flip the marker pose around local X axis so the Z axis points into the marker face, matching the camera-frame convention.
                # # Extract current orientation
                # w = pose_base_dict["orientation"]["w"]
                # x = pose_base_dict["orientation"]["x"]
                # y = pose_base_dict["orientation"]["y"]
                # z = pose_base_dict["orientation"]["z"]

                # dw, dx, dy, dz = 0.0, 0.7071, 0.7071, 0.0

                # pose_base_dict["orientation"]["w"] = dw*w - dx*x - dy*y - dz*z
                # pose_base_dict["orientation"]["x"] = dw*x + dx*w + dy*z - dz*y
                # pose_base_dict["orientation"]["y"] = dw*y - dx*z + dy*w + dz*x
                # pose_base_dict["orientation"]["z"] = dw*z + dx*y - dy*x + dz*w

            except (tf2_ros.LookupException,
                    tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException) as e:
                rospy.logwarn("TF %s -> %s failed for id=%d: %s",
                              camera_frame, self.base_frame, mid, e)
            
            # ---- Cache TF for RViz visualisation ----
            if self.publish_tf:
                self._cache_marker_tf(int(mid), self.base_frame, [pose_base_dict["position"]["x"], pose_base_dict["position"]["y"], pose_base_dict["position"]["z"]], (pose_base_dict["orientation"]["x"], pose_base_dict["orientation"]["y"], pose_base_dict["orientation"]["z"], pose_base_dict["orientation"]["w"]))
                # rospy.loginfo("Detected marker id=%d at (%.3f, %.3f, %.3f) m in camera frame",
                #               mid, tvec[0], tvec[1], tvec[2])
            
            rospy.loginfo(
                f"Detected marker id={mid} at ("
                f"{pose_base_dict['position']['x']:.4f}, "
                f"{pose_base_dict['position']['y']:.4f}, "
                f"{pose_base_dict['position']['z']:.4f}) m in '{self.base_frame}' frame with orientation ("
                f"{pose_base_dict['orientation']['x']:.4f}, "
                f"{pose_base_dict['orientation']['y']:.4f}, "
                f"{pose_base_dict['orientation']['z']:.4f}, "
                f"{pose_base_dict['orientation']['w']:.4f})"
            )
            
            # try:
            #     shift_m = 0.18
            #     Q = [pose_base_dict["orientation"]["x"],
            #         pose_base_dict["orientation"]["y"],
            #         pose_base_dict["orientation"]["z"],
            #         pose_base_dict["orientation"]["w"]]
            #     R = quaternion_matrix(Q)[0:3, 0:3]
            #     shift_global = R.dot(np.array([0.0, 0.0, -shift_m]))

            #     t_shift_x = float(pose_base_dict["position"]["x"] + shift_global[0])
            #     t_shift_y = float(pose_base_dict["position"]["y"] + shift_global[1])
            #     t_shift_z = float(pose_base_dict["position"]["z"] + shift_global[2])

            #     # Still publish the TF for RViz visualization
            #     # if self.publish_tf:
            #     #     shifted_tf = TransformStamped()
            #     #     shifted_tf.header.stamp = rospy.Time.now()
            #     #     shifted_tf.header.frame_id = self.base_frame
            #     #     shifted_tf.child_frame_id = "shifted_aruco_marker_{}".format(mid)
            #     #     shifted_tf.transform.translation.x = t_shift_x
            #     #     shifted_tf.transform.translation.y = t_shift_y
            #     #     shifted_tf.transform.translation.z = t_shift_z
            #     #     shifted_tf.transform.rotation.x = pose_base_dict["orientation"]["x"]
            #     #     shifted_tf.transform.rotation.y = pose_base_dict["orientation"]["y"]
            #     #     shifted_tf.transform.rotation.z = pose_base_dict["orientation"]["z"]
            #     #     shifted_tf.transform.rotation.w = pose_base_dict["orientation"]["w"]
            #     #     self._tf_broadcaster.sendTransform(shifted_tf)

            #     rospy.loginfo(
            #         f"Shifted aruco marker pose in '{self.base_frame}' | "
            #         f"xyz=({t_shift_x:.4f}, {t_shift_y:.4f}, {t_shift_z:.4f})  "
            #         f"quat=({pose_base_dict['orientation']['x']:.4f}, "
            #         f"{pose_base_dict['orientation']['y']:.4f}, "
            #         f"{pose_base_dict['orientation']['z']:.4f}, "
            #         f"{pose_base_dict['orientation']['w']:.4f})"
            #     )

            #     pose_base_dict["position"]["x"] = t_shift_x
            #     pose_base_dict["position"]["y"] = t_shift_y
            #     pose_base_dict["position"]["z"] = t_shift_z
            #     # orientation unchanged

            # except Exception as e:
            #     rospy.logwarn(f"Failed to compute shifted aruco marker pose: {e}")

            markers_out.append({
                "id": int(mid),
                "pose_wrt_camera":    self._pose_to_dict(pose_cam.pose),
                "pose_wrt_base_link": pose_base_dict,
            })

        resp.result_code.result_code = 0
        resp.result_code.message     = "Successfully detected {} marker(s)".format(len(markers_out))
        resp.data                    = json.dumps({"markers": markers_out})

        if self.publish_tf and markers_out:
            self._start_or_refresh_tf_publishing()

        return resp

    # ------------------------------------------------------------------
    def _cache_marker_tf(self, marker_id, parent_frame, tvec, quat_xyzw):
        """Build/update a TransformStamped for this marker in the TF cache."""
        t = TransformStamped()
        t.header.frame_id = parent_frame
        t.child_frame_id  = "{}{}".format(self.marker_frame_prefix, marker_id)
        t.transform.translation.x = float(tvec[0])
        t.transform.translation.y = float(tvec[1])
        t.transform.translation.z = float(tvec[2])
        t.transform.rotation.x = float(quat_xyzw[0])
        t.transform.rotation.y = float(quat_xyzw[1])
        t.transform.rotation.z = float(quat_xyzw[2])
        t.transform.rotation.w = float(quat_xyzw[3])
        with self._tf_cache_lock:
            self._tf_cache[marker_id] = t

    def _start_or_refresh_tf_publishing(self):
        """(Re)start the publish timer with a fresh deadline.

        Called on every successful detection. If the timer is already running
        it just bumps the deadline; otherwise it spins up a new periodic timer.
        """
        self._tf_deadline = rospy.Time.now() + rospy.Duration(self.tf_publish_timeout)
        if self._tf_timer is None and self.tf_publish_rate > 0.0:
            self._tf_timer = rospy.Timer(
                rospy.Duration(1.0 / self.tf_publish_rate),
                self._republish_tf,
            )
            rospy.loginfo(
                "TF publishing started; will stop after %.1fs without new detections.",
                self.tf_publish_timeout,
            )

    def _republish_tf(self, _evt):
        """Re-stamp and broadcast cached marker transforms; self-cancel on timeout."""
        # Timeout reached -> stop the timer and clear the cache.
        if self._tf_deadline is None or rospy.Time.now() > self._tf_deadline:
            timer = self._tf_timer
            self._tf_timer    = None
            self._tf_deadline = None
            with self._tf_cache_lock:
                self._tf_cache.clear()
            rospy.loginfo("TF publishing for markers stopped (timeout).")
            if timer is not None:
                # Safe to call from inside the timer's own callback in rospy.
                timer.shutdown()
            return

        if self._tf_broadcaster is None:
            return
        with self._tf_cache_lock:
            if not self._tf_cache:
                return
            now = rospy.Time.now()
            transforms = []
            for tfm in self._tf_cache.values():
                tfm.header.stamp = now
                transforms.append(tfm)
        self._tf_broadcaster.sendTransform(transforms)

    # ------------------------------------------------------------------
    @staticmethod
    def _color_msg_to_bgr(msg):
        """Decode a sensor_msgs/Image (any common color encoding) to a BGR uint8 array."""
        arr = ros_numpy.numpify(msg)
        enc = (msg.encoding or "").lower()
        if enc in ("bgr8",):
            return arr
        if enc in ("rgb8",):
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        if enc in ("rgba8",):
            return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        if enc in ("bgra8",):
            return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
        if enc in ("mono8", "8uc1"):
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        # Fall back: assume the array is already in a usable layout.
        rospy.logwarn_throttle(10.0, "Unrecognised color encoding '%s', using as-is.", msg.encoding)
        return arr

    @staticmethod
    def _pose_to_dict(pose):
        return {
            "position": {
                "x": pose.position.x,
                "y": pose.position.y,
                "z": pose.position.z,
            },
            "orientation": {
                "x": pose.orientation.x,
                "y": pose.orientation.y,
                "z": pose.orientation.z,
                "w": pose.orientation.w,
            },
        }

    @staticmethod
    def _sample_depth(depth_img, encoding, u, v, win=2):
        """Median of a (2*win+1)^2 patch around (u, v). Returns metres or None."""
        h, w = depth_img.shape[:2]
        ui, vi = int(round(u)), int(round(v))
        if not (0 <= ui < w and 0 <= vi < h):
            return None
        u0, u1 = max(0, ui - win), min(w, ui + win + 1)
        v0, v1 = max(0, vi - win), min(h, vi + win + 1)
        patch = depth_img[v0:v1, u0:u1].astype(np.float32)
        valid = patch[(patch > 0) & np.isfinite(patch)]
        if valid.size == 0:
            return None
        med = float(np.median(valid))
        # RealSense aligned_depth_to_color is published as 16UC1 in millimetres.
        if encoding in ("16UC1", "mono16"):
            med /= 1000.0
        return med

    @staticmethod
    def _fail(resp, code, message):
        resp.result_code.result_code = int(code)
        resp.result_code.message     = message
        resp.data                    = json.dumps({"markers": []})
        rospy.logerr(message)
        return resp


def main():
    rospy.init_node("marker_detection_service", anonymous=False)
    MarkerDetectionService()
    rospy.spin()


if __name__ == "__main__":
    main()