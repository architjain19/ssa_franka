#!/usr/bin/env python3
"""
ROS1 Noetic service node for ArUco marker detection.

Service
-------
  /robot/perception/detect_markers   (robot_api_interfaces/RobotCommand)

Behaviour
---------
On each request the node grabs one fresh frame from the requested camera
("scene" or "wrist"):
  * <cam>_color_topic       (RGB)
  * <cam>_camera_info_topic (intrinsics)
  * <cam>_depth_topic       (optional, for depth-refined translation)

It detects ArUco markers from the DICT_4X4_250 dictionary, filters them by
a configurable target-id list, computes the 6-DoF pose of each marker
centre, and returns the result as a JSON string in the `data` field of the
response.

Request (JSON string in .req field):
{
    "camera": "scene"        # "wrist" or "scene" (default: ~default_camera)
}

Response (JSON string in .data field):
{
  "camera": "scene",
  "markers": [
    {
      "id": <int>,
      "pose_wrt_camera":    {"position": {x,y,z}, "orientation": {x,y,z,w}},
      "pose_wrt_base_link": {"position": {x,y,z}, "orientation": {x,y,z,w}}  // null if TF fails
    },
    ...
  ]
}

ROS1 usage:
    rosrun franka_robot_apis marker_detection_service.py

    # Scene camera (default)
    rosservice call /robot/perception/detect_markers \\
        '{"req": "{\\"camera\\": \\"scene\\"}"}'

    # Wrist camera
    rosservice call /robot/perception/detect_markers \\
        '{"req": "{\\"camera\\": \\"wrist\\"}"}'

Parameters (all private, with sensible defaults)
------------------------------------------------
  ~service_name              (str)   service name to advertise
  ~default_camera            (str)   "wrist" or "scene"  (default "scene")
  ~scene_color_topic         (str)   scene RGB image topic
  ~scene_camera_info_topic   (str)   scene color camera_info topic
  ~scene_depth_topic         (str)   scene aligned depth-to-color image topic
  ~wrist_color_topic         (str)   wrist RGB image topic
  ~wrist_camera_info_topic   (str)   wrist color camera_info topic
  ~wrist_depth_topic         (str)   wrist aligned depth-to-color image topic
  ~marker_size               (float) marker edge length in metres (default 0.08)
  ~target_ids                (int[]) marker ids of interest (default [1,2,3,4,5])
  ~base_frame                (str)   robot base TF frame (default "panda_link0")
  ~msg_timeout               (float) seconds to wait for camera messages (default 2.0)
  ~tf_timeout                (float) seconds to wait for TF (default 1.0)
  ~use_depth_position        (bool)  refine translation with measured depth (default True)
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

from robot_api_interfaces.srv import RobotCommand, RobotCommandResponse
from robot_api_interfaces.msg import ResultCode


# Allowed values for the "camera" field in the request JSON.
VALID_CAMERAS = ("wrist", "scene")


# ---------------------------------------------------------------------------
# Per-camera topic bundle
# ---------------------------------------------------------------------------
class CameraConfig:
    """
    Holds the topic names for a single camera. Frames are pulled
    synchronously via rospy.wait_for_message on each request, so we do not
    need long-running subscribers or a lock here — just the configuration.
    """

    def __init__(self, name, color_topic, camera_info_topic, depth_topic):
        self.name              = name
        self.color_topic       = color_topic
        self.camera_info_topic = camera_info_topic
        self.depth_topic       = depth_topic


class MarkerDetectionService:
    def __init__(self):
        # -------- Parameters --------
        self.service_name = rospy.get_param("~service_name", "/robot/perception/detect_markers")

        # --- Scene camera topics ---
        scene_color  = rospy.get_param("~scene_color_topic",       "/zed/scene/color/image_raw")
        scene_info   = rospy.get_param("~scene_camera_info_topic", "/zed/scene/color/camera_info")
        scene_depth  = rospy.get_param("~scene_depth_topic",       "/zed/scene/aligned_depth_to_color/image_raw")

        # --- Wrist camera topics ---
        wrist_color  = rospy.get_param("~wrist_color_topic",       "/zed/wrist/color/image_raw")
        wrist_info   = rospy.get_param("~wrist_camera_info_topic", "/zed/wrist/color/camera_info")
        wrist_depth  = rospy.get_param("~wrist_depth_topic",       "/zed/wrist/aligned_depth_to_color/image_raw")

        # Default camera when the request does not specify one.
        self.default_camera = rospy.get_param("~default_camera", "scene").lower()
        if self.default_camera not in VALID_CAMERAS:
            rospy.logwarn(
                "~default_camera='%s' is invalid; falling back to 'scene'. Valid options: %s",
                self.default_camera, VALID_CAMERAS,
            )
            self.default_camera = "scene"

        self.cameras = {
            "scene": CameraConfig("scene", scene_color, scene_info, scene_depth),
            "wrist": CameraConfig("wrist", wrist_color, wrist_info, wrist_depth),
        }

        self.marker_size         = float(rospy.get_param("~marker_size",  0.08))
        self.target_ids          = list(rospy.get_param("~target_ids",    [1, 2, 3, 4, 5]))
        self.base_frame          = rospy.get_param("~base_frame",         "panda_link0")
        self.msg_timeout         = float(rospy.get_param("~msg_timeout",  2.0))
        self.tf_timeout          = float(rospy.get_param("~tf_timeout",   1.0))
        self.use_depth_position  = bool(rospy.get_param("~use_depth_position", True))
        self.publish_tf          = bool(rospy.get_param("~publish_tf", True))
        self.tf_publish_rate     = float(rospy.get_param("~tf_publish_rate", 10.0))
        self.tf_publish_timeout  = float(rospy.get_param("~tf_publish_timeout", 10.0))
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

        self.service = rospy.Service(self.service_name, RobotCommand, self._on_request)
        rospy.loginfo("Marker detection service ready: %s", self.service_name)
        rospy.loginfo("  default_camera=%s  target_ids=%s  marker_size=%.3fm  base_frame=%s",
                      self.default_camera, self.target_ids, self.marker_size, self.base_frame)
        rospy.loginfo("  scene: color=%s info=%s depth=%s",
                      scene_color, scene_info, scene_depth)
        rospy.loginfo("  wrist: color=%s info=%s depth=%s",
                      wrist_color, wrist_info, wrist_depth)

    # ------------------------------------------------------------------
    def _on_request(self, request):
        """
        rospy.Service callback — called in a dedicated thread per request.

        Parses .req as JSON, picks the camera, and runs detection.
        """
        rospy.loginfo("detect_markers request received: %s", request.req)
        resp = RobotCommandResponse()
        resp.result_code = ResultCode()

        # --- Parse request JSON ---
        # Empty .req is fine — use the default camera.
        raw = (request.req or "").strip()
        if raw:
            try:
                req_data = json.loads(raw)
            except (json.JSONDecodeError, ValueError) as e:
                return self._fail(resp, ResultCode.FAILURE, f"Bad request JSON: {e}")
            if not isinstance(req_data, dict):
                return self._fail(resp, ResultCode.FAILURE,
                                  "Request JSON must be an object.")
        else:
            req_data = {}

        # --- Resolve which camera to use ---
        camera = req_data.get("camera", self.default_camera)
        if not isinstance(camera, str):
            return self._fail(resp, ResultCode.FAILURE,
                              f"'camera' must be a string, got {type(camera).__name__}.")
        camera = camera.strip().lower()
        if camera not in self.cameras:
            return self._fail(resp, ResultCode.FAILURE,
                              f"Invalid 'camera' value: '{camera}'. "
                              f"Must be one of: {list(self.cameras.keys())}.")

        with self._lock:
            return self._detect(self.cameras[camera])

    def _detect(self, cam):
        """
        Run a one-shot detection on the given camera.

        Args:
            cam (CameraConfig): which camera's topics to read from.
        """
        resp = RobotCommandResponse()
        resp.result_code = ResultCode()

        rospy.loginfo("Detecting markers on camera='%s'", cam.name)

        try:
            cam_info  = rospy.wait_for_message(cam.camera_info_topic, CameraInfo, timeout=self.msg_timeout)
            color_msg = rospy.wait_for_message(cam.color_topic,        Image,      timeout=self.msg_timeout)
        except rospy.ROSException as e:
            return self._fail(resp, ResultCode.FAILURE,
                              "Camera topics unavailable for '{}': {}".format(cam.name, e),
                              camera_name=cam.name)

        depth_msg = None
        if self.use_depth_position:
            try:
                depth_msg = rospy.wait_for_message(cam.depth_topic, Image, timeout=self.msg_timeout)
            except rospy.ROSException:
                rospy.logwarn_throttle(10.0, "Depth image unavailable for '%s'; using PnP-only translation.", cam.name)

        K = np.array(cam_info.K, dtype=np.float64).reshape(3, 3)
        D = np.array(cam_info.D, dtype=np.float64).ravel()
        camera_frame = cam_info.header.frame_id or color_msg.header.frame_id

        try:
            color = self._color_msg_to_bgr(color_msg)
        except Exception as e:
            return self._fail(resp, ResultCode.FAILURE,
                              "Color image decode failed: {}".format(e),
                              camera_name=cam.name)

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
            resp.result_code.result_code = ResultCode.SUCCESS
            resp.result_code.message = "Successfully detected 0 marker(s)"
            resp.data = json.dumps({"camera": cam.name, "markers": []})
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
            corners_cam = (R_pnp @ self.obj_pts.T).T + tvec.reshape(1, 3)

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
            # Z into the marker, X still along height (toward top), Y by right-hand rule
            z_axis_cam = -z_axis_cam
            y_axis_cam = np.cross(z_axis_cam, x_axis_cam)
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

            except (tf2_ros.LookupException,
                    tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException) as e:
                rospy.logwarn("TF %s -> %s failed for id=%d: %s",
                              camera_frame, self.base_frame, mid, e)

            # ---- Cache TF for RViz visualisation ----
            if self.publish_tf and pose_base_dict is not None:
                self._cache_marker_tf(
                    int(mid), self.base_frame,
                    [pose_base_dict["position"]["x"],
                     pose_base_dict["position"]["y"],
                     pose_base_dict["position"]["z"]],
                    (pose_base_dict["orientation"]["x"],
                     pose_base_dict["orientation"]["y"],
                     pose_base_dict["orientation"]["z"],
                     pose_base_dict["orientation"]["w"]),
                )

            if pose_base_dict is not None:
                rospy.loginfo(
                    f"Detected marker id={mid} ({cam.name}) at ("
                    f"{pose_base_dict['position']['x']:.4f}, "
                    f"{pose_base_dict['position']['y']:.4f}, "
                    f"{pose_base_dict['position']['z']:.4f}) m in '{self.base_frame}' frame with orientation ("
                    f"{pose_base_dict['orientation']['x']:.4f}, "
                    f"{pose_base_dict['orientation']['y']:.4f}, "
                    f"{pose_base_dict['orientation']['z']:.4f}, "
                    f"{pose_base_dict['orientation']['w']:.4f})"
                )
            else:
                rospy.loginfo(
                    "Detected marker id=%d (%s) at (%.4f, %.4f, %.4f) m in camera frame '%s' "
                    "(no base-frame TF)",
                    mid, cam.name, tvec[0], tvec[1], tvec[2], camera_frame,
                )

            markers_out.append({
                "id": int(mid),
                "pose_wrt_camera":    self._pose_to_dict(pose_cam.pose),
                "pose_wrt_base_link": pose_base_dict,
            })

        resp.result_code.result_code = ResultCode.SUCCESS
        resp.result_code.message     = "Successfully detected {} marker(s)".format(len(markers_out))
        resp.data                    = json.dumps({"camera": cam.name, "markers": markers_out})

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
        """(Re)start the publish timer with a fresh deadline."""
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
        if self._tf_deadline is None or rospy.Time.now() > self._tf_deadline:
            timer = self._tf_timer
            self._tf_timer    = None
            self._tf_deadline = None
            with self._tf_cache_lock:
                self._tf_cache.clear()
            rospy.loginfo("TF publishing for markers stopped (timeout).")
            if timer is not None:
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
    def _fail(resp, code, message, camera_name=None):
        """
        Populate *resp* as a failure and log the error.

        Args:
            resp (RobotCommandResponse): to mutate
            code (int):                  result_code value (use ResultCode constants)
            message (str):               human-readable error
            camera_name (str | None):    echoed back in the JSON if known
        """
        resp.result_code.result_code = int(code)
        resp.result_code.message     = message
        body = {"status": "error", "message": message, "markers": []}
        if camera_name is not None:
            body["camera"] = camera_name
        resp.data = json.dumps(body)
        rospy.logerr(message)
        return resp


def main():
    rospy.init_node("marker_detection_service", anonymous=False)
    MarkerDetectionService()
    rospy.spin()


if __name__ == "__main__":
    main()