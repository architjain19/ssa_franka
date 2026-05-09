#!/usr/bin/env python3
"""
ROS1 Noetic service node: Grounded SAM2 + AnyGrasp Pipeline
-------------------------------------------------------------

Service: /robot/perception/detect_objects  (robot_api_interfaces/RobotCommand)

Request (JSON string in .req field):
{
    "text_prompt": "glass"
}

Response (JSON string in .data field):
{
    "status": "success",
    "best_grasp": {
        "translation_wrt_base": [x, y, z],
        "quaternion_wrt_base": {"x": ..., "y": ..., "z": ..., "w": ...},
        "score": 0.2,
        "width": 0.047
    }
}

ROS1 usage:
    rosrun <your_pkg> detect_objects_service_node_ros1.py

    rosservice call /robot/perception/detect_objects \
        '{"req": "{\"text_prompt\": \"floor\"}"}'
"""

import json
import asyncio
import base64
import threading
import time
import traceback

import cv2
import numpy as np
import aiohttp

import rospy
import tf
import tf.transformations as tft

from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry
import ros_numpy

from robot_api_interfaces.srv import RobotCommand, RobotCommandResponse
from robot_api_interfaces.msg import ResultCode


# ---------------------------------------------------------------------------
# Helper: rotquaternion (Shepperd)
# ---------------------------------------------------------------------------
def _rotation_matrix_to_quaternion(R):
    """
    Convert a 3x3 rotation matrix to a quaternion dict {x, y, z, w}.

    Args:
        R (list | np.ndarray): 3x3 rotation matrix

    Returns:
        dict: {"x": float, "y": float, "z": float, "w": float}
    """
    R = np.array(R, dtype=np.float64)
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return {"x": float(x), "y": float(y), "z": float(z), "w": float(w)}


# ---------------------------------------------------------------------------
# Helper: quaternion multiply  q_out = q_a * q_b  (Hamilton product)
# ---------------------------------------------------------------------------
def _quaternion_multiply(q_a, q_b):
    """
    Hamilton product of two quaternion dicts {x, y, z, w}.

    Args:
        q_a (dict): first quaternion
        q_b (dict): second quaternion

    Returns:
        dict: composed quaternion
    """
    ax, ay, az, aw = q_a["x"], q_a["y"], q_a["z"], q_a["w"]
    bx, by, bz, bw = q_b["x"], q_b["y"], q_b["z"], q_b["w"]
    return {
        "x": float(aw * bx + ax * bw + ay * bz - az * by),
        "y": float(aw * by - ax * bz + ay * bw + az * bx),
        "z": float(aw * bz + ax * by - ay * bx + az * bw),
        "w": float(aw * bw - ax * bx - ay * by - az * bz),
    }


# ---------------------------------------------------------------------------
# Main service class
# ---------------------------------------------------------------------------
class SegmentAndGraspNode:
    """
    ROS1 service node that chains:
      1. Grounded SAM segments the object and writes NPZ
      2. AnyGrasp Web reads NPZ, returns best grasp pose
      3. TF transform converts grasp pose from camera frame to panda_link0
    """

    def __init__(self):
        # ------------------------------------------------------------------ #
        #  Parameters  (declare rospy.get_param with ~).                     #
        # ------------------------------------------------------------------ #
        self.sam2_url      = rospy.get_param("~sam2_url",     "ws://10.158.54.164:8766/detect_objects")
        self.anygrasp_url  = rospy.get_param("~anygrasp_url", "ws://10.158.54.164:8767/get_saved_grasp")

        # Camera topics — updated to the robot's actual RealSense topics
        self.rgb_topic         = rospy.get_param("~rgb_topic",         "/realsense/scene/color/image_raw")
        self.depth_topic       = rospy.get_param("~depth_topic",       "/realsense/scene/aligned_depth_to_color/image_raw")
        self.camera_info_topic = rospy.get_param("~camera_info_topic", "/realsense/scene/aligned_depth_to_color/camera_info")

        # Fallback intrinsics (used only if camera_info never arrives)
        self.fx_default = float(rospy.get_param("~fx", 752.0038452148438))
        self.fy_default = float(rospy.get_param("~fy", 751.7178344726562))
        self.cx_default = float(rospy.get_param("~cx", 628.4379272460938))
        self.cy_default = float(rospy.get_param("~cy", 335.1157531738281))

        # Timeouts
        self.camera_wait_sec   = float(rospy.get_param("~camera_wait_sec",   1.0))
        self.sam2_timeout_sec  = float(rospy.get_param("~sam2_timeout_sec",  15.0))
        self.grasp_timeout_sec = float(rospy.get_param("~grasp_timeout_sec", 15.0))

        # TF parameters
        self.camera_frame  = rospy.get_param("~camera_frame",   "cam_scene_color_optical_frame")
        self.base_frame    = rospy.get_param("~base_frame",     "panda_link0")
        self.tf_timeout_sec= float(rospy.get_param("~tf_timeout_sec", 2.0))

        # ------------------------------------------------------------------ #
        #  Camera state                                                        #
        # ------------------------------------------------------------------ #
        self.latest_rgb           = None   # np.ndarray uint8  (H,W,3) BGR
        self.latest_depth         = None   # np.ndarray uint16 (H,W)
        self.fx                   = self.fx_default
        self.fy                   = self.fy_default
        self.cx                   = self.cx_default
        self.cy                   = self.cy_default
        self.camera_info_received = False
        self._cam_lock            = threading.Lock()

        # ------------------------------------------------------------------ #
        #  TF listener  (ros1 tf.TransformListener)                          #
        # ------------------------------------------------------------------ #
        self._tf_listener = tf.TransformListener()
        # TF broadcaster for shifted grasp pose
        self._tf_broadcaster = tf.TransformBroadcaster()

        # ------------------------------------------------------------------ #
        #  Subscriptions                                                       #
        # ------------------------------------------------------------------ #
        rospy.Subscriber(
            self.rgb_topic, Image, self._rgb_cb,
            queue_size=1, buff_size=2 ** 24,
        )
        rospy.Subscriber(
            self.depth_topic, Image, self._depth_cb,
            queue_size=1, buff_size=2 ** 24,
        )
        # CameraInfo is typically latched — reliable, small message
        rospy.Subscriber(
            self.camera_info_topic, CameraInfo, self._camera_info_cb,
            queue_size=10,
        )

        # ------------------------------------------------------------------ #
        #  Service                                                             #
        # ------------------------------------------------------------------ #
        self._service = rospy.Service(
            "/robot/perception/detect_objects",
            RobotCommand,
            self._handle_request,
        )

        rospy.loginfo(
            "\nSegmentAndGraspNode (ROS1) ready.\n"
            f"  Service     : /robot/perception/detect_objects\n"
            f"  SAM2        : {self.sam2_url}\n"
            f"  AnyGrasp    : {self.anygrasp_url}\n"
            f"  RGB         : {self.rgb_topic}\n"
            f"  Depth       : {self.depth_topic}\n"
            f"  CamInfo     : {self.camera_info_topic}\n"
            f"  Camera frame: {self.camera_frame}\n"
            f"  Base frame  : {self.base_frame}"
        )

    # ------------------------------------------------------------------ #
    #  Camera callbacks                                                    #
    # ------------------------------------------------------------------ #

    def _rgb_cb(self, msg):
        try:
            # ros_numpy.numpify returns (H,W,3) uint8 in RGB order for rgb8,
            # BGR for bgr8.  Converting to BGR for OpenCV downstream.
            img = ros_numpy.numpify(msg)          # (H,W,3) or (H,W,4) uint8
            enc = msg.encoding.lower()
            if enc in ("rgb8",):
                img = img[..., ::-1].copy()       # RGB -> BGR
            elif enc in ("rgba8",):
                img = img[..., :3][..., ::-1].copy()   # RGBA -> BGR
            elif enc in ("bgra8",):
                img = img[..., :3].copy()         # drop alpha, already BGR
            # bgr8 -> already correct; mono8/16 -> grayscale, left as-is
            with self._cam_lock:
                self.latest_rgb = img
        except Exception as e:
            rospy.logwarn(f"RGB decode error: {e}")

    def _depth_cb(self, msg):
        try:
            # ros_numpy.numpify returns (H,W) for mono16 / 16UC1 depth images.
            # Values are raw millimetres as uint16 — exactly what we need.
            img = ros_numpy.numpify(msg)          # (H,W) typically uint16
            if img.dtype != np.uint16:
                img = img.astype(np.uint16)
            with self._cam_lock:
                self.latest_depth = img
        except Exception as e:
            rospy.logwarn(f"Depth decode error: {e}")

    def _camera_info_cb(self, msg):
        # K = [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        with self._cam_lock:
            if not self.camera_info_received:
                self.fx = float(msg.K[0])
                self.fy = float(msg.K[4])
                self.cx = float(msg.K[2])
                self.cy = float(msg.K[5])
                self.camera_info_received = True
                rospy.loginfo(
                    f"Camera intrinsics received: "
                    f"fx={self.fx:.2f} fy={self.fy:.2f} "
                    f"cx={self.cx:.2f} cy={self.cy:.2f}"
                )

    # ------------------------------------------------------------------ #
    #  Service handler                                                     #
    # ------------------------------------------------------------------ #

    def _handle_request(self, request):
        """
        rospy.Service callback — called in a dedicated thread per request.

        Args:
            request (RobotCommand.Request): .req holds the JSON string

        Returns:
            RobotCommandResponse
        """
        rospy.loginfo(f"Service request received: {request.req}")
        response = RobotCommandResponse()

        # --- 1. Parse request -------------------------------------------
        try:
            req_data    = json.loads(request.req)
            text_prompt = req_data.get("text_prompt", "").strip() or "object"
        except (json.JSONDecodeError, ValueError) as e:
            return self._fail(response, f"Bad request: {e}")

        # --- 2. Wait for camera frames ----------------------------------
        rospy.loginfo(f"Waiting up to {self.camera_wait_sec}s for camera frames ...")
        deadline = time.time() + self.camera_wait_sec
        has_frames = False
        while time.time() < deadline:
            with self._cam_lock:
                has_frames = (self.latest_rgb is not None and
                              self.latest_depth is not None)
            if has_frames:
                break
            time.sleep(0.05)

        if not has_frames:
            return self._fail(
                response,
                f"Timed out waiting for camera frames on "
                f"'{self.rgb_topic}' / '{self.depth_topic}'."
            )

        with self._cam_lock:
            rgb   = self.latest_rgb.copy()
            depth = self.latest_depth.copy()
            fx, fy, cx, cy = self.fx, self.fy, self.cx, self.cy

        if not self.camera_info_received:
            rospy.logwarn(
                "CameraInfo not yet received — using fallback intrinsics. "
                f"fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}"
            )

        # --- 3. Call Grounded SAM2 --------------------------------------
        rospy.loginfo(f"Calling Grounded SAM2 | prompt='{text_prompt}'")
        sam2_result, sam2_err = self._call_sam2(
            rgb, depth, fx, fy, cx, cy, text_prompt
        )

        if sam2_err:
            return self._fail(response, f"Grounded SAM2 failed: {sam2_err}")

        if sam2_result.get("status") != "success":
            return self._fail(
                response,
                f"Grounded SAM2 returned non-success status: "
                f"{sam2_result.get('message', json.dumps(sam2_result))}"
            )

        rospy.loginfo("SAM2 succeeded — mask written to NPZ.")

        # --- 4. Call AnyGrasp -------------------------------------------
        rospy.loginfo("Calling AnyGrasp ...")
        grasp_result, grasp_err = self._call_anygrasp()

        if grasp_err:
            return self._fail(response, f"AnyGrasp failed: {grasp_err}")

        if grasp_result.get("status") != "success":
            return self._fail(
                response,
                f"AnyGrasp returned non-success status: "
                f"{grasp_result.get('message', json.dumps(grasp_result))}"
            )

        # --- 5. Transform grasp pose from camera frame to base_link -----
        best = grasp_result["best_grasp"]
        t    = best["translation"]
        q    = _rotation_matrix_to_quaternion(best["rotation"])

        rospy.loginfo(
            f"Grasp found | score={best['score']:.4f}  width={best['width']:.4f}m  "
            f"xyz=({t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f})  "
            f"quat=({q['x']:.4f}, {q['y']:.4f}, {q['z']:.4f}, {q['w']:.4f})"
        )

        t_base, q_base, tf_err = self._transform_pose_to_base(t, q)

        if tf_err:
            rospy.logwarn(
                f"TF transform to '{self.base_frame}' failed: {tf_err}. "
                "Returning camera-frame pose only."
            )
            t_base = None
            q_base = None
        else:
            # Post-multiply by 180° around X to fix AnyGrasp EE convention
            # q_180x = {"x": 1.0, "y": 0.0, "z": 0.0, "w": 0.0}
            # q_base = _quaternion_multiply(q_base, q_180x)
            # Also rotate 180° around Z to adjust end-effector orientation
            q_180z = {"x": 0.0, "y": 0.0, "z": 1.0, "w": 0.0}
            q_base = _quaternion_multiply(q_base, q_180z)
            # Then rotate 90° around -Y to align gripper approach with +Z in base frame
            q_90yz = {"x": 0.0, "y": -0.7071068, "z": 0.0, "w": 0.7071068}
            q_base = _quaternion_multiply(q_base, q_90yz)
            # q__180z = {"x": 0.0, "y": 0.0, "z": 1.0, "w": 0.0}
            # q_base = _quaternion_multiply(q_base, q__180z)

            # # Apply a backward shift of 0.17m along the grasp pose's local X
            # try:
            #     shift_m = 0.18
            #     # rotation matrix from base-frame quaternion
            #     Q = [q_base['x'], q_base['y'], q_base['z'], q_base['w']]
            #     R = tft.quaternion_matrix(Q)[0:3, 0:3]
            #     # local backward along +Z -> negative Z in local coordinates
            #     shift_global = R.dot(np.array([0.0, 0.0, -shift_m]))
            #     t_shift = [
            #         float(t_base[0] + shift_global[0]),
            #         float(t_base[1] + shift_global[1]),
            #         float(t_base[2] + shift_global[2]),
            #     ]

            #     # Broadcast the shifted pose as frame 'shifted_grasp' (parent = base_frame)
            #     now = rospy.Time.now()
            #     self._tf_broadcaster.sendTransform(
            #         (t_shift[0], t_shift[1], t_shift[2]),
            #         (q_base['x'], q_base['y'], q_base['z'], q_base['w']),
            #         now,
            #         "shifted_grasp",
            #         self.base_frame,
            #     )

            #     # small pause to allow TF to propagate, then lookup shifted pose
            #     rospy.sleep(0.05)
            #     self._tf_listener.waitForTransform(
            #         self.base_frame,
            #         "shifted_grasp",
            #         rospy.Time(0),
            #         rospy.Duration(self.tf_timeout_sec),
            #     )
            #     trans_s, rot_s = self._tf_listener.lookupTransform(
            #         self.base_frame,
            #         "shifted_grasp",
            #         rospy.Time(0),
            #     )

            #     t_base = [float(trans_s[0]), float(trans_s[1]), float(trans_s[2])]
            #     q_base = {"x": float(rot_s[0]), "y": float(rot_s[1]),
            #               "z": float(rot_s[2]), "w": float(rot_s[3])}

            #     rospy.loginfo(
            #         f"Shifted pose in '{self.base_frame}' | "
            #         f"xyz=({t_base[0]:.4f}, {t_base[1]:.4f}, {t_base[2]:.4f})  "
            #         f"quat=({q_base['x']:.4f}, {q_base['y']:.4f}, "
            #         f"{q_base['z']:.4f}, {q_base['w']:.4f})"
            #     )

            # except Exception as e:
            #     rospy.logwarn(f"Failed to apply/lookup shifted grasp TF: {e}")
            #     # keep original t_base/q_base if shifting fails

        rospy.loginfo(
            f"Shifted pose in '{self.base_frame}' | "
            f"xyz=({t_base[0]:.4f}, {t_base[1]:.4f}, {t_base[2]:.4f})  "
            f"quat=({q_base['x']:.4f}, {q_base['y']:.4f}, "
            f"{q_base['z']:.4f}, {q_base['w']:.4f})"
        )

        # --- 6. Build and return response --------------------------------
        payload = {
            "status": "success",
            "best_grasp": {
                "translation_wrt_base": t_base,
                "quaternion_wrt_base":  q_base,
                "score":  best["score"],
                "width":  best["width"],
            },
        }

        response.result_code.result_code = ResultCode.SUCCESS
        response.result_code.message     = "Segmentation and grasp detection succeeded."
        response.data                    = json.dumps(payload)
        return response

    # ------------------------------------------------------------------ #
    #  TF pose transform helper                                          #
    # ------------------------------------------------------------------ #

    def _transform_pose_to_base(self, translation, quaternion):
        """
        Transform a pose from self.camera_frame to self.base_frame using
        the ROS1 tf.TransformListener:
          1. Look up (trans, rot) for target←source with waitForTransform.
          2. Express the pose as a 4x4 homogeneous matrix.
          3. Multiply: pose_base = T_base_cam @ pose_cam.
          4. Extract translation and quaternion from the result.

        Args:
            translation (list): [x, y, z] in camera frame
            quaternion  (dict): {"x", "y", "z", "w"} in camera frame

        Returns:
            tuple: (t_base, q_base, error_string)
                  ([x,y,z                ], {"x","y","z","w"}, None)
                  (None, None,               error_string)
        """
        try:
            # Wait for the transform to become available
            self._tf_listener.waitForTransform(
                self.base_frame,
                self.camera_frame,
                rospy.Time(0),                       # latest available
                rospy.Duration(self.tf_timeout_sec),
            )

            # Get the transform: (trans=[x,y,z], rot=[x,y,z,w])
            trans, rot = self._tf_listener.lookupTransform(
                self.base_frame,
                self.camera_frame,
                rospy.Time(0),
            )

            # Build 4x4 homogeneous transform  T_base_cam
            T_base_cam = tft.quaternion_matrix(rot)          # 4x4, rotation part
            T_base_cam[0, 3] = trans[0]
            T_base_cam[1, 3] = trans[1]
            T_base_cam[2, 3] = trans[2]

            # Build 4x4 pose matrix  T_pose_cam  (grasp in camera frame)
            q_list = [quaternion["x"], quaternion["y"],
                      quaternion["z"], quaternion["w"]]
            T_pose_cam = tft.quaternion_matrix(q_list)
            T_pose_cam[0, 3] = translation[0]
            T_pose_cam[1, 3] = translation[1]
            T_pose_cam[2, 3] = translation[2]

            # Transform: grasp pose in base frame
            T_pose_base = np.dot(T_base_cam, T_pose_cam)

            t_base = [
                T_pose_base[0, 3],
                T_pose_base[1, 3],
                T_pose_base[2, 3],
            ]

            # Extract quaternion from the rotational part
            q_out = tft.quaternion_from_matrix(T_pose_base)   # [x, y, z, w]
            q_base = {"x": float(q_out[0]), "y": float(q_out[1]),
                      "z": float(q_out[2]), "w": float(q_out[3])}

            return t_base, q_base, None

        except tf.LookupException as e:
            return None, None, f"LookupException: {e}"
        except tf.ConnectivityException as e:
            return None, None, f"ConnectivityException: {e}"
        except tf.ExtrapolationException as e:
            return None, None, f"ExtrapolationException: {e}"
        except Exception as e:
            return None, None, f"Unexpected TF error: {e}\n{traceback.format_exc()}"

    def _get_cam_to_base_quat(self):
        """
        Look up the TF from base_frame and return
        [qx              , qy, qz, qw], or None if unavailable.
        """
        try:
            self._tf_listener.waitForTransform(
                self.base_frame,
                self.camera_frame,
                rospy.Time(0),
                rospy.Duration(self.tf_timeout_sec),
            )
            _trans, rot = self._tf_listener.lookupTransform(
                self.base_frame,
                self.camera_frame,
                rospy.Time(0),
            )
            rospy.loginfo(
                f"cam_to_base_quat: [{rot[0]:.4f}, {rot[1]:.4f}, "
                f"{rot[2]:.4f}, {rot[3]:.4f}]"
            )
            return list(rot)   # [x, y, z, w]

        except Exception as e:
            rospy.logwarn(
                f"TF lookup for cam_to_base_quat failed - "
                f"AnyGrasp horizontal filter will use fallback. Error: {e}"
            )
            return None

    # ------------------------------------------------------------------ #
    #  SAM2 WebSocket call                                                 #
    # ------------------------------------------------------------------ #

    def _call_sam2(self, rgb, depth, fx, fy, cx, cy, text_prompt):
        """
        Returns (result_dict, error_string). Exactly one is None.
        """
        try:
            role = "scene"
            depth_scale = rospy.get_param(f"/realsense/{role}/depth_scale", 0.001)
            payload = {
                "text_prompt": text_prompt,
                "rgb":         self._encode_rgb(rgb),
                "depth":       self._encode_depth(depth),
                "fx": fx, "fy": fy, "cx": cx, "cy": cy,
                "depth_scale": float(depth_scale)
            }

            cam_to_base_quat = self._get_cam_to_base_quat()
            if cam_to_base_quat is not None:
                payload["cam_to_base_quat"] = cam_to_base_quat
            else:
                rospy.logwarn(
                    "cam_to_base_quat unavailable — AnyGrasp horizontal filter "
                    "will fall back to camera_tilt_rad CLI flag."
                )

            result = asyncio.run(
                self._ws_send_recv(
                    self.sam2_url, json.dumps(payload),
                    self.sam2_timeout_sec, max_msg_mb=50,
                )
            )
            return result, None

        except aiohttp.ClientConnectorError:
            return None, f"Could not connect to SAM2 server at {self.sam2_url}"
        except aiohttp.WSServerHandshakeError as e:
            return None, f"SAM2 WebSocket handshake failed: {e}"
        except asyncio.TimeoutError:
            return None, f"SAM2 server timed out after {self.sam2_timeout_sec}s"
        except json.JSONDecodeError as e:
            return None, f"SAM2 response is not valid JSON: {e}"
        except Exception as e:
            return None, f"Unexpected SAM2 error: {e}\n{traceback.format_exc()}"

    # ------------------------------------------------------------------ #
    #  AnyGrasp WebSocket call                                             #
    # ------------------------------------------------------------------ #

    def _call_anygrasp(self):
        """
        Returns (result_dict, error_string). Exactly one is None.
        """
        try:
            result = asyncio.run(
                self._ws_send_recv(
                    self.anygrasp_url,
                    json.dumps({"trigger": True}),
                    self.grasp_timeout_sec, max_msg_mb=10,
                )
            )
            return result, None

        except aiohttp.ClientConnectorError:
            return None, f"Could not connect to AnyGrasp server at {self.anygrasp_url}"
        except aiohttp.WSServerHandshakeError as e:
            return None, f"AnyGrasp WebSocket handshake failed: {e}"
        except asyncio.TimeoutError:
            return None, f"AnyGrasp server timed out after {self.grasp_timeout_sec}s"
        except json.JSONDecodeError as e:
            return None, f"AnyGrasp response is not valid JSON: {e}"
        except Exception as e:
            return None, f"Unexpected AnyGrasp error: {e}\n{traceback.format_exc()}"

    # ------------------------------------------------------------------ #
    #  Shared async WebSocket primitive                                  #
    # ------------------------------------------------------------------ #

    async def _ws_send_recv(self, url, payload_str, timeout_sec, max_msg_mb=50):
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(
                url,
                max_msg_size=max_msg_mb * 1024 * 1024,
            ) as ws:
                await ws.send_str(payload_str)
                msg = await asyncio.wait_for(ws.receive(), timeout=timeout_sec)

                if msg.type == aiohttp.WSMsgType.TEXT:
                    return json.loads(msg.data)
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    raise RuntimeError(f"WebSocket error frame from {url}")
                elif msg.type in (aiohttp.WSMsgType.CLOSE,
                                  aiohttp.WSMsgType.CLOSING,
                                  aiohttp.WSMsgType.CLOSED):
                    raise RuntimeError(f"WebSocket closed unexpectedly by {url}")
                else:
                    raise RuntimeError(
                        f"Unexpected WebSocket message type {msg.type} from {url}"
                    )

    # ------------------------------------------------------------------ #
    #  Image encoding helpers                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _encode_rgb(bgr):
        ok, buf = cv2.imencode(".png", bgr)
        if not ok:
            raise RuntimeError("cv2.imencode failed for RGB image.")
        return base64.b64encode(buf.tobytes()).decode("utf-8")

    @staticmethod
    def _encode_depth(depth):
        if depth.dtype != np.uint16:
            depth = depth.astype(np.uint16)
        ok, buf = cv2.imencode(".png", depth)
        if not ok:
            raise RuntimeError("cv2.imencode failed for depth image.")
        return base64.b64encode(buf.tobytes()).decode("utf-8")

    # ------------------------------------------------------------------ #
    #  Response helper                                                     #
    # ------------------------------------------------------------------ #

    def _fail(self, response, msg):
        """
        Populate *response* as a failure and log the error.

        Args:
            response (RobotCommandResponse): to mutate
            msg      (str):                 human-readable error

        Returns:
            RobotCommandResponse
        """
        rospy.logerr(f"detect_objects service error: {msg}")
        response.result_code.result_code = ResultCode.FAILURE
        response.result_code.message     = msg
        response.data = json.dumps({"status": "error", "message": msg})
        return response

    # ------------------------------------------------------------------ #
    #  Spin                                                                #
    # ------------------------------------------------------------------ #

    def spin(self):
        """Block until ROS shutdown."""
        rospy.spin()


def main():
    """Initialize and spin the SegmentAndGraspNode."""
    rospy.init_node("detect_objects_service", anonymous=False)

    try:
        rospy.loginfo("Creating SegmentAndGraspNode ...")
        node = SegmentAndGraspNode()
        rospy.loginfo("SegmentAndGraspNode spinning ...")
        node.spin()

    except rospy.ROSInterruptException:
        rospy.loginfo("ROS interrupt - shutting down SegmentAndGraspNode.")
    except Exception as e:
        rospy.logerr(f"Fatal error: {e}\n{traceback.format_exc()}")
    finally:
        rospy.loginfo("SegmentAndGraspNode shutdown complete.")


if __name__ == "__main__":
    main()