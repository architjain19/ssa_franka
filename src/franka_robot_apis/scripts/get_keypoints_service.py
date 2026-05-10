#!/usr/bin/env python3
"""
ROS1 Noetic service node: DIFT Keypoint Planner Pipeline (with TF to base frame)
---------------------------------------------------------------------------------
Service: /robot/perception/get_keypoints  (robot_api_interfaces/RobotCommand)

Pipeline:
  1. Capture latest RGB-D + intrinsics from the RealSense
  2. Look up TF (base_frame <- camera_frame)
  3. Send RGB-D + task to the DIFT WebSocket server -> camera-frame keypoints
  4. Transform grasp/place points from camera frame to base frame
  5. Construct a top-down gripper orientation in base frame (DIFT returns a
     point only, not an orientation; for cloth folding we want a straight-down
     gripper aligned with the place-direction)

Request (JSON string in .req field):
{
    "text_prompt"   : "white cloth on table",     # what SAM2 should segment
    "task"          : "fold the cloth in half",   # natural-language task
    "stage_idx"     : 0,                          # optional, default 0
    "stage_history" : [                           # optional list of prior stages
        {"stage_description": "grasped left corner and folded to right"}
    ],
    "max_stages"    : 4                            # optional cap
}

Response (JSON string in .data field):
{
    "status": "success",

    # === Camera-frame outputs (from DIFT server) ===
    "keypoint_uv"        : [u, v],
    "keypoint_3d_camera" : [X, Y, Z],
    "place_uv"           : [u, v]    or null,
    "place_3d_camera"    : [X, Y, Z] or null,

    # === Base-frame outputs (computed here via TF) ===
    "grasp_pose_base": {
        "translation_wrt_base": [x, y, z],
        "quaternion_wrt_base":  {"x": ..., "y": ..., "z": ..., "w": ...}
    },
    "place_pose_base": {
        "translation_wrt_base": [x, y, z],
        "quaternion_wrt_base":  {"x": ..., "y": ..., "z": ..., "w": ...}
    },

    # === VLM planner outputs ===
    "stage_description" : "grasp top-left corner and align with bottom-left",
    "is_final_stage"    : false,
    "reasoning"         : "...VLM's explanation...",

    # === Diagnostics ===
    "num_candidates"    : 8,
    "selected_indices"  : {"grasp": 3, "place_target": 6},
    "debug_overlay_b64" : "<base64-png>",
    "timings"           : {"sam2_s": 0.3, "dift_s": 0.5, "vlm_s": 4.2, ...}
}

ROS1 usage:
    rosrun franka_robot_apis get_keypoints_service.py

    rosservice call /robot/perception/get_keypoints \
        '{"req": "{\"text_prompt\":\"white cloth\",\"task\":\"fold the cloth in half\"}"}'
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
import ros_numpy

from robot_api_interfaces.srv import RobotCommand, RobotCommandResponse
from robot_api_interfaces.msg import ResultCode


# ---------------------------------------------------------------------------
# Helper: build a top-down gripper pose in base frame
# ---------------------------------------------------------------------------
def _build_topdown_pose(point_base, yaw_rad=0.0):
    """
    Construct a 4x4 pose for a top-down grasp at a 3D point in base frame.

    The Franka panda_hand convention is:
      +Z = approach direction (out of fingers)
      +Y = closing direction (between fingers)
      +X = orthogonal to closing

    For a top-down grasp:
      - Gripper +Z points DOWN in base frame, i.e. -Z_base
      - Gripper +X points along base +X (rotated by yaw)
      - Gripper +Y points along base +Y (rotated by yaw)

    Args:
        point_base (list): [x, y, z] in base frame (meters)
        yaw_rad   (float): rotation about world Z; lets the gripper close
                           along a chosen direction in the table plane.

    Returns:
        tuple: (translation_list, quaternion_dict)
    """
    # Rotation: gripper axes -> base axes
    # Start with "no yaw" top-down: gripper +Z = -base_Z, gripper +X = base_X
    # That's a 180-deg rotation about base_X.
    R_topdown = np.array([
        [1.0,  0.0,  0.0],
        [0.0, -1.0,  0.0],
        [0.0,  0.0, -1.0],
    ], dtype=np.float64)

    # Yaw about world Z (in base frame), applied AFTER the topdown flip
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    R_yaw = np.array([
        [ c, -s, 0.0],
        [ s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    R = R_yaw @ R_topdown

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = point_base

    q = tft.quaternion_from_matrix(T)  # [x, y, z, w]
    return (
        [float(point_base[0]), float(point_base[1]), float(point_base[2])],
        {"x": float(q[0]), "y": float(q[1]), "z": float(q[2]), "w": float(q[3])},
    )


def _yaw_from_grasp_to_place(grasp_xy, place_xy):
    """
    Compute the yaw angle (in base-frame XY plane) such that the gripper's
    closing direction (panda_hand +Y) points along grasp -> place.

    For folding, this aligns the gripper's "drag" direction with the fold
    direction so the cloth gets pulled along the right vector.

    Args:
        grasp_xy (sequence): (x, y) of grasp in base frame
        place_xy (sequence): (x, y) of place in base frame

    Returns:
        float: yaw in radians, or 0.0 if direction is degenerate
    """
    dx = place_xy[0] - grasp_xy[0]
    dy = place_xy[1] - grasp_xy[1]
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return 0.0
    # atan2 of the direction vector — y-axis of the gripper aligns with this
    return float(np.arctan2(dy, dx))


# ---------------------------------------------------------------------------
# Main service class
# ---------------------------------------------------------------------------
class GetKeypointsNode:
    """
    ROS1 service node wrapping the DIFT /plan_action WebSocket endpoint and
    transforming the camera-frame keypoints into base-frame grasp/place poses.
    """

    def __init__(self):
        # ------------------------------------------------------------------ #
        #  Parameters                                                         #
        # ------------------------------------------------------------------ #
        self.dift_url = rospy.get_param(
            "~dift_url", "ws://10.158.54.164:8769/plan_action"
        )

        # Camera topics
        self.rgb_topic         = rospy.get_param("~rgb_topic",         "/realsense/scene/color/image_raw")
        self.depth_topic       = rospy.get_param("~depth_topic",       "/realsense/scene/aligned_depth_to_color/image_raw")
        self.camera_info_topic = rospy.get_param("~camera_info_topic", "/realsense/scene/aligned_depth_to_color/camera_info")

        # Fallback intrinsics
        self.fx_default = float(rospy.get_param("~fx", 752.0038452148438))
        self.fy_default = float(rospy.get_param("~fy", 751.7178344726562))
        self.cx_default = float(rospy.get_param("~cx", 628.4379272460938))
        self.cy_default = float(rospy.get_param("~cy", 335.1157531738281))

        # Depth scale — RealSense default is 0.001 (mm -> m)
        self.depth_scale = float(rospy.get_param("~depth_scale", 0.001))

        # Default fallbacks for optional request fields
        self.default_max_stages = int(rospy.get_param("~default_max_stages", 4))

        # Timeouts
        self.camera_wait_sec     = float(rospy.get_param("~camera_wait_sec",     1.0))
        self.dift_timeout_sec    = float(rospy.get_param("~dift_timeout_sec",    60.0))
        self.response_max_npz_mb = float(rospy.get_param("~response_max_npz_mb", 25.0))

        # TF
        self.camera_frame   = rospy.get_param("~camera_frame", "cam_scene_color_optical_frame")
        self.base_frame     = rospy.get_param("~base_frame",   "panda_link0")
        self.tf_timeout_sec = float(rospy.get_param("~tf_timeout_sec", 2.0))

        # Grasp pose construction
        # If True, yaw the gripper so its closing direction points from grasp
        # toward place (helps the fold action drag the cloth along the right vector).
        # If no place point is returned, this is ignored.
        self.align_yaw_to_place = bool(rospy.get_param("~align_yaw_to_place", True))

        # ------------------------------------------------------------------ #
        #  Camera state                                                       #
        # ------------------------------------------------------------------ #
        self.latest_rgb           = None
        self.latest_depth         = None
        self.fx                   = self.fx_default
        self.fy                   = self.fy_default
        self.cx                   = self.cx_default
        self.cy                   = self.cy_default
        self.camera_info_received = False
        self._cam_lock            = threading.Lock()

        # ------------------------------------------------------------------ #
        #  TF                                                                 #
        # ------------------------------------------------------------------ #
        self._tf_listener = tf.TransformListener()

        # ------------------------------------------------------------------ #
        #  Subscriptions                                                      #
        # ------------------------------------------------------------------ #
        rospy.Subscriber(
            self.rgb_topic, Image, self._rgb_cb,
            queue_size=1, buff_size=2 ** 24,
        )
        rospy.Subscriber(
            self.depth_topic, Image, self._depth_cb,
            queue_size=1, buff_size=2 ** 24,
        )
        rospy.Subscriber(
            self.camera_info_topic, CameraInfo, self._camera_info_cb,
            queue_size=10,
        )

        # ------------------------------------------------------------------ #
        #  Service                                                            #
        # ------------------------------------------------------------------ #
        self._service = rospy.Service(
            "/robot/perception/get_keypoints",
            RobotCommand,
            self._handle_request,
        )

        rospy.loginfo(
            "\nGetKeypointsNode (ROS1) ready.\n"
            f"  Service     : /robot/perception/get_keypoints\n"
            f"  DIFT        : {self.dift_url}\n"
            f"  RGB         : {self.rgb_topic}\n"
            f"  Depth       : {self.depth_topic}\n"
            f"  CamInfo     : {self.camera_info_topic}\n"
            f"  Camera frame: {self.camera_frame}\n"
            f"  Base frame  : {self.base_frame}\n"
            f"  DepthScale  : {self.depth_scale}\n"
            f"  Align yaw   : {self.align_yaw_to_place}"
        )

    # ------------------------------------------------------------------ #
    #  Camera callbacks                                                    #
    # ------------------------------------------------------------------ #

    def _rgb_cb(self, msg):
        try:
            img = ros_numpy.numpify(msg)
            enc = msg.encoding.lower()
            if enc == "rgb8":
                img = img[..., ::-1].copy()
            elif enc == "rgba8":
                img = img[..., :3][..., ::-1].copy()
            elif enc == "bgra8":
                img = img[..., :3].copy()
            with self._cam_lock:
                self.latest_rgb = img
        except Exception as e:
            rospy.logwarn(f"RGB decode error: {e}")

    def _depth_cb(self, msg):
        try:
            img = ros_numpy.numpify(msg)
            if img.dtype != np.uint16:
                img = img.astype(np.uint16)
            with self._cam_lock:
                self.latest_depth = img
        except Exception as e:
            rospy.logwarn(f"Depth decode error: {e}")

    def _camera_info_cb(self, msg):
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
        rospy.loginfo(f"Get-keypoints request received: {request.req}")
        response = RobotCommandResponse()

        # --- 1. Parse request -------------------------------------------
        try:
            req_data = json.loads(request.req)
        except (json.JSONDecodeError, ValueError) as e:
            return self._fail(response, f"Bad request (not valid JSON): {e}")

        text_prompt = (req_data.get("text_prompt") or "").strip() or "object"
        task        = (req_data.get("task")        or "").strip()
        if not task:
            return self._fail(response, "Missing required field 'task' (non-empty string).")

        stage_idx     = int(req_data.get("stage_idx", 0))
        stage_history = req_data.get("stage_history", []) or []
        max_stages    = int(req_data.get("max_stages", self.default_max_stages))

        if not isinstance(stage_history, list):
            return self._fail(response, "'stage_history' must be a list (or omitted).")

        # --- 2. Wait for camera frames ----------------------------------
        rospy.loginfo(f"Waiting up to {self.camera_wait_sec}s for camera frames ...")
        deadline   = time.time() + self.camera_wait_sec
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
            rgb            = self.latest_rgb.copy()
            depth          = self.latest_depth.copy()
            fx, fy, cx, cy = self.fx, self.fy, self.cx, self.cy

        if not self.camera_info_received:
            rospy.logwarn(
                "CameraInfo not yet received — using fallback intrinsics. "
                f"fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}"
            )

        # --- 3. Call DIFT /plan_action ----------------------------------
        rospy.loginfo(
            f"Calling DIFT plan_action | prompt='{text_prompt}' "
            f"task='{task}' stage={stage_idx}/{max_stages} "
            f"history_len={len(stage_history)}"
        )
        dift_result, dift_err = self._call_dift(
            rgb, depth, fx, fy, cx, cy,
            text_prompt, task, stage_idx, stage_history, max_stages,
        )

        if dift_err:
            return self._fail(response, f"DIFT server failed: {dift_err}")

        if dift_result.get("status") != "success":
            return self._fail(
                response,
                f"DIFT server returned non-success status: "
                f"{dift_result.get('message', json.dumps(dift_result))}"
            )

        # --- 4. Validate the keypoint payload ---------------------------
        for required_field in ("keypoint_uv", "keypoint_3d_camera",
                               "stage_description", "is_final_stage"):
            if required_field not in dift_result:
                return self._fail(
                    response,
                    f"DIFT response missing required field '{required_field}'."
                )

        grasp_cam = dift_result["keypoint_3d_camera"]
        place_cam = dift_result.get("place_3d_camera")  # may be None

        rospy.loginfo(
            f"DIFT returned | grasp_cam=({grasp_cam[0]:.4f}, "
            f"{grasp_cam[1]:.4f}, {grasp_cam[2]:.4f})  "
            f"place_cam={place_cam}"
        )

        # --- 5. TF: camera -> base for both points ---------------------
        T_base_cam, tf_err = self._lookup_T_base_cam()
        if tf_err:
            return self._fail(
                response,
                f"TF lookup '{self.base_frame}' <- '{self.camera_frame}' failed: {tf_err}"
            )

        grasp_base = self._transform_point(T_base_cam, grasp_cam)
        place_base = self._transform_point(T_base_cam, place_cam) if place_cam else None

        rospy.loginfo(
            f"Transformed | grasp_base=({grasp_base[0]:.4f}, "
            f"{grasp_base[1]:.4f}, {grasp_base[2]:.4f})  "
            f"place_base={place_base}"
        )

        # --- 6. Build top-down poses with optional yaw alignment --------
        # Yaw aligns gripper closing direction with grasp->place vector.
        # Falls back to 0 (gripper +X along base +X) if no place point.
        if place_base is not None and self.align_yaw_to_place:
            yaw = _yaw_from_grasp_to_place(grasp_base[:2], place_base[:2])
            rospy.loginfo(f"Yaw aligned to grasp->place: {np.degrees(yaw):.1f} deg")
        else:
            yaw = 0.0

        grasp_t, grasp_q = _build_topdown_pose(grasp_base, yaw_rad=yaw)
        place_pose = None
        if place_base is not None:
            place_t, place_q = _build_topdown_pose(place_base, yaw_rad=yaw)
            place_pose = {
                "position": {"x": place_t[0], "y": place_t[1], "z": place_t[2]},
                "orientation":  {"x": place_q["x"], "y": place_q["y"], "z": place_q["z"], "w": place_q["w"]},
            }

        rospy.loginfo(
            f"Grasp pose (base) | xyz=({grasp_t[0]:.4f}, {grasp_t[1]:.4f}, {grasp_t[2]:.4f}) "
            f"quat=({grasp_q['x']:.4f}, {grasp_q['y']:.4f}, "
            f"{grasp_q['z']:.4f}, {grasp_q['w']:.4f})"
        )

        # --- 7. Optional debug overlay (size cap) -----------------------
        overlay_b64 = dift_result.get("debug_overlay_b64", "")
        overlay_bytes = 0
        if overlay_b64:
            try:
                overlay_bytes = len(base64.b64decode(overlay_b64))
            except Exception as e:
                rospy.logwarn(f"Could not decode debug_overlay_b64: {e}")
                overlay_b64 = ""

            max_bytes = int(self.response_max_npz_mb * 1024 * 1024)
            if overlay_bytes > max_bytes:
                rospy.logwarn(
                    f"debug_overlay_b64 too large ({overlay_bytes} bytes) — dropping."
                )
                overlay_b64 = ""
                overlay_bytes = 0

        # --- 8. Build response ------------------------------------------
        payload = {
            "status":  "success",
            "message": "DIFT planning complete. Camera-frame and base-frame poses returned.",

            # Camera-frame (raw from DIFT server)
            # "keypoint_uv":          dift_result.get("keypoint_uv"),
            # "keypoint_3d_camera":   dift_result.get("keypoint_3d_camera"),
            # "depth_at_grasp_m":     dift_result.get("depth_at_grasp_m"),
            # "place_uv":             dift_result.get("place_uv"),
            # "place_3d_camera":      dift_result.get("place_3d_camera"),
            # "depth_at_place_m":     dift_result.get("depth_at_place_m"),

            # Base-frame poses (computed here)
            "grasp_pose_base": {
                "position": {"x": grasp_t[0], "y": grasp_t[1], "z": grasp_t[2]},
                "orientation":  {"x": grasp_q["x"], "y": grasp_q["y"], "z": grasp_q["z"], "w": grasp_q["w"]},
            },
            "place_pose_base": place_pose,

            # VLM planner outputs
            "stage_description": dift_result.get("stage_description"),
            "is_final_stage":    bool(dift_result.get("is_final_stage", False)),
            # "reasoning":         dift_result.get("reasoning", ""),

            # Diagnostics
            # "num_candidates":     dift_result.get("num_candidates"),
            # "selected_indices":   dift_result.get("selected_indices", {}),
            # "debug_overlay_b64":  overlay_b64,
            # "debug_overlay_bytes": overlay_bytes,
            # "timings":            dift_result.get("timings", {}),
        }

        response.result_code.result_code = ResultCode.SUCCESS
        response.result_code.message     = "Keypoint planning succeeded."
        response.data                    = json.dumps(payload)

        rospy.loginfo(
            f"get_keypoints stage_description: '{payload.get('stage_description')}' "
            f"| is_final_stage={payload.get('is_final_stage')}"
        )
        return response

    # ------------------------------------------------------------------ #
    #  TF helpers                                                          #
    # ------------------------------------------------------------------ #

    def _lookup_T_base_cam(self):
        """
        Return (T_base_cam, error_string). T_base_cam is a 4x4 numpy array
        encoding the rigid transform from camera optical frame to base frame.
        Exactly one of the return values is None.
        """
        try:
            self._tf_listener.waitForTransform(
                self.base_frame,
                self.camera_frame,
                rospy.Time(0),
                rospy.Duration(self.tf_timeout_sec),
            )
            trans, rot = self._tf_listener.lookupTransform(
                self.base_frame,
                self.camera_frame,
                rospy.Time(0),
            )
            T = tft.quaternion_matrix(rot)  # 4x4 with rotation only
            T[0, 3] = trans[0]
            T[1, 3] = trans[1]
            T[2, 3] = trans[2]
            return T, None
        except tf.LookupException as e:
            return None, f"LookupException: {e}"
        except tf.ConnectivityException as e:
            return None, f"ConnectivityException: {e}"
        except tf.ExtrapolationException as e:
            return None, f"ExtrapolationException: {e}"
        except Exception as e:
            return None, f"Unexpected TF error: {e}\n{traceback.format_exc()}"

    @staticmethod
    def _transform_point(T_base_cam, p_cam):
        """Apply 4x4 transform to a 3-vector. Returns list[float]."""
        p_h = np.array([p_cam[0], p_cam[1], p_cam[2], 1.0], dtype=np.float64)
        p_b = T_base_cam @ p_h
        return [float(p_b[0]), float(p_b[1]), float(p_b[2])]

    # ------------------------------------------------------------------ #
    #  DIFT WebSocket call                                                 #
    # ------------------------------------------------------------------ #

    def _call_dift(self, rgb, depth, fx, fy, cx, cy,
                   text_prompt, task, stage_idx, stage_history, max_stages):
        try:
            payload = {
                "rgb":            self._encode_rgb(rgb),
                "depth":          self._encode_depth(depth),
                "fx": fx, "fy": fy, "cx": cx, "cy": cy,
                "depth_scale":    self.depth_scale,

                "text_prompt":    text_prompt,
                "task":           task,
                "stage_idx":      stage_idx,
                "stage_history":  stage_history,
                "max_stages":     max_stages,
            }

            result = asyncio.run(
                self._ws_send_recv(
                    self.dift_url, json.dumps(payload),
                    self.dift_timeout_sec, max_msg_mb=50,
                )
            )
            return result, None

        except aiohttp.ClientConnectorError:
            return None, f"Could not connect to DIFT server at {self.dift_url}"
        except aiohttp.WSServerHandshakeError as e:
            return None, f"DIFT WebSocket handshake failed: {e}"
        except asyncio.TimeoutError:
            return None, f"DIFT server timed out after {self.dift_timeout_sec}s"
        except json.JSONDecodeError as e:
            return None, f"DIFT response is not valid JSON: {e}"
        except Exception as e:
            return None, f"Unexpected DIFT error: {e}\n{traceback.format_exc()}"

    # ------------------------------------------------------------------ #
    #  Shared async WebSocket primitive                                    #
    # ------------------------------------------------------------------ #

    async def _ws_send_recv(self, url, payload_str, timeout_sec, max_msg_mb=50):
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(
                url, max_msg_size=max_msg_mb * 1024 * 1024,
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
    #  Image encoding helpers                                              #
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
        rospy.logerr(f"get_keypoints service error: {msg}")
        response.result_code.result_code = ResultCode.FAILURE
        response.result_code.message     = msg
        response.data = json.dumps({"status": "error", "message": msg})
        return response

    # ------------------------------------------------------------------ #
    #  Spin                                                                #
    # ------------------------------------------------------------------ #

    def spin(self):
        rospy.spin()


def main():
    rospy.init_node("get_keypoints_service", anonymous=False)
    try:
        rospy.loginfo("Creating GetKeypointsNode ...")
        node = GetKeypointsNode()
        rospy.loginfo("GetKeypointsNode spinning ...")
        node.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS interrupt — shutting down GetKeypointsNode.")
    except Exception as e:
        rospy.logerr(f"Fatal error: {e}\n{traceback.format_exc()}")
    finally:
        rospy.loginfo("GetKeypointsNode shutdown complete.")


if __name__ == "__main__":
    main()