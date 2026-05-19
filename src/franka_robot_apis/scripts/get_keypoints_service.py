#!/usr/bin/env python3
"""
ROS1 Noetic service node: DIFT Keypoint Planner Pipeline
---------------------------------------------------------
Service: /robot/perception/detect_keypoints  (robot_api_interfaces/RobotCommand)

Pipeline:
  1. Capture latest RGB-D + intrinsics from the RealSense
  2. Look up TF (base_frame <- camera_frame) -> T_base_cam (4x4)
  3. Send RGB-D + intrinsics + T_base_cam + task/prompt to the DIFT
     WebSocket server (/plan_action)
  4. Log the returned payload via rospy and forward it verbatim to the caller

Request (JSON string in .req field):
{
    "text_prompt"          : "white cloth on table",      # SAM2 prompt
    "task"                 : "fold the cloth in half",    # natural-language task
    "current_ee_pose_base" : [[...4x4...]]                # optional, 4x4 list
    "num_candidates"       : 5                           # optional, int
    "num_path_waypoints"   : 20                          # optional, int
}

Response (JSON string in .data field):
    Forwarded verbatim from the DIFT server (no post-processing).

ROS1 usage:
    rosrun franka_robot_apis detect_keypoints_service.py

    rosservice call /robot/perception/detect_keypoints '{"req": "{\"text_prompt\":\"white cloth\",\"task\":\"fold the cloth in half\"}"}'
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

from sensor_msgs.msg import Image, CameraInfo
import ros_numpy

from robot_api_interfaces.srv import RobotCommand, RobotCommandResponse, RobotCommandRequest
from robot_api_interfaces.msg import ResultCode


# ---------------------------------------------------------------------------
# Main service class
# ---------------------------------------------------------------------------
class DetectKeypointsNode:
    """
    ROS1 service node wrapping the DIFT /plan_action WebSocket endpoint.

    Captures an RGB + depth frame, gathers intrinsics and T_base_cam, ships
    everything to the DIFT server alongside the user's text_prompt + task,
    and forwards the response verbatim to the caller.
    """

    def __init__(self):
        # ------------------------------------------------------------------ #
        #  Parameters                                                          #
        # ------------------------------------------------------------------ #
        self.dift_url = rospy.get_param(
            "~dift_url", "ws://10.158.54.164:8769/select_keypoint"
        )

        # Camera topics — same RealSense serial used in detect_objects node
        self.rgb_topic         = rospy.get_param("~rgb_topic",         "/zed/scene/color/image_raw")
        self.depth_topic       = rospy.get_param("~depth_topic",       "/zed/scene/aligned_depth_to_color/image_raw")
        self.camera_info_topic = rospy.get_param("~camera_info_topic", "/zed/scene/aligned_depth_to_color/camera_info")

        # Fallback intrinsics (used only if camera_info never arrives)
        self.fx_default = float(rospy.get_param("~fx", 752.0038452148438))
        self.fy_default = float(rospy.get_param("~fy", 751.7178344726562))
        self.cx_default = float(rospy.get_param("~cx", 628.4379272460938))
        self.cy_default = float(rospy.get_param("~cy", 335.1157531738281))

        # Depth scale — RealSense default is 0.001 (mm -> m)
        self.depth_scale = float(rospy.get_param("~depth_scale", 0.001))

        # Timeouts
        self.camera_wait_sec  = float(rospy.get_param("~camera_wait_sec",  1.0))
        self.dift_timeout_sec = float(rospy.get_param("~dift_timeout_sec", 60.0))

        # TF parameters
        self.camera_frame   = rospy.get_param("~camera_frame", "zed_scene_left_optical_frame")
        self.base_frame     = rospy.get_param("~base_frame",   "panda_link0")
        self.tf_timeout_sec = float(rospy.get_param("~tf_timeout_sec", 2.0))

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
        #  Subscriptions                                                       #
        # ------------------------------------------------------------------ #
        rospy.Subscriber(
            self.rgb_topic, Image, self._rgb_cb,
            queue_size=1, buff_size=2 ** 24,   # 16 MB — fits HD colour frames
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
        #  Service                                                             #
        # ------------------------------------------------------------------ #
        self._service = rospy.Service(
            "/robot/perception/detect_keypoints",
            RobotCommand,
            self._handle_request,
        )

        rospy.loginfo(
            "\nDetectKeypointsNode (ROS1) ready.\n"
            f"  Service     : /robot/perception/detect_keypoints\n"
            f"  DIFT        : {self.dift_url}\n"
            f"  RGB         : {self.rgb_topic}\n"
            f"  Depth       : {self.depth_topic}\n"
            f"  CamInfo     : {self.camera_info_topic}\n"
            f"  Camera frame: {self.camera_frame}\n"
            f"  Base frame  : {self.base_frame}\n"
            f"  DepthScale  : {self.depth_scale}\n"
        )

    # ------------------------------------------------------------------ #
    #  Camera callbacks — ros_numpy replaces CvBridge                     #
    # ------------------------------------------------------------------ #

    def _rgb_cb(self, msg):
        """
        Convert sensor_msgs/Image -> BGR uint8 numpy array using ros_numpy.
        """
        try:
            img = ros_numpy.numpify(msg)
            enc = msg.encoding.lower()
            if enc == "rgb8":
                img = img[..., ::-1].copy()                  # RGB -> BGR
            elif enc == "rgba8":
                img = img[..., :3][..., ::-1].copy()         # RGBA -> BGR
            elif enc == "bgra8":
                img = img[..., :3].copy()                    # drop alpha, BGR
            # bgr8 / mono encodings: use as-is
            with self._cam_lock:
                self.latest_rgb = img
        except Exception as e:
            rospy.logwarn(f"RGB decode error: {e}")

    def _depth_cb(self, msg):
        """
        Convert sensor_msgs/Image -> uint16 numpy array using ros_numpy.
        """
        try:
            img = ros_numpy.numpify(msg)
            if img.dtype != np.uint16:
                img = img.astype(np.uint16)
            with self._cam_lock:
                self.latest_depth = img
        except Exception as e:
            rospy.logwarn(f"Depth decode error: {e}")

    def _camera_info_cb(self, msg):
        """
        Cache camera intrinsics on first receipt.
        K = [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        """
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
        rospy.Service callback - called in a dedicated thread per request.
        """
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
        
        # Optional: client may specify number of candidates and path waypoints
        num_req_candidates = req_data.get("num_candidates", 50)
        if not isinstance(num_req_candidates, int) or num_req_candidates < 5 or num_req_candidates > 100:
            return self._fail(response, f"'num_candidates' must be a positive integer between 5 and 100, got {num_req_candidates}.")

        num_req_path_waypoints = req_data.get("num_path_waypoints", 20)
        if not isinstance(num_req_path_waypoints, int) or num_req_path_waypoints < 2 or num_req_path_waypoints > 20:
            return self._fail(response, f"'num_path_waypoints' must be a positive integer between 2 and 20, got {num_req_path_waypoints}.")

        # Optional: client may pass a current EE pose in base frame (4x4 list)
        current_ee_pose_base = req_data.get("current_ee_pose_base", None)
        if current_ee_pose_base is not None:
            try:
                ee_arr = np.array(current_ee_pose_base, dtype=np.float64)
                if ee_arr.shape != (4, 4):
                    return self._fail(
                        response,
                        f"'current_ee_pose_base' must be a 4x4 list, got shape {ee_arr.shape}."
                    )
                current_ee_pose_base = ee_arr.tolist()
            except Exception as e:
                return self._fail(
                    response,
                    f"Could not parse 'current_ee_pose_base': {e}"
                )

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

        # --- 3. Get T_base_cam from TF (camera frame -> robot base frame)
        T_base_cam = np.eye(4).tolist()  # default to identity if no TF available
        try:
            import tf2_ros
            import tf2_geometry_msgs
            tf_buffer = tf2_ros.Buffer()
            tf_listener = tf2_ros.TransformListener(tf_buffer)
            transform = tf_buffer.lookup_transform(
                self.base_frame,            # target frame (robot base)
                self.camera_frame,          # source frame (RealSense RGB camera)
                rospy.Time(0),              # latest available
                rospy.Duration(self.tf_timeout_sec)
            )
            tf_to_kdl = tf2_geometry_msgs.transform_to_kdl(transform)
            kdl_rot = tf_to_kdl.M
            kdl_pos = tf_to_kdl.p
            hom_matrix = np.array([
                [kdl_rot[0, 0], kdl_rot[0, 1], kdl_rot[0, 2], kdl_pos[0]],
                [kdl_rot[1, 0], kdl_rot[1, 1], kdl_rot[1, 2], kdl_pos[1]],
                [kdl_rot[2, 0], kdl_rot[2, 1], kdl_rot[2, 2], kdl_pos[2]],
                [0, 0, 0, 1]
            ])
            T_base_cam = hom_matrix.tolist()  # JSON-serializable
            rospy.loginfo("Successfully obtained T_base_cam from TF.")
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(f"Could not obtain T_base_cam from TF: {e}. Using identity.")
        except Exception as e:
            rospy.logwarn(f"Unexpected TF error: {e}. Using identity.")

        # --- 4. Call DIFT /plan_action ----------------------------------
        rospy.loginfo(
            f"Calling DIFT plan_action | prompt='{text_prompt}' task='{task}'"
        )
        dift_result, dift_err = self._call_dift(
            rgb, depth, fx, fy, cx, cy,
            text_prompt, task, T_base_cam, current_ee_pose_base, num_req_candidates, num_req_path_waypoints
        )

        if dift_err:
            return self._fail(response, f"DIFT server failed: {dift_err}")

        if not isinstance(dift_result, dict) or dift_result.get("status") != "success":
            return self._fail(
                response,
                f"DIFT server returned non-success status: "
                f"{dift_result.get('message', json.dumps(dift_result)) if isinstance(dift_result, dict) else dift_result}"
            )

        # --- 5. Log the payload and forward verbatim --------------------
        try:
            payload_str = json.dumps(dift_result)
            rospy.loginfo(f"DIFT response: {payload_str}")
            return RobotCommandResponse(
                result_code=ResultCode(result_code=0, message="DIFT call successful."),
                data=payload_str,
            )
        except (TypeError, ValueError) as e:
            return self._fail(
                response,
                f"DIFT response is not JSON-serializable: {e}"
            )


    # ------------------------------------------------------------------ #
    #  DIFT WebSocket call                                                 #
    # ------------------------------------------------------------------ #

    def _call_dift(self, rgb, depth, fx, fy, cx, cy,
                   text_prompt, task, T_base_cam, current_ee_pose_base, num_req_candidates=50, num_req_path_waypoints=10):
        """
        Send RGB + depth + intrinsics + T_base_cam + task/prompt to the DIFT
        /plan_action endpoint.

        Returns:
            tuple: (result_dict, error_string) — exactly one is None.
        """
        try:
            payload = {
                "rgb":          self._encode_rgb(rgb),
                "depth":        self._encode_depth(depth),
                "fx": fx, "fy": fy, "cx": cx, "cy": cy,
                "depth_scale":  self.depth_scale,
                "T_base_cam":   T_base_cam,
                "text_prompt":  text_prompt,
                "task":         task,
                "num_candidates": num_req_candidates,
                "num_path_waypoints": num_req_path_waypoints,
            }
            if current_ee_pose_base is not None:
                payload["current_ee_pose_base"] = current_ee_pose_base

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
    #  Image encoding helpers                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _encode_rgb(bgr):
        """Encode a BGR uint8 numpy array as a lossless PNG base64 string."""
        ok, buf = cv2.imencode(".png", bgr)
        if not ok:
            raise RuntimeError("cv2.imencode failed for RGB image.")
        return base64.b64encode(buf.tobytes()).decode("utf-8")

    @staticmethod
    def _encode_depth(depth):
        """Encode a uint16 depth numpy array as a lossless PNG base64 string."""
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
        """
        rospy.logerr(f"detect_keypoints service error: {msg}")
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
    """Initialize and spin the DetectKeypointsNode."""
    rospy.init_node("detect_keypoints_service", anonymous=False)

    try:
        rospy.loginfo("Creating DetectKeypointsNode ...")
        node = DetectKeypointsNode()
        rospy.loginfo("DetectKeypointsNode spinning ...")
        node.spin()

    except rospy.ROSInterruptException:
        rospy.loginfo("ROS interrupt — shutting down DetectKeypointsNode.")
    except Exception as e:
        rospy.logerr(f"Fatal error: {e}\n{traceback.format_exc()}")
    finally:
        rospy.loginfo("DetectKeypointsNode shutdown complete.")


if __name__ == "__main__":
    main()