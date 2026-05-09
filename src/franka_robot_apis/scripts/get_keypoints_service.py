#!/usr/bin/env python3
"""
ROS1 Noetic service node: DIFT Keypoint Planner Pipeline
---------------------------------------------------------
Service: /robot/perception/get_keypoints  (robot_api_interfaces/RobotCommand)

Request (JSON string in .req field):
{
    "text_prompt"   : "white cloth on table",            # what SAM2 should segment
    "task"          : "fold the cloth in half",          # natural-language task for VLM
    "stage_idx"     : 0,                                 # optional, default 0
    "stage_history" : [                                  # optional, descriptions of completed stages
        {"stage_description": "grasped left corner and folded to right"}
    ],
    "max_stages"    : 4                                  # optional cap on number of stages
}

Response (JSON string in .data field):
{
    "status": "success",

    "keypoint_uv":         [u, v],                       # pixel to grasp
    "keypoint_3d_camera":  [X, Y, Z],                    # camera-frame point in METERS
    "depth_at_grasp_m":    0.65,

    "place_uv":            [u, v]    or null,            # pixel to place at (alignment target)
    "place_3d_camera":     [X, Y, Z] or null,            # camera-frame point or null
    "depth_at_place_m":    0.71      or null,

    "stage_description":   "grasp top-left corner and align with bottom-left",
    "is_final_stage":      false,                        # true on the LAST stage of the task
    "reasoning":           "...VLM's explanation...",

    "num_candidates":      8,
    "selected_indices":    {"grasp": 3, "place_target": 6},

    "debug_overlay_b64":   "<base64-png>",               # annotated image with all candidates

    "timings": {"sam2_s": 0.3, "dift_s": 0.5, "vlm_s": 4.2, "total_s": 5.0}
}

ROS1 usage:
    rosrun franka_robot_apis get_keypoints_service.py

    rosservice call /robot/perception/get_keypoints \
        '{"req": "{\"text_prompt\":\"white cloth\",\"task\":\"fold the cloth in half\",\"stage_idx\":0}"}'
"""

import json
import asyncio
import base64
import io
import threading
import time
import traceback

import cv2
import numpy as np
import aiohttp

import rospy

from sensor_msgs.msg import Image, CameraInfo
import ros_numpy

from robot_api_interfaces.srv import RobotCommand, RobotCommandResponse
from robot_api_interfaces.msg import ResultCode


# ---------------------------------------------------------------------------
# Main service class
# ---------------------------------------------------------------------------
class GetKeypointsNode:
    """
    ROS1 service node wrapping the DIFT /plan_action WebSocket endpoint.

    Captures an RGB + depth frame, ships them to the DIFT server along with the
    task instruction and stage info, and forwards the planner's response (grasp
    + place keypoints in camera frame, plus VLM reasoning) to the caller.
    """

    def __init__(self):
        # ------------------------------------------------------------------ #
        #  Parameters                                                         #
        # ------------------------------------------------------------------ #
        self.dift_url = rospy.get_param(
            "~dift_url", "ws://10.158.54.164:8769/plan_action"
        )

        # Camera topics — same RealSense serial used elsewhere
        self.rgb_topic         = rospy.get_param("~rgb_topic",         "/realsense/scene/color/image_raw")
        self.depth_topic       = rospy.get_param("~depth_topic",       "/realsense/scene/aligned_depth_to_color/image_raw")
        self.camera_info_topic = rospy.get_param("~camera_info_topic", "/realsense/scene/aligned_depth_to_color/camera_info")

        # Fallback intrinsics (used only if camera_info never arrives)
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
        self.dift_timeout_sec    = float(rospy.get_param("~dift_timeout_sec",    60.0))   # DIFT + VLM > SAM2 alone
        self.response_max_npz_mb = float(rospy.get_param("~response_max_npz_mb", 25.0))

        # ------------------------------------------------------------------ #
        #  Camera state                                                       #
        # ------------------------------------------------------------------ #
        self.latest_rgb           = None    # np.ndarray uint8  (H,W,3) BGR
        self.latest_depth         = None    # np.ndarray uint16 (H,W)
        self.fx                   = self.fx_default
        self.fy                   = self.fy_default
        self.cx                   = self.cx_default
        self.cy                   = self.cy_default
        self.camera_info_received = False
        self._cam_lock            = threading.Lock()

        # ------------------------------------------------------------------ #
        #  Subscriptions                                                      #
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
            f"  DepthScale  : {self.depth_scale}"
        )

    # ------------------------------------------------------------------ #
    #  Camera callbacks — ros_numpy replaces CvBridge                     #
    # ------------------------------------------------------------------ #

    def _rgb_cb(self, msg):
        """Convert sensor_msgs/Image -> BGR uint8 numpy array using ros_numpy."""
        try:
            img = ros_numpy.numpify(msg)
            enc = msg.encoding.lower()
            if enc == "rgb8":
                img = img[..., ::-1].copy()                  # RGB -> BGR
            elif enc == "rgba8":
                img = img[..., :3][..., ::-1].copy()         # RGBA -> BGR
            elif enc == "bgra8":
                img = img[..., :3].copy()                    # drop alpha
            with self._cam_lock:
                self.latest_rgb = img
        except Exception as e:
            rospy.logwarn(f"RGB decode error: {e}")

    def _depth_cb(self, msg):
        """Convert sensor_msgs/Image -> uint16 numpy array using ros_numpy."""
        try:
            img = ros_numpy.numpify(msg)
            if img.dtype != np.uint16:
                img = img.astype(np.uint16)
            with self._cam_lock:
                self.latest_depth = img
        except Exception as e:
            rospy.logwarn(f"Depth decode error: {e}")

    def _camera_info_cb(self, msg):
        """Cache camera intrinsics on first receipt. K = [fx,0,cx,0,fy,cy,0,0,1]."""
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
        """rospy.Service callback - called in a dedicated thread per request."""
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

        # --- 5. Optionally validate the debug overlay (size cap) --------
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
                    f"debug_overlay_b64 too large ({overlay_bytes} bytes) — dropping. "
                    f"Limit is {self.response_max_npz_mb} MB."
                )
                overlay_b64 = ""
                overlay_bytes = 0

        # --- 6. Build response ------------------------------------------
        rospy.loginfo(
            f"DIFT plan_action OK | grasp_uv={dift_result.get('keypoint_uv')} "
            f"place_uv={dift_result.get('place_uv')} "
            f"is_final_stage={dift_result.get('is_final_stage')} "
            f"num_candidates={dift_result.get('num_candidates')}"
        )

        payload = {
            "status":                "success",
            "message":               dift_result.get(
                "message",
                "DIFT planning complete. Camera-frame keypoints returned."
            ),

            # Grasp keypoint
            "keypoint_uv":           dift_result.get("keypoint_uv"),
            "keypoint_3d_camera":    dift_result.get("keypoint_3d_camera"),
            "depth_at_grasp_m":      dift_result.get("depth_at_grasp_m"),

            # Place / alignment target
            "place_uv":              dift_result.get("place_uv"),
            "place_3d_camera":       dift_result.get("place_3d_camera"),
            "depth_at_place_m":      dift_result.get("depth_at_place_m"),

            # VLM planner outputs
            "stage_description":     dift_result.get("stage_description"),
            "is_final_stage":        bool(dift_result.get("is_final_stage", False)),
            "reasoning":             dift_result.get("reasoning", ""),

            # Diagnostics
            "num_candidates":        dift_result.get("num_candidates"),
            "selected_indices":      dift_result.get("selected_indices", {}),

            # Debug overlay (annotated image with all candidates)
            # "debug_overlay_b64":     overlay_b64,
            # "debug_overlay_bytes":   overlay_bytes,

            # Timings forwarded from DIFT server
            "timings":               dift_result.get("timings", {}),
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
    #  DIFT WebSocket call                                                 #
    # ------------------------------------------------------------------ #

    def _call_dift(self, rgb, depth, fx, fy, cx, cy,
                   text_prompt, task, stage_idx, stage_history, max_stages):
        """
        Send RGB + depth + task info to the DIFT /plan_action endpoint.

        Returns:
            tuple: (result_dict, error_string) — exactly one is None.
        """
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
        """Populate *response* as a failure and log the error."""
        rospy.logerr(f"get_keypoints service error: {msg}")
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
    """Initialize and spin the GetKeypointsNode."""
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