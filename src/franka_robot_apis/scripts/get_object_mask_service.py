#!/usr/bin/env python3
"""
ROS1 Noetic service node: Grounded SAM2 get_object_mask Pipeline
-----------------------------------------------------------------
Service: /robot/perception/get_object_mask  (robot_api_interfaces/RobotCommand)

Request (JSON string in .req field):
{
    "text_prompt": "glass"
}

Response (JSON string in .data field):
{
    "status": "success",
    "message": "Segmentation complete. Returning mask/rgb/depth as lossless NPZ base64 payload.",
    "data_encoding": "npz_base64",
    "npz_base64": "UEs...",
    "npz_bytes": 295448,
    "npz_fields": ["rgb", "depth", "mask", "intrinsics"],
    "decoded": {
        "rgb":        {"shape": [270, 480, 3], "dtype": "uint8"},
        "depth":      {"shape": [270, 480],    "dtype": "uint16"},
        "mask":       {"shape": [270, 480],    "dtype": "uint8"},
        "intrinsics": {"shape": [3, 3],        "dtype": "float64"}
    },
    "mask_pixels":      57420,
    "num_detections":   1,
    "detected_labels":  ["purple floor"],
    "confidences":      [0.6858],
    "has_orientation":  false
}

ROS1 usage:
    rosrun franka_robot_apis get_object_mask_service.py

    rosservice call /robot/perception/get_object_mask \
        '{"req": "{\"text_prompt\": \"floor\"}"}'
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
class GetObjectMaskNode:
    """
    ROS1 service node wrapping the Grounded SAM2 /get_object_mask WebSocket
    endpoint.

    Captures an RGB + depth frame, ships them to the SAM2 server, validates
    the returned NPZ payload, and forwards it verbatim to the caller.
    """

    def __init__(self):
        # ------------------------------------------------------------------ #
        #  Parameters                                                          #
        # ------------------------------------------------------------------ #
        self.sam2_url = rospy.get_param(
            "~sam2_url", "ws://10.158.54.164:8766/get_object_mask"
        )

        # Camera topics — same RealSense serial used in detect_objects node
        self.rgb_topic         = rospy.get_param("~rgb_topic",         "/realsense/scene/color/image_raw")
        self.depth_topic       = rospy.get_param("~depth_topic",       "/realsense/scene/aligned_depth_to_color/image_raw")
        self.camera_info_topic = rospy.get_param("~camera_info_topic", "/realsense/scene/aligned_depth_to_color/camera_info")

        # Fallback intrinsics (used only if camera_info never arrives)
        self.fx_default = float(rospy.get_param("~fx", 752.0038452148438))
        self.fy_default = float(rospy.get_param("~fy", 751.7178344726562))
        self.cx_default = float(rospy.get_param("~cx", 628.4379272460938))
        self.cy_default = float(rospy.get_param("~cy", 335.1157531738281))

        # Timeouts
        self.camera_wait_sec     = float(rospy.get_param("~camera_wait_sec",     1.0))
        self.sam2_timeout_sec    = float(rospy.get_param("~sam2_timeout_sec",    15.0))
        self.response_max_npz_mb = float(rospy.get_param("~response_max_npz_mb", 25.0))

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
        # CameraInfo is small and typically latched — default reliable QoS fine
        rospy.Subscriber(
            self.camera_info_topic, CameraInfo, self._camera_info_cb,
            queue_size=10,
        )

        # ------------------------------------------------------------------ #
        #  Service                                                             #
        # ------------------------------------------------------------------ #
        self._service = rospy.Service(
            "/robot/perception/get_object_mask",
            RobotCommand,
            self._handle_request,
        )

        rospy.loginfo(
            "\nGetObjectMaskNode (ROS1) ready.\n"
            f"  Service     : /robot/perception/get_object_mask\n"
            f"  SAM2        : {self.sam2_url}\n"
            f"  RGB         : {self.rgb_topic}\n"
            f"  Depth       : {self.depth_topic}\n"
            f"  CamInfo     : {self.camera_info_topic}"
        )

    # ------------------------------------------------------------------ #
    #  Camera callbacks — ros_numpy replaces CvBridge                     #
    # ------------------------------------------------------------------ #

    def _rgb_cb(self, msg):
        """
        Convert sensor_msgs/Image -> BGR uint8 numpy array using ros_numpy.

        ros_numpy.numpify returns:
          rgb8  -> (H,W,3) uint8 in RGB order  -> flip to BGR for OpenCV
          bgr8  -> (H,W,3) uint8 in BGR order  -> use as-is
          rgba8 -> (H,W,4) uint8 RGBA          -> drop alpha, flip to BGR
          bgra8 -> (H,W,4) uint8 BGRA          -> drop alpha, already BGR
        """
        try:
            img = ros_numpy.numpify(msg)          # raw numpy from message
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

        ros_numpy.numpify on a mono16 / 16UC1 depth image returns (H,W)
        uint16 with raw millimetre values — exactly what _encode_depth needs.
        """
        try:
            img = ros_numpy.numpify(msg)          # (H,W) uint16 typically
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

        Args:
            request (RobotCommand.Request): .req holds the JSON string

        Returns:
            RobotCommandResponse
        """
        rospy.loginfo(f"Get-object-mask request received: {request.req}")
        response = RobotCommandResponse()

        # --- 1. Parse request -------------------------------------------
        try:
            req_data    = json.loads(request.req)
            text_prompt = req_data.get("text_prompt", "").strip() or "object"
        except (json.JSONDecodeError, ValueError) as e:
            return self._fail(response, f"Bad request: {e}")

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
            rgb             = self.latest_rgb.copy()
            depth           = self.latest_depth.copy()
            fx, fy, cx, cy  = self.fx, self.fy, self.cx, self.cy

        if not self.camera_info_received:
            rospy.logwarn(
                "CameraInfo not yet received — using fallback intrinsics. "
                f"fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}"
            )
        
        # get homogeneous transform from camera frame to robot base frame if available
        T_base_cam = np.eye(4)  # default to identity if no TF available
        try:
            import tf2_ros
            import tf2_geometry_msgs
            tf_buffer = tf2_ros.Buffer()
            tf_listener = tf2_ros.TransformListener(tf_buffer)
            transform = tf_buffer.lookup_transform(
                self.base_frame,           # target frame (robot base)
                self.camera_frame,  # source frame (RealSense RGB camera)
                rospy.Time(0),  # get latest available
                rospy.Duration(self.tf_timeout_sec)  # timeout
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
            T_base_cam = hom_matrix.tolist()  # convert to regular Python list for JSON serialization

            rospy.loginfo("Successfully obtained T_base_cam from TF.")
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(f"Could not obtain T_base_cam from TF: {e}. Using identity.")

        # --- 3. Call Grounded SAM2 /get_object_mask ---------------------
        rospy.loginfo(
            f"Calling Grounded SAM2 get_object_mask | prompt='{text_prompt}'"
        )
        sam2_result, sam2_err = self._call_sam2(
            rgb, depth, fx, fy, cx, cy, text_prompt, T_base_cam
        )

        if sam2_err:
            return self._fail(response, f"Grounded SAM2 failed: {sam2_err}")

        if sam2_result.get("status") != "success":
            return self._fail(
                response,
                f"Grounded SAM2 returned non-success status: "
                f"{sam2_result.get('message', json.dumps(sam2_result))}"
            )

        # --- 4. Validate / summarise the NPZ payload --------------------
        npz_b64       = sam2_result.get("npz_base64")
        data_encoding = sam2_result.get("data_encoding")

        if data_encoding != "npz_base64" or not npz_b64:
            return self._fail(
                response,
                "Grounded SAM2 response missing lossless npz_base64 payload."
            )

        decode_summary, npz_byte_count, decode_err = self._decode_npz_summary(npz_b64)
        if decode_err:
            return self._fail(response, f"Failed decoding npz_base64: {decode_err}")

        max_bytes = int(self.response_max_npz_mb * 1024 * 1024)
        if npz_byte_count > max_bytes:
            return self._fail(
                response,
                f"NPZ payload too large ({npz_byte_count} bytes). "
                f"Limit is {self.response_max_npz_mb} MB."
            )

        rospy.loginfo(
            f"Decoded npz_base64 successfully | npz_bytes={npz_byte_count} "
            f"| fields={list(decode_summary.keys())}"
        )
        for field_name, field_info in decode_summary.items():
            rospy.loginfo(
                f"  field='{field_name}' shape={field_info['shape']} "
                f"dtype={field_info['dtype']}"
            )

        # --- 5. Build and return response -
        payload = {
            "status":       "success",
            "message":      sam2_result.get(
                "message",
                "Segmentation complete. Returning mask/rgb/depth as lossless NPZ base64 payload.",
            ),
            "data_encoding":    "npz_base64",
            "npz_base64":       npz_b64,
            "npz_bytes":        npz_byte_count,
            "npz_fields":       list(decode_summary.keys()),
            "decoded":          decode_summary,
            "mask_pixels":      sam2_result.get("mask_pixels"),
            "num_detections":   sam2_result.get("num_detections"),
            "detected_labels":  sam2_result.get("detected_labels", []),
            "confidences":      sam2_result.get("confidences", []),
            "has_orientation":  sam2_result.get("has_orientation", False),
            "has_T_base_cam":      sam2_result.get("has_T_base_cam", None),
        }

        response.result_code.result_code = ResultCode.SUCCESS
        response.result_code.message     = "Object mask segmentation succeeded."
        response.data                    = json.dumps(payload)

        rospy.loginfo(
            f"get_object_mask detected labels: "
            f"{json.dumps(payload.get('detected_labels', []))}"
        )
        return response

    # ------------------------------------------------------------------ #
    #  NPZ decode helper                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _decode_npz_summary(npz_b64):
        """
        Decode a base64-encoded NPZ blob and return per-field shape/dtype info.

        Args:
            npz_b64 (str): base64-encoded NPZ bytes

        Returns:
            tuple: (summary_dict, byte_count, error_string)
                   On success  -> ({"field": {"shape": [...], "dtype": "..."}}, int, None)
                   On failure  -> (None, None, error_string)
        """
        try:
            npz_bytes = base64.b64decode(npz_b64)
            summary   = {}
            with np.load(io.BytesIO(npz_bytes), allow_pickle=False) as npz_data:
                for key in npz_data.files:
                    arr = npz_data[key]
                    summary[str(key)] = {
                        "shape": [int(v) for v in arr.shape],
                        "dtype": str(arr.dtype),
                    }
            return summary, len(npz_bytes), None
        except Exception as e:
            return None, None, str(e)

    # ------------------------------------------------------------------ #
    #  SAM2 WebSocket call                                                 #
    # ------------------------------------------------------------------ #

    def _call_sam2(self, rgb, depth, fx, fy, cx, cy, text_prompt, T_base_cam=np.eye(4)):
        """
        Send RGB + depth + prompt to the SAM2 /get_object_mask endpoint.

        Returns:
            tuple: (result_dict, error_string) — exactly one is None.
        """
        try:
            payload = {
                "text_prompt": text_prompt,
                "rgb":         self._encode_rgb(rgb),
                "depth":       self._encode_depth(depth),
                "fx": fx, "fy": fy, "cx": cx, "cy": cy,
                "T_base_cam":  T_base_cam,
                "mode":        "get_object_mask",   # tells SAM2 which endpoint mode
            }

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
    #  Shared async WebSocket primitive  (unchanged from ROS2 version)    #
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
    #  Image encoding helpers  (unchanged from ROS2 version)              #
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

        Args:
            response (RobotCommandResponse): to mutate
            msg      (str):                 human-readable error

        Returns:
            RobotCommandResponse
        """
        rospy.logerr(f"get_object_mask service error: {msg}")
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
    """Initialize and spin the GetObjectMaskNode."""
    rospy.init_node("get_object_mask_service", anonymous=False)

    try:
        rospy.loginfo("Creating GetObjectMaskNode ...")
        node = GetObjectMaskNode()
        rospy.loginfo("GetObjectMaskNode spinning ...")
        node.spin()

    except rospy.ROSInterruptException:
        rospy.loginfo("ROS interrupt — shutting down GetObjectMaskNode.")
    except Exception as e:
        rospy.logerr(f"Fatal error: {e}\n{traceback.format_exc()}")
    finally:
        rospy.loginfo("GetObjectMaskNode shutdown complete.")


if __name__ == "__main__":
    main()