#!/usr/bin/env python3
"""
ROS1 Noetic service node: FoundationStereo + Grounded SAM2 + AnyGrasp Pipeline
-------------------------------------------------------------------------------

Same service as the original detect_objects_service_node_ros1.py, but the
RGB/depth/intrinsics inputs to SAM2 now come from a FoundationStereo
WebSocket server (fs_ws_server.py) instead of ROS image topics.

Pipeline per request:
    1. {"trigger": true} → FoundationStereo WS  →  RGB + depth + fx,fy,cx,cy
    2. Grounded SAM2  (segments object given text_prompt)
    3. AnyGrasp        (best grasp from segmented PC)
    4. TF transform   (camera frame → base_link) + grasp orientation fixup

Service: /robot/perception/detect_objects  (robot_api_interfaces/RobotCommand)

Request (JSON in .req):
    {"text_prompt": "glass"}

Response (JSON in .data):
    {
        "status": "success",
        "best_grasp": {
            "translation_wrt_base": [x, y, z],
            "quaternion_wrt_base":  {"x":…, "y":…, "z":…, "w":…},
            "score":  float,
            "width":  float   # metres
        }
    }

Wire format from FoundationStereo WS server (fs_ws_server.py):
    {
        "status":      "success",
        "rgb":         "<base64 PNG, BGR uint8>",
        "depth":       "<base64 PNG, uint16, millimetres>",
        "fx":          float,
        "fy":          float,
        "cx":          float,
        "cy":          float,
        "depth_scale": 0.001    # multiply raw uint16 value by this to get metres
    }

Usage:
    rosrun <your_pkg> detect_objects_with_fs_service_node_ros1.py \\
        _foundation_stereo_url:=ws://localhost:8768/foundation_stereo
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

from robot_api_interfaces.srv import RobotCommand, RobotCommandResponse
from robot_api_interfaces.msg import ResultCode


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────
def _rotation_matrix_to_quaternion(R):
    """Convert a 3×3 rotation matrix to {x, y, z, w} (Shepperd's method)."""
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


def _quaternion_multiply(q_a, q_b):
    """Hamilton product of two {x, y, z, w} quaternion dicts."""
    ax, ay, az, aw = q_a["x"], q_a["y"], q_a["z"], q_a["w"]
    bx, by, bz, bw = q_b["x"], q_b["y"], q_b["z"], q_b["w"]
    return {
        "x": float(aw * bx + ax * bw + ay * bz - az * by),
        "y": float(aw * by - ax * bz + ay * bw + az * bx),
        "z": float(aw * bz + ax * by - ay * bx + az * bw),
        "w": float(aw * bw - ax * bx - ay * by - az * bz),
    }


def _decode_png_b64(b64_str: str, expected_dtype=np.uint8):
    """
    Decode a base64-encoded PNG string into a numpy array.

    For uint16 depth PNGs cv2.IMREAD_UNCHANGED is essential — the default
    IMREAD_COLOR flag would silently downcast to uint8.
    """
    raw = base64.b64decode(b64_str)
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("cv2.imdecode returned None — corrupt PNG payload?")
    if expected_dtype is not None and img.dtype != expected_dtype:
        img = img.astype(expected_dtype)
    return img


# ────────────────────────────────────────────────────────────────────────────
# Service node
# ────────────────────────────────────────────────────────────────────────────
class SegmentAndGraspNode:
    """
    Service node — chains FoundationStereo → SAM2 → AnyGrasp → TF.
    """

    def __init__(self):
        # ── Server URLs ────────────────────────────────────────────────
        self.fs_url       = rospy.get_param(
            "~foundation_stereo_url",
            "ws://localhost:8768/foundation_stereo")
        self.sam2_url     = rospy.get_param(
            "~sam2_url", "ws://10.158.54.164:8766/detect_objects")
        self.anygrasp_url = rospy.get_param(
            "~anygrasp_url", "ws://10.158.54.164:8767/get_saved_grasp")

        # ── Timeouts ───────────────────────────────────────────────────
        #   FS timeout should cover cam_server round-trip + GPU inference
        #   (typically 3–8 s depending on resolution and GPU).
        self.fs_timeout_sec    = float(rospy.get_param("~fs_timeout_sec",    30.0))
        self.sam2_timeout_sec  = float(rospy.get_param("~sam2_timeout_sec",  15.0))
        self.grasp_timeout_sec = float(rospy.get_param("~grasp_timeout_sec", 15.0))

        # ── TF parameters ──────────────────────────────────────────────
        self.camera_frame   = rospy.get_param("~camera_frame",
                                              "cam_scene_depth_optical_frame")
        self.base_frame     = rospy.get_param("~base_frame", "panda_link0")
        self.tf_timeout_sec = float(rospy.get_param("~tf_timeout_sec", 2.0))

        # ── aiohttp max message sizes ───────────────────────────────────
        # FS response can be large (two images at full IR resolution).
        # 64 MB is a safe upper bound.
        self._fs_max_msg_mb    = int(rospy.get_param("~fs_max_msg_mb",    64))
        self._sam2_max_msg_mb  = int(rospy.get_param("~sam2_max_msg_mb",  50))
        self._grasp_max_msg_mb = int(rospy.get_param("~grasp_max_msg_mb", 10))

        # ── TF listener / broadcaster ──────────────────────────────────
        self._tf_listener    = tf.TransformListener()
        self._tf_broadcaster = tf.TransformBroadcaster()

        # ── Service ────────────────────────────────────────────────────
        self._service = rospy.Service(
            "/robot/perception/detect_objects",
            RobotCommand,
            self._handle_request,
        )

        rospy.loginfo(
            "\nSegmentAndGraspNode (FoundationStereo backend) ready.\n"
            f"  Service           : /robot/perception/detect_objects\n"
            f"  FoundationStereo  : {self.fs_url}  "
            f"(timeout {self.fs_timeout_sec}s)\n"
            f"  SAM2              : {self.sam2_url}\n"
            f"  AnyGrasp          : {self.anygrasp_url}\n"
            f"  Camera frame      : {self.camera_frame}\n"
            f"  Base frame        : {self.base_frame}"
        )

    # ────────────────────────────────────────────────────────────────────
    # Service handler
    # ────────────────────────────────────────────────────────────────────
    def _handle_request(self, request):
        rospy.loginfo(f"Service request received: {request.req}")
        response = RobotCommandResponse()

        # ── 1. Parse text prompt ───────────────────────────────────────
        try:
            req_data    = json.loads(request.req)
            text_prompt = req_data.get("text_prompt", "").strip() or "object"
        except (json.JSONDecodeError, ValueError) as e:
            return self._fail(response, f"Bad request: {e}")

        # ── 2. Pull RGB + depth + intrinsics from FoundationStereo WS ──
        rospy.loginfo("Calling FoundationStereo WS ...")
        fs_result, fs_err = self._call_foundation_stereo()

        if fs_err:
            return self._fail(response, f"FoundationStereo failed: {fs_err}")

        if fs_result.get("status") != "success":
            return self._fail(
                response,
                f"FoundationStereo returned non-success: "
                f"{fs_result.get('message', 'unknown')}"
            )

        try:
            # rgb  : BGR uint8  PNG  (H×W×3)
            # depth: uint16 PNG in millimetres (H×W)
            rgb         = _decode_png_b64(fs_result["rgb"],   np.uint8)
            depth       = _decode_png_b64(fs_result["depth"], np.uint16)
            fx          = float(fs_result["fx"])
            fy          = float(fs_result["fy"])
            cx          = float(fs_result["cx"])
            cy          = float(fs_result["cy"])
            # depth_scale: multiply raw uint16 mm value → metres  (0.001)
            depth_scale = float(fs_result.get("depth_scale", 0.001))
        except (KeyError, ValueError) as e:
            return self._fail(response, f"Bad FoundationStereo payload: {e}")

        rospy.loginfo(
            f"FS frames received: rgb={rgb.shape} depth={depth.shape} "
            f"fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f} "
            f"depth_scale={depth_scale}"
        )

        # ── 3. Call Grounded SAM2 ──────────────────────────────────────
        rospy.loginfo(f"Calling Grounded SAM2 | prompt='{text_prompt}'")
        sam2_result, sam2_err = self._call_sam2(
            rgb, depth, fx, fy, cx, cy, text_prompt, depth_scale
        )

        if sam2_err:
            return self._fail(response, f"Grounded SAM2 failed: {sam2_err}")

        if sam2_result.get("status") != "success":
            return self._fail(
                response,
                f"Grounded SAM2 returned non-success: "
                f"{sam2_result.get('message', json.dumps(sam2_result))}"
            )

        rospy.loginfo("SAM2 succeeded — mask written to NPZ.")

        # ── 4. Call AnyGrasp ───────────────────────────────────────────
        rospy.loginfo("Calling AnyGrasp ...")
        grasp_result, grasp_err = self._call_anygrasp()

        if grasp_err:
            return self._fail(response, f"AnyGrasp failed: {grasp_err}")

        if grasp_result.get("status") != "success":
            return self._fail(
                response,
                f"AnyGrasp returned non-success: "
                f"{grasp_result.get('message', json.dumps(grasp_result))}"
            )

        # ── 5. Camera-frame → base-frame + EE convention fixup ─────────
        best = grasp_result["best_grasp"]
        t    = best["translation"]
        q    = _rotation_matrix_to_quaternion(best["rotation"])

        rospy.loginfo(
            f"Grasp (cam frame) | score={best['score']:.4f} "
            f"width={best['width']:.4f}m  "
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
            # Same orientation fixups as original node
            q_180z  = {"x": 0.0, "y": 0.0, "z": 1.0, "w": 0.0}
            q_base  = _quaternion_multiply(q_base, q_180z)
            q_90yz  = {"x": 0.0, "y": -0.7071068, "z": 0.0, "w": 0.7071068}
            q_base  = _quaternion_multiply(q_base, q_90yz)
            q__180z = {"x": 0.0, "y": 0.0, "z": 1.0, "w": 0.0}
            q_base  = _quaternion_multiply(q_base, q__180z)

            # Apply 0.15 m back-off along grasp local -Z
            try:
                shift_m = 0.15
                Q = [q_base["x"], q_base["y"], q_base["z"], q_base["w"]]
                R = tft.quaternion_matrix(Q)[0:3, 0:3]
                shift_global = R.dot(np.array([0.0, 0.0, -shift_m]))
                t_shift = [
                    float(t_base[0] + shift_global[0]),
                    float(t_base[1] + shift_global[1]),
                    float(t_base[2] + shift_global[2]),
                ]

                now = rospy.Time.now()
                self._tf_broadcaster.sendTransform(
                    (t_shift[0], t_shift[1], t_shift[2]),
                    (q_base["x"], q_base["y"], q_base["z"], q_base["w"]),
                    now,
                    "shifted_grasp",
                    self.base_frame,
                )

                rospy.sleep(0.05)
                self._tf_listener.waitForTransform(
                    self.base_frame, "shifted_grasp",
                    rospy.Time(0),
                    rospy.Duration(self.tf_timeout_sec),
                )
                trans_s, rot_s = self._tf_listener.lookupTransform(
                    self.base_frame, "shifted_grasp", rospy.Time(0)
                )

                t_base = [float(trans_s[0]), float(trans_s[1]), float(trans_s[2])]
                q_base = {"x": float(rot_s[0]), "y": float(rot_s[1]),
                          "z": float(rot_s[2]), "w": float(rot_s[3])}

                rospy.loginfo(
                    f"Shifted pose in '{self.base_frame}' | "
                    f"xyz=({t_base[0]:.4f}, {t_base[1]:.4f}, {t_base[2]:.4f})  "
                    f"quat=({q_base['x']:.4f}, {q_base['y']:.4f}, "
                    f"{q_base['z']:.4f}, {q_base['w']:.4f})"
                )

            except Exception as e:
                rospy.logwarn(f"Shifted-grasp TF lookup failed: {e}")

        # ── 6. Build response ──────────────────────────────────────────
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
        response.result_code.message     = (
            "FoundationStereo + SAM2 + AnyGrasp succeeded."
        )
        response.data = json.dumps(payload)
        return response

    # ────────────────────────────────────────────────────────────────────
    # FoundationStereo WS call
    # ────────────────────────────────────────────────────────────────────
    def _call_foundation_stereo(self):
        """
        Returns (result_dict, error_string).  Exactly one is None.

        Sends {"trigger": true} to fs_ws_server.py and waits for the
        response containing RGB + depth + intrinsics.  The response can be
        large (two base64-encoded full-resolution images), so max_msg_mb is
        set generously via the ~fs_max_msg_mb ROS param.
        """
        try:
            result = asyncio.run(
                self._ws_send_recv(
                    self.fs_url,
                    json.dumps({"trigger": True}),
                    self.fs_timeout_sec,
                    max_msg_mb=self._fs_max_msg_mb,
                )
            )
            return result, None

        except aiohttp.ClientConnectorError:
            return None, (f"Could not connect to FoundationStereo server "
                          f"at {self.fs_url}.  Is fs_ws_server.py running?")
        except aiohttp.WSServerHandshakeError as e:
            return None, f"FS WebSocket handshake failed: {e}"
        except asyncio.TimeoutError:
            return None, (f"FS server timed out after {self.fs_timeout_sec}s — "
                          "consider increasing ~fs_timeout_sec if the GPU is slow.")
        except json.JSONDecodeError as e:
            return None, f"FS response is not valid JSON: {e}"
        except Exception as e:
            return None, f"Unexpected FS error: {e}\n{traceback.format_exc()}"

    # ────────────────────────────────────────────────────────────────────
    # SAM2 WS call
    # ────────────────────────────────────────────────────────────────────
    def _call_sam2(self, rgb, depth, fx, fy, cx, cy, text_prompt, depth_scale):
        """Returns (result_dict, error_string).  Exactly one is None."""
        try:
            payload = {
                "text_prompt": text_prompt,
                "rgb":         self._encode_rgb(rgb),
                "depth":       self._encode_depth(depth),
                "fx": fx, "fy": fy, "cx": cx, "cy": cy,
                "depth_scale": float(depth_scale),
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
                    self.sam2_timeout_sec,
                    max_msg_mb=self._sam2_max_msg_mb,
                )
            )
            return result, None

        except aiohttp.ClientConnectorError:
            return None, f"Could not connect to SAM2 at {self.sam2_url}"
        except aiohttp.WSServerHandshakeError as e:
            return None, f"SAM2 WebSocket handshake failed: {e}"
        except asyncio.TimeoutError:
            return None, f"SAM2 server timed out after {self.sam2_timeout_sec}s"
        except json.JSONDecodeError as e:
            return None, f"SAM2 response is not valid JSON: {e}"
        except Exception as e:
            return None, f"Unexpected SAM2 error: {e}\n{traceback.format_exc()}"

    # ────────────────────────────────────────────────────────────────────
    # AnyGrasp WS call
    # ────────────────────────────────────────────────────────────────────
    def _call_anygrasp(self):
        """Returns (result_dict, error_string).  Exactly one is None."""
        try:
            result = asyncio.run(
                self._ws_send_recv(
                    self.anygrasp_url,
                    json.dumps({"trigger": True}),
                    self.grasp_timeout_sec,
                    max_msg_mb=self._grasp_max_msg_mb,
                )
            )
            return result, None

        except aiohttp.ClientConnectorError:
            return None, f"Could not connect to AnyGrasp at {self.anygrasp_url}"
        except aiohttp.WSServerHandshakeError as e:
            return None, f"AnyGrasp WebSocket handshake failed: {e}"
        except asyncio.TimeoutError:
            return None, f"AnyGrasp timed out after {self.grasp_timeout_sec}s"
        except json.JSONDecodeError as e:
            return None, f"AnyGrasp response is not valid JSON: {e}"
        except Exception as e:
            return None, f"Unexpected AnyGrasp error: {e}\n{traceback.format_exc()}"

    # ────────────────────────────────────────────────────────────────────
    # Shared aiohttp WS primitive
    # ────────────────────────────────────────────────────────────────────
    async def _ws_send_recv(self, url, payload_str, timeout_sec, max_msg_mb=64):
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

    # ────────────────────────────────────────────────────────────────────
    # TF helpers
    # ────────────────────────────────────────────────────────────────────
    def _transform_pose_to_base(self, translation, quaternion):
        try:
            self._tf_listener.waitForTransform(
                self.base_frame, self.camera_frame,
                rospy.Time(0), rospy.Duration(self.tf_timeout_sec),
            )
            trans, rot = self._tf_listener.lookupTransform(
                self.base_frame, self.camera_frame, rospy.Time(0)
            )

            T_base_cam = tft.quaternion_matrix(rot)
            T_base_cam[0:3, 3] = trans

            q_list = [quaternion["x"], quaternion["y"],
                      quaternion["z"], quaternion["w"]]
            T_pose_cam = tft.quaternion_matrix(q_list)
            T_pose_cam[0:3, 3] = translation

            T_pose_base = np.dot(T_base_cam, T_pose_cam)
            t_base = [T_pose_base[0, 3], T_pose_base[1, 3], T_pose_base[2, 3]]
            q_out  = tft.quaternion_from_matrix(T_pose_base)
            q_base = {"x": float(q_out[0]), "y": float(q_out[1]),
                      "z": float(q_out[2]), "w": float(q_out[3])}
            return t_base, q_base, None

        except (tf.LookupException, tf.ConnectivityException,
                tf.ExtrapolationException) as e:
            return None, None, f"{type(e).__name__}: {e}"
        except Exception as e:
            return None, None, (f"Unexpected TF error: {e}\n"
                                f"{traceback.format_exc()}")

    def _get_cam_to_base_quat(self):
        try:
            self._tf_listener.waitForTransform(
                self.base_frame, self.camera_frame,
                rospy.Time(0), rospy.Duration(self.tf_timeout_sec),
            )
            _trans, rot = self._tf_listener.lookupTransform(
                self.base_frame, self.camera_frame, rospy.Time(0)
            )
            rospy.loginfo(
                f"cam_to_base_quat: [{rot[0]:.4f}, {rot[1]:.4f}, "
                f"{rot[2]:.4f}, {rot[3]:.4f}]"
            )
            return list(rot)
        except Exception as e:
            rospy.logwarn(f"cam_to_base_quat lookup failed: {e}")
            return None

    # ────────────────────────────────────────────────────────────────────
    # Image encoding helpers
    # ────────────────────────────────────────────────────────────────────
    @staticmethod
    def _encode_rgb(bgr):
        """Re-encode the BGR image received from FS server for forwarding to SAM2."""
        ok, buf = cv2.imencode(".png", bgr)
        if not ok:
            raise RuntimeError("cv2.imencode failed for RGB image.")
        return base64.b64encode(buf.tobytes()).decode("utf-8")

    @staticmethod
    def _encode_depth(depth):
        """Re-encode the uint16 depth image received from FS server for SAM2."""
        if depth.dtype != np.uint16:
            depth = depth.astype(np.uint16)
        ok, buf = cv2.imencode(".png", depth)
        if not ok:
            raise RuntimeError("cv2.imencode failed for depth image.")
        return base64.b64encode(buf.tobytes()).decode("utf-8")

    # ────────────────────────────────────────────────────────────────────
    # Failure helper
    # ────────────────────────────────────────────────────────────────────
    def _fail(self, response, msg):
        rospy.logerr(f"detect_objects service error: {msg}")
        response.result_code.result_code = ResultCode.FAILURE
        response.result_code.message     = msg
        response.data = json.dumps({"status": "error", "message": msg})
        return response

    def spin(self):
        rospy.spin()


def main():
    rospy.init_node("detect_objects_service", anonymous=False)
    try:
        rospy.loginfo("Creating SegmentAndGraspNode (FoundationStereo backend) ...")
        node = SegmentAndGraspNode()
        rospy.loginfo("Spinning ...")
        node.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS interrupt — shutting down.")
    except Exception as e:
        rospy.logerr(f"Fatal error: {e}\n{traceback.format_exc()}")
    finally:
        rospy.loginfo("Shutdown complete.")


if __name__ == "__main__":
    main()