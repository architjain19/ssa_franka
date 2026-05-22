#!/usr/bin/env python3
"""
ROS1 Noetic service node: AnyGrasp-from-Mask Pipeline
-------------------------------------------------------------

Standalone grasp service. The caller is expected to have already obtained a
segmentation NPZ (rgb / depth / mask / intrinsics) from some other service
- typically /robot/perception/get_object_mask - and passes the base64 NPZ
string to this service. This node:

  1. Repacks the incoming NPZ with cam_to_base_quat + depth_scale so AnyGrasp
     can horizontal-filter correctly.
  2. Sends it to AnyGrasp's payload-mode WebSocket endpoint (/get_grasp_config).
  3. Transforms the resulting grasp pose into the robot base frame and
     applies the same orientation / offset fix-ups used by
     /robot/perception/detect_objects so the response format is identical.

Service: /robot/perception/get_grasp_config
         (robot_api_interfaces/RobotCommand)

Typical usage:
    mask_resp  = /robot/perception/get_object_mask  (separate service)
    mask_data  = json.loads(mask_resp.data)
    grasp_req  = {
        "npz_base64": mask_data["npz_base64"],
        "camera":     mask_data["camera"],
    }
    grasp_resp = /robot/perception/get_grasp_config  (this service)

Request (JSON string in .req):
{
    "npz_base64": "UEs...",        # REQUIRED - base64 NPZ from get_object_mask
    "camera":     "scene",         # REQUIRED - "wrist" or "scene", used for TF
    "x_offset":   0.0,             # optional - local-frame shift along grasp X (m)
    "y_offset":   0.0,             # optional - local-frame shift along grasp Y (m)
    "z_offset":   0.0              # optional - local-frame backward shift along grasp Z (m)
}

Response (JSON string in .data):
{
    "status": "success",
    "best_grasp": {
        "translation_wrt_base": [x, y, z],
        "quaternion_wrt_base":  {"x": ..., "y": ..., "z": ..., "w": ...},
        "score": 0.2,
        "width": 0.047
    },
    "camera": "scene"
}

Usage:
    rosrun <your_pkg> get_grasp_config_service.py
"""

import json
import asyncio
import base64
import io
import traceback

import numpy as np
import aiohttp

import rospy
import tf
import tf.transformations as tft

from robot_api_interfaces.srv import RobotCommand, RobotCommandResponse
from robot_api_interfaces.msg import ResultCode


# Allowed values for the "camera" field in the request JSON.
VALID_CAMERAS = ("wrist", "scene")


# ---------------------------------------------------------------------------
# Quaternion helpers (identical to detect_objects, kept local so this node
# has no shared-utility dependency on the other perception nodes)
# ---------------------------------------------------------------------------
def _rotation_matrix_to_quaternion(R):
    """Convert a 3x3 rotation matrix to a quaternion dict {x, y, z, w}."""
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
    """Hamilton product q_a * q_b for quaternion dicts {x, y, z, w}."""
    ax, ay, az, aw = q_a["x"], q_a["y"], q_a["z"], q_a["w"]
    bx, by, bz, bw = q_b["x"], q_b["y"], q_b["z"], q_b["w"]
    return {
        "x": float(aw * bx + ax * bw + ay * bz - az * by),
        "y": float(aw * by - ax * bz + ay * bw + az * bx),
        "z": float(aw * bz + ax * by - ay * bx + az * bw),
        "w": float(aw * bw - ax * bx - ay * by - az * bz),
    }


# ---------------------------------------------------------------------------
# Main node
# ---------------------------------------------------------------------------
class GraspFromMaskNode:
    """
    ROS1 service node that:
      1. Receives a base64 NPZ (rgb / depth / mask / intrinsics) in the request
         - typically taken straight out of a get_object_mask response.
      2. Repacks the NPZ with cam_to_base_quat + depth_scale so AnyGrasp can
         horizontal-filter correctly.
      3. Calls the AnyGrasp WebSocket /get_grasp_config endpoint.
      4. Transforms the grasp pose to the robot base frame and applies the
         same orientation / offset fix-ups used by /robot/perception/detect_objects.
    """

    def __init__(self):
        # ------------------------------------------------------------------ #
        #  Parameters                                                          #
        # ------------------------------------------------------------------ #
        # NOTE: payload-mode endpoint, NOT /get_saved_grasp.
        self.anygrasp_url = rospy.get_param(
            "~anygrasp_url", "ws://10.158.54.164:8767/get_grasp_config"
        )

        # Per-camera TF frames - must match those used by get_object_mask.
        self.camera_frames = {
            "scene": rospy.get_param(
                "~scene_camera_frame", "zed_scene_left_optical_frame"
            ),
            "wrist": rospy.get_param(
                "~wrist_camera_frame", "zed_wrist_left_optical_frame"
            ),
        }

        self.default_camera = rospy.get_param("~default_camera", "scene").lower()
        if self.default_camera not in VALID_CAMERAS:
            rospy.logwarn(
                f"~default_camera='{self.default_camera}' is invalid; "
                f"falling back to 'scene'. Valid options: {VALID_CAMERAS}"
            )
            self.default_camera = "scene"

        self.base_frame        = rospy.get_param("~base_frame", "panda_link0")
        self.tf_timeout_sec    = float(rospy.get_param("~tf_timeout_sec",    2.0))
        self.grasp_timeout_sec = float(rospy.get_param("~grasp_timeout_sec", 15.0))

        # ------------------------------------------------------------------ #
        #  TF                                                                  #
        # ------------------------------------------------------------------ #
        self._tf_listener    = tf.TransformListener()
        self._tf_broadcaster = tf.TransformBroadcaster()

        # ------------------------------------------------------------------ #
        #  Service                                                             #
        # ------------------------------------------------------------------ #
        self._service = rospy.Service(
            "/robot/perception/get_grasp_config",
            RobotCommand,
            self._handle_request,
        )

        rospy.loginfo(
            "\nGraspFromMaskNode (ROS1) ready.\n"
            f"  Service        : /robot/perception/get_grasp_config\n"
            f"  AnyGrasp       : {self.anygrasp_url}\n"
            f"  Default camera : {self.default_camera}\n"
            f"  Base frame     : {self.base_frame}\n"
            f"  Scene frame    : {self.camera_frames['scene']}\n"
            f"  Wrist frame    : {self.camera_frames['wrist']}"
        )

    # ------------------------------------------------------------------ #
    #  Service handler                                                     #
    # ------------------------------------------------------------------ #
    def _handle_request(self, request):
        rospy.loginfo("get_grasp_config request received.")
        response = RobotCommandResponse()

        # --- 1. Parse request ------------------------------------------------
        try:
            req_data = json.loads(request.req)
            x_offset = float(req_data.get("x_offset", 0.0))
            y_offset = float(req_data.get("y_offset", 0.0))
            z_offset = float(req_data.get("z_offset", 0.0))
        except (json.JSONDecodeError, ValueError) as e:
            return self._fail(response, f"Bad request: {e}")

        # --- 1a. Extract npz_base64 -----------------------------------------
        npz_b64 = req_data.get("npz_base64")
        if not isinstance(npz_b64, str) or not npz_b64:
            return self._fail(
                response,
                "Request is missing required string field 'npz_base64'.",
            )

        # --- 1b. Resolve which camera the mask came from --------------------
        camera = req_data.get("camera", self.default_camera)
        if not isinstance(camera, str):
            return self._fail(
                response,
                f"'camera' must be a string, got {type(camera).__name__}.",
            )
        camera = camera.strip().lower()
        if camera not in self.camera_frames:
            return self._fail(
                response,
                f"Invalid 'camera' value: '{camera}'. "
                f"Must be one of: {list(self.camera_frames.keys())}.",
            )
        camera_frame = self.camera_frames[camera]
        rospy.loginfo(
            f"Using camera='{camera}' frame='{camera_frame}' "
            f"npz_bytes_b64={len(npz_b64)} "
            f"offsets=({x_offset:.4f}, {y_offset:.4f}, {z_offset:.4f})"
        )

        # --- 2. Look up cam_to_base_quat for AnyGrasp's horizontal filter ---
        cam_to_base_quat = self._get_cam_to_base_quat(camera_frame)

        # --- 3. Repack NPZ with cam_to_base_quat + depth_scale --------------
        depth_scale = float(
            rospy.get_param(f"/realsense/{camera}/depth_scale", 0.001)
        )
        try:
            npz_b64_for_grasp = self._repack_npz_with_extras(
                npz_b64, cam_to_base_quat, depth_scale,
            )
        except Exception as e:
            return self._fail(
                response,
                f"Failed to repack NPZ for AnyGrasp: {e}\n{traceback.format_exc()}",
            )

        # --- 4. Call AnyGrasp (payload mode) --------------------------------
        rospy.loginfo("Calling AnyGrasp /get_grasp_config ...")
        grasp_result, grasp_err = self._call_anygrasp(npz_b64_for_grasp)
        if grasp_err:
            return self._fail(response, f"AnyGrasp failed: {grasp_err}")
        if grasp_result.get("status") != "success":
            return self._fail(
                response,
                f"AnyGrasp returned non-success status: "
                f"{grasp_result.get('message', json.dumps(grasp_result))}",
            )

        # --- 5. Extract grasp pose, transform to base frame ------------------
        best = grasp_result["best_grasp"]
        t    = best["translation"]
        q    = _rotation_matrix_to_quaternion(best["rotation"])

        rospy.loginfo(
            f"Grasp found | score={best['score']:.4f}  width={best['width']:.4f}m  "
            f"xyz=({t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f})  "
            f"quat=({q['x']:.4f}, {q['y']:.4f}, {q['z']:.4f}, {q['w']:.4f})"
        )

        t_base, q_base, tf_err = self._transform_pose_to_base(t, q, camera_frame)
        if tf_err:
            return self._fail(
                response,
                f"TF transform to '{self.base_frame}' failed: {tf_err}",
            )

        # --- 6. Orientation fix-ups (mirror detect_objects exactly) ---------
        # 180 deg around Z to adjust end-effector orientation.
        q_180z = {"x": 0.0, "y": 0.0, "z": 1.0, "w": 0.0}
        q_base = _quaternion_multiply(q_base, q_180z)
        # Then -90 deg around Y to align gripper approach with +Z in base frame.
        q_90yz = {"x": 0.0, "y": -0.7071068, "z": 0.0, "w": 0.7071068}
        q_base = _quaternion_multiply(q_base, q_90yz)

        # --- 7. Local-frame offset shift via temporary TF frame -------------
        try:
            Q = [q_base["x"], q_base["y"], q_base["z"], q_base["w"]]
            R = tft.quaternion_matrix(Q)[0:3, 0:3]
            # NB: negative Z mirrors detect_objects (backward shift along local Z).
            shift_global = R.dot(np.array([x_offset, y_offset, -z_offset]))
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
                self.base_frame,
                "shifted_grasp",
                rospy.Time(0),
                rospy.Duration(self.tf_timeout_sec),
            )
            trans_s, rot_s = self._tf_listener.lookupTransform(
                self.base_frame, "shifted_grasp", rospy.Time(0)
            )
            t_base = [float(trans_s[0]), float(trans_s[1]), float(trans_s[2])]
            q_base = {"x": float(rot_s[0]), "y": float(rot_s[1]),
                      "z": float(rot_s[2]), "w": float(rot_s[3])}
        except Exception as e:
            rospy.logwarn(f"Failed to apply/lookup shifted grasp TF: {e}")
            # Keep pre-shift t_base/q_base if shifting fails.

        rospy.loginfo(
            f"Final pose in '{self.base_frame}' | "
            f"xyz=({t_base[0]:.4f}, {t_base[1]:.4f}, {t_base[2]:.4f})  "
            f"quat=({q_base['x']:.4f}, {q_base['y']:.4f}, "
            f"{q_base['z']:.4f}, {q_base['w']:.4f})"
        )

        # --- 8. Build response in detect_objects format ---------------------
        payload = {
            "status": "success",
            "best_grasp": {
                "translation_wrt_base": t_base,
                "quaternion_wrt_base":  q_base,
                "score": best["score"],
                "width": best["width"],
            },
            "camera": camera,
        }
        response.result_code.result_code = ResultCode.SUCCESS
        response.result_code.message     = "Grasp-from-mask succeeded."
        response.data                    = json.dumps(payload)
        return response

    # ------------------------------------------------------------------ #
    #  NPZ repacking                                                       #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _repack_npz_with_extras(npz_b64_in, cam_to_base_quat, depth_scale):
        """
        Decode the incoming NPZ (from get_object_mask), inject
        cam_to_base_quat + depth_scale, and re-encode as base64.

        AnyGrasp's payload-mode handler (_load_inputs_from_payload) reads
        these keys from inside the NPZ when the outer JSON contains
        'npz_b64', so we must add them there - not as top-level JSON keys.
        """
        npz_bytes = base64.b64decode(npz_b64_in)
        with np.load(io.BytesIO(npz_bytes), allow_pickle=False) as data:
            fields = {key: np.array(data[key]) for key in data.files}

        if cam_to_base_quat is not None:
            fields["cam_to_base_quat"] = np.asarray(
                cam_to_base_quat, dtype=np.float64
            )

        # 0-d float scalar; AnyGrasp casts it via float() so dtype doesn't matter.
        fields["depth_scale"] = np.asarray(float(depth_scale), dtype=np.float64)

        buf = io.BytesIO()
        np.savez(buf, **fields)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    # ------------------------------------------------------------------ #
    #  TF helpers                                                          #
    # ------------------------------------------------------------------ #
    def _get_cam_to_base_quat(self, camera_frame):
        """
        Returns [qx, qy, qz, qw] mapping vectors from camera_frame
        into the base frame, or None if the lookup fails.
        """
        try:
            self._tf_listener.waitForTransform(
                self.base_frame, camera_frame,
                rospy.Time(0), rospy.Duration(self.tf_timeout_sec),
            )
            _trans, rot = self._tf_listener.lookupTransform(
                self.base_frame, camera_frame, rospy.Time(0)
            )
            rospy.loginfo(
                f"cam_to_base_quat ({camera_frame} -> {self.base_frame}): "
                f"[{rot[0]:.4f}, {rot[1]:.4f}, {rot[2]:.4f}, {rot[3]:.4f}]"
            )
            return list(rot)  # [x, y, z, w]
        except Exception as e:
            rospy.logwarn(
                f"TF lookup for cam_to_base_quat failed - AnyGrasp horizontal "
                f"filter will fall back to camera_tilt_rad CLI flag. Error: {e}"
            )
            return None

    def _transform_pose_to_base(self, translation, quaternion, camera_frame):
        """Transform a pose in camera_frame into self.base_frame."""
        try:
            self._tf_listener.waitForTransform(
                self.base_frame, camera_frame,
                rospy.Time(0), rospy.Duration(self.tf_timeout_sec),
            )
            trans, rot = self._tf_listener.lookupTransform(
                self.base_frame, camera_frame, rospy.Time(0)
            )

            T_base_cam = tft.quaternion_matrix(rot)
            T_base_cam[0, 3] = trans[0]
            T_base_cam[1, 3] = trans[1]
            T_base_cam[2, 3] = trans[2]

            q_list = [quaternion["x"], quaternion["y"],
                      quaternion["z"], quaternion["w"]]
            T_pose_cam = tft.quaternion_matrix(q_list)
            T_pose_cam[0, 3] = translation[0]
            T_pose_cam[1, 3] = translation[1]
            T_pose_cam[2, 3] = translation[2]

            T_pose_base = np.dot(T_base_cam, T_pose_cam)
            t_base = [T_pose_base[0, 3], T_pose_base[1, 3], T_pose_base[2, 3]]
            q_out  = tft.quaternion_from_matrix(T_pose_base)
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

    # ------------------------------------------------------------------ #
    #  AnyGrasp WebSocket call (payload mode)                              #
    # ------------------------------------------------------------------ #
    def _call_anygrasp(self, npz_b64):
        """
        Send the NPZ to AnyGrasp's /get_grasp_config endpoint as
        {"npz_b64": "<base64>"}. AnyGrasp's _load_inputs_from_payload
        recognises the 'npz_b64' key and decodes the embedded fields.
        """
        try:
            result = asyncio.run(
                self._ws_send_recv(
                    self.anygrasp_url,
                    json.dumps({"npz_b64": npz_b64}),
                    self.grasp_timeout_sec,
                    max_msg_mb=50,
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
    #  Response helper                                                     #
    # ------------------------------------------------------------------ #
    def _fail(self, response, msg):
        rospy.logerr(f"get_grasp_config error: {msg}")
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
    rospy.init_node("get_grasp_config_service", anonymous=False)
    try:
        rospy.loginfo("Creating GraspFromMaskNode ...")
        node = GraspFromMaskNode()
        rospy.loginfo("GraspFromMaskNode spinning ...")
        node.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS interrupt - shutting down GraspFromMaskNode.")
    except Exception as e:
        rospy.logerr(f"Fatal error: {e}\n{traceback.format_exc()}")
    finally:
        rospy.loginfo("GraspFromMaskNode shutdown complete.")


if __name__ == "__main__":
    main()