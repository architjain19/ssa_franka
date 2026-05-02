#!/usr/bin/env python3
"""
agentlace_realsense_ros_client.py
ROS1 client — publishes calibrated intrinsics, aligned depth, extrinsics TF,
and IR stereo streams.

IMPORTANT — agentlace config-hash matching
------------------------------------------
agentlace hashes the ActionConfig (observation_keys list) on both sides and
raises "Incompatible config with hash" if they differ.  The server only adds
keys for cameras that are physically connected at startup, so the client must
pass exactly the same serial list.

Common invocations
------------------
  # Single scene camera (default — matches server with one D415 attached)
  rosrun franka_robot_apis stream_camera_client_node.py

  # Wrist + scene cameras
  rosrun franka_robot_apis stream_camera_client_node.py \
      --serials 123622270802 947122060531

  # D455 as the calibrated scene camera
  rosrun franka_robot_apis stream_camera_client_node.py \
      --calibrated-serial 032522250211

  # Server started with --no-ir
  rosrun franka_robot_apis stream_camera_client_node.py --no-ir
"""

import argparse
import time
import numpy as np

import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped
import tf2_ros

from agentlace.action import ActionClient, ActionConfig

# ── Known serial → role mapping ────────────────────────────────────────────────
CAMERA_ROLES = {
    "123622270802": "wrist",   # D405
    "947122060531": "scene",   # D415
    "032522250211": "scene",   # D455
}

# Defaults — overridden by CLI args at runtime.
_DEFAULT_CALIBRATED_SERIAL = "947122060531"          # D415
_DEFAULT_SERIALS            = [_DEFAULT_CALIBRATED_SERIAL]

action_keys = ["command"]

QUEUE_SIZE_IMAGE = 1
QUEUE_SIZE_INFO  = 10
LATCH_INFO       = True

# ── Reconnect settings ─────────────────────────────────────────────────────────
MAX_CONSECUTIVE_NONE = 30
RECONNECT_AFTER_NONE = 100


# ── Helpers ────────────────────────────────────────────────────────────────────
RS_MODEL_TO_ROS = {
    "distortion.brown_conrady":         "plumb_bob",
    "distortion.inverse_brown_conrady": "plumb_bob",
    "distortion.kannala_brandt4":       "fisheye",
    "distortion.fov":                   "plumb_bob",
    "distortion.none":                  "plumb_bob",
    # Calibrated intrinsics use these model names directly
    "plumb_bob":                        "plumb_bob",
    "rational_polynomial":              "rational_polynomial",
}


def rs_model_to_ros(model_str: str) -> str:
    return RS_MODEL_TO_ROS.get(model_str, "plumb_bob")


def serial_has_ir(serial: str, enable_ir: bool) -> bool:
    """True when the server is expected to publish IR for this serial.
    Mirrors server logic: IR is always skipped for 'wrist' cameras."""
    return enable_ir and CAMERA_ROLES.get(serial, "scene") != "wrist"


def build_observation_keys(serials, calibrated_serial, enable_ir: bool):
    """
    Build the observation key list in the same order and with the same
    conditional IR / extrinsics keys as the server's obs_keys loop, so
    the agentlace config hash matches exactly.
    """
    keys = []
    for s in serials:
        keys += [
            f"cam_{s}_color",
            f"cam_{s}_depth",
            f"cam_{s}_depth_aligned",
            f"cam_{s}_color_info",
            f"cam_{s}_depth_info",
            f"cam_{s}_meta",
        ]
        if serial_has_ir(s, enable_ir):
            keys += [
                f"cam_{s}_ir_left",
                f"cam_{s}_ir_right",
                f"cam_{s}_ir_info",
            ]
        if s == calibrated_serial:
            keys.append(f"cam_{s}_extrinsics")
    return keys


def numpy_to_image_msg(arr: np.ndarray, encoding: str,
                       frame_id: str, stamp) -> Image:
    msg = Image()
    msg.header       = Header(stamp=stamp, frame_id=frame_id)
    msg.height       = arr.shape[0]
    msg.width        = arr.shape[1]
    msg.encoding     = encoding
    msg.is_bigendian = False

    if encoding == "bgr8":
        assert arr.ndim == 3 and arr.shape[2] == 3, \
            f"Expected HxWx3 for bgr8, got {arr.shape}"
        msg.step = arr.shape[1] * 3
        msg.data = np.ascontiguousarray(arr, dtype=np.uint8).tobytes()
    elif encoding == "16UC1":
        assert arr.ndim == 2, \
            f"Expected HxW for 16UC1, got {arr.shape}"
        msg.step = arr.shape[1] * 2
        msg.data = np.ascontiguousarray(arr, dtype=np.uint16).tobytes()
    elif encoding == "mono8":
        assert arr.ndim == 2, \
            f"Expected HxW for mono8, got {arr.shape}"
        msg.step = arr.shape[1]
        msg.data = np.ascontiguousarray(arr, dtype=np.uint8).tobytes()
    else:
        raise ValueError(f"Unsupported encoding: {encoding}")

    return msg


def build_camera_info(info_dict: dict, frame_id: str, stamp) -> CameraInfo:
    ci = CameraInfo()
    ci.header = Header(stamp=stamp, frame_id=frame_id)
    ci.width  = info_dict["width"]
    ci.height = info_dict["height"]
    fx = info_dict["fx"];  fy = info_dict["fy"]
    cx = info_dict["cx"];  cy = info_dict["cy"]
    ci.K = [fx,  0., cx,
             0., fy, cy,
             0.,  0., 1.]
    ci.R = [1., 0., 0.,
            0., 1., 0.,
            0., 0., 1.]
    ci.P = [fx,  0., cx, 0.,
             0., fy, cy, 0.,
             0.,  0., 1., 0.]
    # Handle both factory format ("coeffs") and calibrated format ("dist_coeffs")
    ci.D = list(info_dict.get("dist_coeffs",
                              info_dict.get("coeffs",
                                            [0., 0., 0., 0., 0.])))
    ci.distortion_model = rs_model_to_ros(info_dict.get("model", ""))
    return ci


def build_ir_right_camera_info(ir_info: dict, frame_id: str, stamp) -> CameraInfo:
    """
    CameraInfo for IR2 (right) following the ROS stereo_image_proc convention:
      P[0,3] = -fx * baseline_m  (negative Tx)
    All other fields are identical to the left camera.
    """
    ci = build_camera_info(ir_info, frame_id, stamp)
    baseline_m = ir_info.get("baseline_m", 0.0)
    p = list(ci.P)
    p[3] = -ir_info["fx"] * baseline_m
    ci.P = p
    return ci


def extrinsics_to_tf(ext_dict: dict, parent_frame: str, child_frame: str,
                     stamp) -> TransformStamped:
    t = TransformStamped()
    t.header.stamp    = stamp
    t.header.frame_id = parent_frame
    t.child_frame_id  = child_frame
    pos = ext_dict["position_xyz"]
    t.transform.translation.x = pos[0]
    t.transform.translation.y = pos[1]
    t.transform.translation.z = pos[2]
    q = ext_dict["quat_xyzw"]
    t.transform.rotation.x = q[0]
    t.transform.rotation.y = q[1]
    t.transform.rotation.z = q[2]
    t.transform.rotation.w = q[3]
    return t


def make_publishers(serials, enable_ir: bool):
    pubs = {}
    for s in serials:
        role = CAMERA_ROLES.get(s, s)
        base = f"/realsense/{role}"

        pubs[s] = {
            # ── RGB ───────────────────────────────────────────────────────────
            "color_img":  rospy.Publisher(
                f"{base}/color/image_raw", Image,
                queue_size=QUEUE_SIZE_IMAGE),
            "color_info": rospy.Publisher(
                f"{base}/color/camera_info", CameraInfo,
                queue_size=QUEUE_SIZE_INFO, latch=LATCH_INFO),
            # ── Raw depth (native depth-sensor resolution & frame) ─────────────
            "depth_img":  rospy.Publisher(
                f"{base}/depth/image_raw", Image,
                queue_size=QUEUE_SIZE_IMAGE),
            "depth_info": rospy.Publisher(
                f"{base}/depth/camera_info", CameraInfo,
                queue_size=QUEUE_SIZE_INFO, latch=LATCH_INFO),
            # ── Aligned depth (color resolution & frame) ──────────────────────
            "depth_aligned_img": rospy.Publisher(
                f"{base}/aligned_depth_to_color/image_raw", Image,
                queue_size=QUEUE_SIZE_IMAGE),
            "depth_aligned_info": rospy.Publisher(
                f"{base}/aligned_depth_to_color/camera_info", CameraInfo,
                queue_size=QUEUE_SIZE_INFO, latch=LATCH_INFO),
        }

        # ── IR stereo (scene cameras only) ────────────────────────────────────
        if serial_has_ir(s, enable_ir):
            pubs[s]["ir_left_img"]   = rospy.Publisher(
                f"{base}/infra1/image_rect_raw", Image,
                queue_size=QUEUE_SIZE_IMAGE)
            pubs[s]["ir_left_info"]  = rospy.Publisher(
                f"{base}/infra1/camera_info", CameraInfo,
                queue_size=QUEUE_SIZE_INFO, latch=LATCH_INFO)
            pubs[s]["ir_right_img"]  = rospy.Publisher(
                f"{base}/infra2/image_rect_raw", Image,
                queue_size=QUEUE_SIZE_IMAGE)
            pubs[s]["ir_right_info"] = rospy.Publisher(
                f"{base}/infra2/camera_info", CameraInfo,
                queue_size=QUEUE_SIZE_INFO, latch=LATCH_INFO)
            ir_note = " +infra1/infra2"
        else:
            ir_note = ""

        rospy.loginfo(
            f"Advertising {base}/{{color,depth,aligned_depth_to_color}}"
            f"{ir_note}/...")

    return pubs


def make_client(ip, port, obs_keys):
    config = ActionConfig(
        port_number=port,
        action_keys=action_keys,
        observation_keys=obs_keys,
    )
    return ActionClient(ip, config=config)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="agentlace RealSense → ROS1 bridge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--ip",   default="localhost")
    parser.add_argument("--port", type=int, default=6379)
    parser.add_argument(
        "--serials", nargs="+",
        default=_DEFAULT_SERIALS,
        metavar="SERIAL",
        help=(
            "Space-separated serials of cameras to subscribe to. "
            "Must exactly match what the server has connected. "
            f"Default: {_DEFAULT_SERIALS}"
        ),
    )
    parser.add_argument(
        "--calibrated-serial",
        default=_DEFAULT_CALIBRATED_SERIAL,
        metavar="SERIAL",
        help=(
            "Serial of the camera whose ChArUco extrinsics TF to broadcast. "
            f"Default: {_DEFAULT_CALIBRATED_SERIAL} (D415)"
        ),
    )
    parser.add_argument(
        "--enable-ir", dest="enable_ir",
        action="store_true", default=True,
        help="Subscribe to IR streams (default: on). Must match server.",
    )
    parser.add_argument(
        "--no-ir", dest="enable_ir",
        action="store_false",
        help="Disable IR streams. Use when server was started with --no-ir.",
    )
    parser.add_argument(
        "--base-frame", default="panda_link0",
        help="TF parent frame for camera extrinsics (default: panda_link0)",
    )
    parser.add_argument(
        "--camera-frame", default=None,
        help=(
            "TF child frame for camera extrinsics. "
            "Defaults to cam_<role>_depth_optical_frame "
            "derived from --calibrated-serial."
        ),
    )
    args, _ = parser.parse_known_args()

    # Derive camera-frame from calibrated serial when not explicitly given
    if args.camera_frame is None:
        role = CAMERA_ROLES.get(args.calibrated_serial, "scene")
        args.camera_frame = f"cam_{role}_depth_optical_frame"

    # Build observation_keys AFTER argparse — this is what gets hashed and
    # compared against the server; it must match the server's obs_keys exactly.
    obs_keys = build_observation_keys(
        args.serials, args.calibrated_serial, args.enable_ir)

    rospy.init_node("agentlace_realsense_client", anonymous=False)
    rospy.loginfo(f"Connecting to agentlace server at {args.ip}:{args.port}")
    rospy.loginfo(f"Serials         : {args.serials}")
    rospy.loginfo(f"Calibrated cam  : {args.calibrated_serial}")
    rospy.loginfo(f"IR enabled      : {args.enable_ir}")
    rospy.loginfo(f"Obs keys ({len(obs_keys)}): {obs_keys}")

    client = make_client(args.ip, args.port, obs_keys)
    pubs   = make_publishers(args.serials, args.enable_ir)

    # TF broadcaster for extrinsics (static — publish once)
    tf_broadcaster       = tf2_ros.StaticTransformBroadcaster()
    extrinsics_published = False

    # Cache intrinsics per serial — persist across frames
    info_cache = {s: {"color": None, "depth": None, "ir": None}
                  for s in args.serials}

    # Diagnostics
    none_count  = 0
    frame_count = 0
    t_last_log  = time.time()
    LOG_INTERVAL = 10.0

    while not rospy.is_shutdown():
        obs = client.obs()

        # ── Handle None ───────────────────────────────────────────────────────
        if obs is None:
            none_count += 1
            if none_count == MAX_CONSECUTIVE_NONE:
                rospy.logwarn(
                    f"Received {none_count} consecutive None observations — "
                    f"is the server running at {args.ip}:{args.port}?"
                )
            if none_count >= RECONNECT_AFTER_NONE:
                rospy.logwarn("Attempting to reconnect to agentlace server...")
                try:
                    client = make_client(args.ip, args.port, obs_keys)
                    none_count = 0
                    rospy.loginfo("Reconnected.")
                except Exception as e:
                    rospy.logerr(f"Reconnect failed: {e}")
                    time.sleep(1.0)
            continue

        none_count = 0
        frame_count += 1
        stamp = rospy.Time.now()

        # ── Periodic FPS log ──────────────────────────────────────────────────
        now = time.time()
        if now - t_last_log >= LOG_INTERVAL:
            fps = frame_count / (now - t_last_log)
            rospy.logdebug(f"Publishing at ~{fps:.1f} fps")
            frame_count = 0
            t_last_log  = now

        # ── Extrinsics → TF (publish once, it's static) ──────────────────────
        if not extrinsics_published:
            ext = obs.get(f"cam_{args.calibrated_serial}_extrinsics")
            if ext is not None and isinstance(ext, dict):
                tf_msg = extrinsics_to_tf(
                    ext, args.base_frame, args.camera_frame, stamp)
                tf_broadcaster.sendTransform(tf_msg)
                pos = ext["position_xyz"]
                rospy.loginfo(
                    f"Published static TF: {args.base_frame} → "
                    f"{args.camera_frame}  "
                    f"pos=[{pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f}]")
                extrinsics_published = True

        # ── Per-camera publish ────────────────────────────────────────────────
        for serial in args.serials:
            p = pubs[serial]

            # Update intrinsics cache
            c_info  = obs.get(f"cam_{serial}_color_info")
            d_info  = obs.get(f"cam_{serial}_depth_info")
            ir_info = obs.get(f"cam_{serial}_ir_info")
            if c_info:
                info_cache[serial]["color"] = c_info
            if d_info:
                info_cache[serial]["depth"] = d_info
            if ir_info:
                info_cache[serial]["ir"] = ir_info

            color_fid = f"cam_{serial}_color_optical_frame"
            depth_fid = f"cam_{serial}_depth_optical_frame"
            ir1_fid   = f"cam_{serial}_infra1_optical_frame"
            ir2_fid   = f"cam_{serial}_infra2_optical_frame"

            # ── Color ─────────────────────────────────────────────────────────
            color = obs.get(f"cam_{serial}_color")
            if color is not None and isinstance(color, np.ndarray):
                try:
                    p["color_img"].publish(
                        numpy_to_image_msg(color, "bgr8", color_fid, stamp))
                    if info_cache[serial]["color"]:
                        p["color_info"].publish(
                            build_camera_info(
                                info_cache[serial]["color"], color_fid, stamp))
                except Exception as e:
                    rospy.logerr(f"[{serial}] color publish error: {e}")

            # ── Raw depth (native resolution, depth sensor frame) ─────────────
            depth = obs.get(f"cam_{serial}_depth")
            if depth is not None and isinstance(depth, np.ndarray):
                try:
                    p["depth_img"].publish(
                        numpy_to_image_msg(depth, "16UC1", depth_fid, stamp))
                    if info_cache[serial]["depth"]:
                        p["depth_info"].publish(
                            build_camera_info(
                                info_cache[serial]["depth"], depth_fid, stamp))
                except Exception as e:
                    rospy.logerr(f"[{serial}] depth publish error: {e}")

            # ── Aligned depth (color resolution, color frame) ─────────────────
            depth_aligned = obs.get(f"cam_{serial}_depth_aligned")
            if depth_aligned is not None and isinstance(depth_aligned, np.ndarray):
                try:
                    # Aligned depth lives in the COLOR frame — use color_fid
                    # and color intrinsics, since each pixel corresponds to the
                    # same (u,v) in the color image.
                    p["depth_aligned_img"].publish(
                        numpy_to_image_msg(
                            depth_aligned, "16UC1", color_fid, stamp))
                    if info_cache[serial]["color"]:
                        p["depth_aligned_info"].publish(
                            build_camera_info(
                                info_cache[serial]["color"], color_fid, stamp))
                except Exception as e:
                    rospy.logerr(f"[{serial}] depth_aligned publish error: {e}")

            # ── IR left / right (scene cameras only) ──────────────────────────
            if serial_has_ir(serial, args.enable_ir):
                cached_ir = info_cache[serial]["ir"]

                ir_left = obs.get(f"cam_{serial}_ir_left")
                if ir_left is not None and isinstance(ir_left, np.ndarray):
                    try:
                        p["ir_left_img"].publish(
                            numpy_to_image_msg(ir_left, "mono8", ir1_fid, stamp))
                        if cached_ir:
                            # Left camera: standard pinhole P (Tx = 0)
                            p["ir_left_info"].publish(
                                build_camera_info(cached_ir, ir1_fid, stamp))
                    except Exception as e:
                        rospy.logerr(f"[{serial}] IR left publish error: {e}")

                ir_right = obs.get(f"cam_{serial}_ir_right")
                if ir_right is not None and isinstance(ir_right, np.ndarray):
                    try:
                        p["ir_right_img"].publish(
                            numpy_to_image_msg(ir_right, "mono8", ir2_fid, stamp))
                        if cached_ir:
                            # Right camera: P[0,3] = -fx * baseline_m
                            # (ROS stereo_image_proc convention)
                            p["ir_right_info"].publish(
                                build_ir_right_camera_info(
                                    cached_ir, ir2_fid, stamp))
                    except Exception as e:
                        rospy.logerr(f"[{serial}] IR right publish error: {e}")

            # ── Meta → ROS params ─────────────────────────────────────────────
            meta = obs.get(f"cam_{serial}_meta")
            if meta:
                role = meta["role"]
                rospy.set_param(f"/realsense/{role}/depth_min_mm",
                                meta["depth_min_mm"])
                rospy.set_param(f"/realsense/{role}/depth_max_mm",
                                meta["depth_max_mm"])
                # Forward ir_available so downstream nodes (e.g. a
                # FoundationStereo launcher) can check before subscribing.
                rospy.set_param(f"/realsense/{role}/ir_available",
                                meta.get("ir_available", False))
                if (info_cache[serial]["depth"] and
                        "depth_scale" in info_cache[serial]["depth"]):
                    rospy.set_param(
                        f"/realsense/{role}/depth_scale",
                        info_cache[serial]["depth"]["depth_scale"])

            # ── Wrist: depth range warning ────────────────────────────────────
            if CAMERA_ROLES.get(serial) == "wrist":
                if (depth_aligned is not None and
                        isinstance(depth_aligned, np.ndarray) and meta):
                    depth_scale = 1.0
                    if (info_cache[serial]["depth"] and
                            "depth_scale" in info_cache[serial]["depth"]):
                        # depth_scale is metres-per-unit; convert to mm-per-unit
                        depth_scale = (
                            info_cache[serial]["depth"]["depth_scale"] * 1000.0)

                    valid = depth_aligned[depth_aligned > 0]
                    if valid.size > 0:
                        median_mm = float(np.median(valid)) * depth_scale
                        if not (meta["depth_min_mm"] < median_mm <
                                meta["depth_max_mm"]):
                            rospy.logwarn_throttle(
                                5.0,
                                f"[{serial}] Median depth {median_mm:.0f}mm is "
                                f"outside valid range "
                                f"[{meta['depth_min_mm']}, "
                                f"{meta['depth_max_mm']}]mm "
                                f"— is the camera aimed correctly?"
                            )
                        else:
                            rospy.logdebug(
                                f"[{serial}] Median depth {median_mm:.0f}mm ✓")


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass