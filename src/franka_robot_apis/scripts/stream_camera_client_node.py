#!/usr/bin/env python3
"""
agentlace_realsense_ros_client.py
ROS1 client — publishes calibrated intrinsics, aligned depth, and extrinsics TF.

Changes from original:
  - depth_aligned: published as a separate topic (aligned to color frame)
  - Calibrated intrinsics: handles both factory ("coeffs") and calibrated
    ("dist_coeffs") keys from the server
  - Extrinsics: broadcast as a TF static transform for the calibrated camera
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

# ── Camera serials ────────────────────────────────────────────────────────────
USE_D455 = False  # Set to True to use D455 as the calibrated camera (instead of D415)
CALIBRATED_SERIAL_D455 = "032522250211"   # d455
CALIBRATED_SERIAL_D415 = "947122060531"   # d415

CALIBRATED_SERIAL = CALIBRATED_SERIAL_D455 if USE_D455 else CALIBRATED_SERIAL_D415
SERIALS = ["123622270802", CALIBRATED_SERIAL]

# ── Camera role config — must match server ────────────────────────────────
CAMERA_ROLES = {
    "123622270802": "wrist",   # D405
    "947122060531": "scene",   # D415
    "032522250211": "scene",   # D455
}

observation_keys = []
for s in SERIALS:
    observation_keys += [
        f"cam_{s}_color",
        f"cam_{s}_depth",
        f"cam_{s}_depth_aligned",
        f"cam_{s}_color_info",
        f"cam_{s}_depth_info",
        f"cam_{s}_meta",
    ]
    if s == CALIBRATED_SERIAL:
        observation_keys.append(f"cam_{s}_extrinsics")

action_keys = ["command"]

QUEUE_SIZE_IMAGE = 1
QUEUE_SIZE_INFO  = 10
LATCH_INFO = True

# ── Reconnect settings ────────────────────────────────────────────────────────
MAX_CONSECUTIVE_NONE = 30
RECONNECT_AFTER_NONE = 100


# ── Helpers ───────────────────────────────────────────────────────────────────
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


def numpy_to_image_msg(arr: np.ndarray, encoding: str,
                        frame_id: str, stamp) -> Image:
    msg = Image()
    msg.header = Header(stamp=stamp, frame_id=frame_id)
    msg.height = arr.shape[0]
    msg.width  = arr.shape[1]
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
    coeffs = info_dict.get("dist_coeffs",
                           info_dict.get("coeffs",
                                         [0., 0., 0., 0., 0.]))
    ci.D = list(coeffs)

    model = info_dict.get("model", "")
    ci.distortion_model = rs_model_to_ros(model)
    return ci


def extrinsics_to_tf(ext_dict: dict, parent_frame: str, child_frame: str,
                     stamp) -> TransformStamped:
    """Convert the extrinsics dict from the server into a TF message."""
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

def make_publishers(serials):
    pubs = {}
    for s in serials:
        role = CAMERA_ROLES.get(s, s)   # "wrist" or "scene"
        base = f"/realsense/{role}"     # /realsense/wrist or /realsense/scene

        pubs[s] = {
            "color_img":  rospy.Publisher(
                f"{base}/color/image_raw", Image,
                queue_size=QUEUE_SIZE_IMAGE),
            "color_info": rospy.Publisher(
                f"{base}/color/camera_info", CameraInfo,
                queue_size=QUEUE_SIZE_INFO, latch=LATCH_INFO),
            "depth_img":  rospy.Publisher(
                f"{base}/depth/image_raw", Image,
                queue_size=QUEUE_SIZE_IMAGE),
            "depth_info": rospy.Publisher(
                f"{base}/depth/camera_info", CameraInfo,
                queue_size=QUEUE_SIZE_INFO, latch=LATCH_INFO),
            # Aligned depth: same resolution + frame as color
            "depth_aligned_img": rospy.Publisher(
                f"{base}/aligned_depth_to_color/image_raw", Image,
                queue_size=QUEUE_SIZE_IMAGE),
            "depth_aligned_info": rospy.Publisher(
                f"{base}/aligned_depth_to_color/camera_info", CameraInfo,
                queue_size=QUEUE_SIZE_INFO, latch=LATCH_INFO),
        }
        rospy.loginfo(
            f"Advertising {base}/{{color,depth,aligned_depth_to_color}}/...")
    return pubs


def make_client(ip, port):
    config = ActionConfig(
        port_number=port,
        action_keys=action_keys,
        observation_keys=observation_keys,
    )
    return ActionClient(ip, config=config)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip",   default="localhost")
    parser.add_argument("--port", type=int, default=6379)
    parser.add_argument("--base-frame", default="panda_link0",
                        help="TF parent frame for extrinsics")
    role_var = CAMERA_ROLES.get(CALIBRATED_SERIAL, "scene")
    parser.add_argument("--camera-frame", default=f"cam_{role_var}_depth_optical_frame",
                        help="TF child frame for extrinsics")
    args, _ = parser.parse_known_args()

    rospy.init_node("agentlace_realsense_client", anonymous=False)
    rospy.loginfo(f"Connecting to agentlace server at {args.ip}:{args.port}")

    client = make_client(args.ip, args.port)
    pubs   = make_publishers(SERIALS)

    # TF broadcaster for extrinsics
    tf_broadcaster = tf2_ros.StaticTransformBroadcaster()
    extrinsics_published = False

    # Cache intrinsics — persist across frames
    info_cache = {s: {"color": None, "depth": None} for s in SERIALS}

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
                    client = make_client(args.ip, args.port)
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
            ext = obs.get(f"cam_{CALIBRATED_SERIAL}_extrinsics")
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
        for serial in SERIALS:
            p = pubs[serial]

            # Update intrinsics cache
            c_info = obs.get(f"cam_{serial}_color_info")
            d_info = obs.get(f"cam_{serial}_depth_info")
            if c_info:
                info_cache[serial]["color"] = c_info
            if d_info:
                info_cache[serial]["depth"] = d_info

            color_fid = f"cam_{serial}_color_optical_frame"
            depth_fid = f"cam_{serial}_depth_optical_frame"

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

            meta = obs.get(f"cam_{serial}_meta")
            if meta:
                rospy.set_param(f"/realsense/{meta['role']}/depth_min_mm", meta["depth_min_mm"])
                rospy.set_param(f"/realsense/{meta['role']}/depth_max_mm", meta["depth_max_mm"])
                # In the meta block in the client, add:
                if info_cache[serial]["depth"] and "depth_scale" in info_cache[serial]["depth"]:
                    rospy.set_param(
                        f"/realsense/{meta['role']}/depth_scale",
                        info_cache[serial]["depth"]["depth_scale"]
                    )

            # only do if camera type is wrist
            if CAMERA_ROLES.get(serial) == "wrist":
                # ── Depth range warning — scale raw units → mm using depth_scale ──
                if depth_aligned is not None and isinstance(depth_aligned, np.ndarray) and meta:
                    depth_scale = 1.0  # default: assume 1 unit = 1mm
                    if info_cache[serial]["depth"] and "depth_scale" in info_cache[serial]["depth"]:
                        # depth_scale is in metres-per-unit, convert to mm-per-unit
                        depth_scale = info_cache[serial]["depth"]["depth_scale"] * 1000.0

                    valid = depth_aligned[depth_aligned > 0]
                    if valid.size > 0:
                        median_mm = float(np.median(valid)) * depth_scale  # ← correct unit
                        if not (meta["depth_min_mm"] < median_mm < meta["depth_max_mm"]):
                            rospy.logwarn_throttle(
                                5.0,
                                f"[{serial}] Median depth {median_mm:.0f}mm is outside "
                                f"valid range [{meta['depth_min_mm']}, {meta['depth_max_mm']}]mm "
                                f"— is the camera aimed correctly?"
                            )
                        else:
                            rospy.logdebug(
                                f"[{serial}] Median depth {median_mm:.0f}mm ✓"
                            )


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass