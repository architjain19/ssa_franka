#!/usr/bin/env python3
"""
agentlace_realsense_ros_client.py
ROS1 client — publishes calibrated intrinsics, aligned depth, and a static TF
per camera (scene D415 + wrist D405) relative to panda_link0.
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
SCENE_SERIAL = "947122060531"   # D415 — scene
WRIST_SERIAL = "123622270802"   # D405 — wrist
# SERIALS = [WRIST_SERIAL, SCENE_SERIAL]
SERIALS = [SCENE_SERIAL]

CAMERA_ROLES = {
    WRIST_SERIAL: "wrist",
    SCENE_SERIAL: "scene",
}

# Subscribe to extrinsics for BOTH cameras now
observation_keys = []
for s in SERIALS:
    observation_keys += [
        f"cam_{s}_color",
        f"cam_{s}_depth",
        f"cam_{s}_depth_aligned",
        f"cam_{s}_color_info",
        f"cam_{s}_depth_info",
        f"cam_{s}_meta",
        f"cam_{s}_extrinsics",   # both cameras
    ]

action_keys = ["command"]

QUEUE_SIZE_IMAGE = 1
QUEUE_SIZE_INFO  = 10
LATCH_INFO = True

MAX_CONSECUTIVE_NONE = 30
RECONNECT_AFTER_NONE = 100


# ── Helpers ───────────────────────────────────────────────────────────────────
RS_MODEL_TO_ROS = {
    "distortion.brown_conrady":         "plumb_bob",
    "distortion.inverse_brown_conrady": "plumb_bob",
    "distortion.kannala_brandt4":       "fisheye",
    "distortion.fov":                   "plumb_bob",
    "distortion.none":                  "plumb_bob",
    "plumb_bob":                        "plumb_bob",
    "rational_polynomial":              "rational_polynomial",
}


def rs_model_to_ros(model_str: str) -> str:
    return RS_MODEL_TO_ROS.get(model_str, "plumb_bob")


def numpy_to_image_msg(arr, encoding, frame_id, stamp):
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


def build_camera_info(info, frame_id, stamp):
    ci = CameraInfo()
    ci.header = Header(stamp=stamp, frame_id=frame_id)
    ci.width  = info["width"]
    ci.height = info["height"]
    fx = info["fx"]; fy = info["fy"]
    cx = info["cx"]; cy = info["cy"]
    ci.K = [fx, 0., cx,  0., fy, cy,  0., 0., 1.]
    ci.R = [1., 0., 0.,  0., 1., 0.,  0., 0., 1.]
    ci.P = [fx, 0., cx, 0.,  0., fy, cy, 0.,  0., 0., 1., 0.]
    coeffs = info.get("dist_coeffs",
                      info.get("coeffs", [0., 0., 0., 0., 0.]))
    ci.D = list(coeffs)
    ci.distortion_model = rs_model_to_ros(info.get("model", ""))
    return ci


def extrinsics_to_tf(ext, parent_frame, child_frame, stamp):
    t = TransformStamped()
    t.header.stamp    = stamp
    t.header.frame_id = parent_frame
    t.child_frame_id  = child_frame
    pos = ext["position_xyz"]
    t.transform.translation.x = pos[0]
    t.transform.translation.y = pos[1]
    t.transform.translation.z = pos[2]
    q = ext["quat_xyzw"]
    t.transform.rotation.x = q[0]
    t.transform.rotation.y = q[1]
    t.transform.rotation.z = q[2]
    t.transform.rotation.w = q[3]
    return t


def make_publishers(serials):
    pubs = {}
    for s in serials:
        role = CAMERA_ROLES.get(s, s)
        base = f"/realsense/{role}"
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
                        help="TF parent frame for both camera extrinsics")
    parser.add_argument("--scene-frame", default="cam_scene_color_optical_frame",
                        help="TF child frame for scene (D415) extrinsics")
    parser.add_argument("--wrist-frame", default="cam_wrist_color_optical_frame",
                        help="TF child frame for wrist (D405) extrinsics")
    args, _ = parser.parse_known_args()

    rospy.init_node("agentlace_realsense_client", anonymous=False)
    rospy.loginfo(f"Connecting to agentlace server at {args.ip}:{args.port}")

    client = make_client(args.ip, args.port)
    pubs   = make_publishers(SERIALS)

    # Per-camera TF child frame
    child_frames = {
        SCENE_SERIAL: args.scene_frame,
        WRIST_SERIAL: args.wrist_frame,
    }

    # One static broadcaster for everything. StaticTransformBroadcaster
    # internally accumulates transforms by child_frame_id, so calling
    # sendTransform once per camera as their extrinsics arrive is fine —
    # all known TFs stay live.
    tf_broadcaster = tf2_ros.StaticTransformBroadcaster()
    extrinsics_published = {s: False for s in SERIALS}

    info_cache = {s: {"color": None, "depth": None} for s in SERIALS}

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

        now = time.time()
        if now - t_last_log >= LOG_INTERVAL:
            fps = frame_count / (now - t_last_log)
            rospy.logdebug(f"Publishing at ~{fps:.1f} fps")
            frame_count = 0
            t_last_log  = now

        # ── Extrinsics → static TF, once per camera ───────────────────────────
        for serial in SERIALS:
            if extrinsics_published[serial]:
                continue
            ext = obs.get(f"cam_{serial}_extrinsics")
            if ext is not None and isinstance(ext, dict):
                child = child_frames[serial]
                tf_msg = extrinsics_to_tf(ext, args.base_frame, child, stamp)
                tf_broadcaster.sendTransform(tf_msg)
                pos = ext["position_xyz"]
                rospy.loginfo(
                    f"Published static TF: {args.base_frame} → {child}  "
                    f"pos=[{pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f}]")
                extrinsics_published[serial] = True

        # ── Per-camera image publish ──────────────────────────────────────────
        for serial in SERIALS:
            p = pubs[serial]
            role = CAMERA_ROLES[serial]

            c_info = obs.get(f"cam_{serial}_color_info")
            d_info = obs.get(f"cam_{serial}_depth_info")
            if c_info: info_cache[serial]["color"] = c_info
            if d_info: info_cache[serial]["depth"] = d_info

            color_fid = f"cam_{role}_color_optical_frame"
            depth_fid = f"cam_{role}_depth_optical_frame"

            # Color
            color = obs.get(f"cam_{serial}_color")
            if color is not None and isinstance(color, np.ndarray):
                try:
                    p["color_img"].publish(
                        numpy_to_image_msg(color, "bgr8", color_fid, stamp))
                    if info_cache[serial]["color"]:
                        p["color_info"].publish(build_camera_info(
                            info_cache[serial]["color"], color_fid, stamp))
                except Exception as e:
                    rospy.logerr(f"[{serial}] color publish error: {e}")

            # Raw depth (depth sensor frame)
            depth = obs.get(f"cam_{serial}_depth")
            if depth is not None and isinstance(depth, np.ndarray):
                try:
                    p["depth_img"].publish(
                        numpy_to_image_msg(depth, "16UC1", depth_fid, stamp))
                    if info_cache[serial]["depth"]:
                        p["depth_info"].publish(build_camera_info(
                            info_cache[serial]["depth"], depth_fid, stamp))
                except Exception as e:
                    rospy.logerr(f"[{serial}] depth publish error: {e}")

            # Aligned depth (color frame, color intrinsics)
            depth_aligned = obs.get(f"cam_{serial}_depth_aligned")
            if depth_aligned is not None and isinstance(depth_aligned, np.ndarray):
                try:
                    p["depth_aligned_img"].publish(numpy_to_image_msg(
                        depth_aligned, "16UC1", color_fid, stamp))
                    if info_cache[serial]["color"]:
                        p["depth_aligned_info"].publish(build_camera_info(
                            info_cache[serial]["color"], color_fid, stamp))
                except Exception as e:
                    rospy.logerr(f"[{serial}] depth_aligned publish error: {e}")

            # Meta → ROS params
            meta = obs.get(f"cam_{serial}_meta")
            if meta:
                rospy.set_param(f"/realsense/{meta['role']}/depth_min_mm",
                                meta["depth_min_mm"])
                rospy.set_param(f"/realsense/{meta['role']}/depth_max_mm",
                                meta["depth_max_mm"])
                if info_cache[serial]["depth"] and \
                        "depth_scale" in info_cache[serial]["depth"]:
                    rospy.set_param(
                        f"/realsense/{meta['role']}/depth_scale",
                        info_cache[serial]["depth"]["depth_scale"])

            # Wrist depth-range sanity warning
            if role == "wrist" and depth_aligned is not None and \
                    isinstance(depth_aligned, np.ndarray) and meta:
                depth_scale = 1.0
                if info_cache[serial]["depth"] and \
                        "depth_scale" in info_cache[serial]["depth"]:
                    depth_scale = info_cache[serial]["depth"]["depth_scale"] * 1000.0
                valid = depth_aligned[depth_aligned > 0]
                if valid.size > 0:
                    median_mm = float(np.median(valid)) * depth_scale
                    if not (meta["depth_min_mm"] < median_mm < meta["depth_max_mm"]):
                        rospy.logwarn_throttle(5.0,
                            f"[{serial}] Median depth {median_mm:.0f}mm outside "
                            f"[{meta['depth_min_mm']}, {meta['depth_max_mm']}]mm "
                            f"— is the camera aimed correctly?")
                    else:
                        rospy.logdebug(
                            f"[{serial}] Median depth {median_mm:.0f}mm ✓")


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass