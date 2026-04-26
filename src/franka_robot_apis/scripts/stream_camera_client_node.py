#!/usr/bin/env python3
"""
agentlace_realsense_ros_client.py
ROS1 client - no cv_bridge, robust publisher with proper ROS1 patterns.
"""

import argparse
import time
import numpy as np

import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import Image, CameraInfo

from agentlace.action import ActionClient, ActionConfig

# ── Camera serials ────────────────────────────────────────────────────────────
SERIALS = ["123622270802", "947122060531", "032522250211"]

observation_keys = []
for s in SERIALS:
    observation_keys += [
        f"cam_{s}_color",
        f"cam_{s}_depth",
        f"cam_{s}_color_info",
        f"cam_{s}_depth_info",
    ]
action_keys = ["command"]

QUEUE_SIZE_IMAGE = 1       # latest frame only - drop stale images
QUEUE_SIZE_INFO  = 10      # small msg, keep a buffer so it's never lost

# latch=True on camera_info means late-joining subscribers get the last
# published info immediately - important for tools like image_proc.
LATCH_INFO = True

# ── Reconnect settings ────────────────────────────────────────────────────────
MAX_CONSECUTIVE_NONE = 30   # warn after this many consecutive None obs
RECONNECT_AFTER_NONE = 100  # attempt reconnect after this many


# ── Helpers ───────────────────────────────────────────────────────────────────
RS_MODEL_TO_ROS = {
    "distortion.brown_conrady":         "plumb_bob",
    "distortion.inverse_brown_conrady": "plumb_bob",
    "distortion.kannala_brandt4":       "fisheye",
    "distortion.fov":                   "plumb_bob",
    "distortion.none":                  "plumb_bob",
}

def rs_model_to_ros(model_str: str) -> str:
    return RS_MODEL_TO_ROS.get(model_str, "plumb_bob")

def numpy_to_image_msg(arr: np.ndarray, encoding: str,
                        frame_id: str, stamp) -> Image:
    """
    Equivalent to cv_bridge for bgr8 and 16UC1.
    bgr8  : uint8  H x W x 3,  step = W*3
    16UC1 : uint16 H x W,      step = W*2
    """
    msg = Image()
    msg.header = Header(stamp=stamp, frame_id=frame_id)
    msg.height = arr.shape[0]
    msg.width  = arr.shape[1]
    msg.encoding     = encoding
    msg.is_bigendian = False   # x86/ARM are little-endian

    if encoding == "bgr8":
        assert arr.ndim == 3 and arr.shape[2] == 3, \
            f"Expected HxWx3 for bgr8, got {arr.shape}"
        msg.step = arr.shape[1] * 3
        # Ensure contiguous memory before tobytes (agentlace may return non-contiguous views)
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
    ci.R = [1., 0., 0.,   # identity - no rectification for raw streams
            0., 1., 0.,
            0., 0., 1.]
    ci.P = [fx,  0., cx, 0.,
             0., fy, cy, 0.,
             0.,  0., 1., 0.]
    ci.D = list(info_dict.get("coeffs", [0., 0., 0., 0., 0.]))
    ci.distortion_model = rs_model_to_ros(info_dict.get("model", ""))
    return ci

def make_publishers(serials):
    pubs = {}
    for s in serials:
        base = f"/realsense/{s}"
        pubs[s] = {
            "color_img":  rospy.Publisher(
                f"{base}/color/image_raw",   Image,
                queue_size=QUEUE_SIZE_IMAGE),
            "color_info": rospy.Publisher(
                f"{base}/color/camera_info", CameraInfo,
                queue_size=QUEUE_SIZE_INFO, latch=LATCH_INFO),
            "depth_img":  rospy.Publisher(
                f"{base}/depth/image_raw",   Image,
                queue_size=QUEUE_SIZE_IMAGE),
            "depth_info": rospy.Publisher(
                f"{base}/depth/camera_info", CameraInfo,
                queue_size=QUEUE_SIZE_INFO, latch=LATCH_INFO),
        }
        rospy.loginfo(f"Advertising {base}/{{color,depth}}/...")
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
    args, _ = parser.parse_known_args()

    rospy.init_node("agentlace_realsense_client", anonymous=False)
    rospy.loginfo(f"Connecting to agentlace server at {args.ip}:{args.port}")

    client = make_client(args.ip, args.port)
    pubs   = make_publishers(SERIALS)

    # Cache intrinsics - persist across frames, only update when server sends them
    info_cache = {s: {"color": None, "depth": None} for s in SERIALS}

    # Diagnostics
    none_count  = 0
    frame_count = 0
    t_last_log  = time.time()
    LOG_INTERVAL = 10.0   # log fps every N seconds

    # No Rate() here - publish as fast as server sends, don't artificially cap
    while not rospy.is_shutdown():
        obs = client.obs()

        # ── Handle None (server not ready / network hiccup) ───────────────────
        if obs is None:
            none_count += 1
            if none_count == MAX_CONSECUTIVE_NONE:
                rospy.logwarn(
                    f"Received {none_count} consecutive None observations - "
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
            rospy.loginfo(f"Publishing at ~{fps:.1f} fps")
            frame_count = 0
            t_last_log  = now

        # ── Per-camera publish ────────────────────────────────────────────────
        for serial in SERIALS:
            p = pubs[serial]

            # Update intrinsics cache (only when server sends - not every frame)
            c_info = obs.get(f"cam_{serial}_color_info")
            d_info = obs.get(f"cam_{serial}_depth_info")
            if c_info:
                info_cache[serial]["color"] = c_info
            if d_info:
                info_cache[serial]["depth"] = d_info

            color_fid = f"cam_{serial}_color_optical_frame"
            depth_fid = f"cam_{serial}_depth_optical_frame"

            # Color
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

            # Depth
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

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass