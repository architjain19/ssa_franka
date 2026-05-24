#!/usr/bin/env python3
"""
agentlace_zed_ros_client.py
ROS1 client for the ZED agentlace server.

Publishes calibrated intrinsics, depth-aligned-to-color, and a static TF
per camera:

  Scene (ZED 2i, fixed):   parent = panda_link0   (extrinsics.frame == 'base')
  Wrist (ZED-Mini, on EE): parent = panda_hand    (extrinsics.frame == 'ee')

Topics published (per camera, role = 'scene' or 'wrist'):
  /zed/{role}/color/image_raw                       sensor_msgs/Image
  /zed/{role}/color/camera_info                     sensor_msgs/CameraInfo
  /zed/{role}/depth/image_raw                       sensor_msgs/Image  (16UC1, mm)
  /zed/{role}/depth/camera_info                     sensor_msgs/CameraInfo
  /zed/{role}/aligned_depth_to_color/image_raw      sensor_msgs/Image  (16UC1, mm)
  /zed/{role}/aligned_depth_to_color/camera_info    sensor_msgs/CameraInfo

ROS params set per role:
  /zed/{role}/depth_min_mm
  /zed/{role}/depth_max_mm
  /zed/{role}/depth_scale
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
# EDIT to match your ZED serials (integers, not strings).
SCENE_SERIAL = 39668372    # ZED 2i — scene
WRIST_SERIAL = 16744838    # ZED-Mini — wrist

# SERIALS = [SCENE_SERIAL, WRIST_SERIAL]
SERIALS = [WRIST_SERIAL]

CAMERA_ROLES = {
    WRIST_SERIAL: "wrist",
    SCENE_SERIAL: "scene",
}

# Subscribe to all keys for both cameras
observation_keys = []
for s in SERIALS:
    observation_keys += [
        f"cam_{s}_color",
        f"cam_{s}_depth",
        f"cam_{s}_depth_aligned",
        f"cam_{s}_color_info",
        f"cam_{s}_depth_info",
        f"cam_{s}_meta",
        f"cam_{s}_extrinsics",
    ]
observation_keys.pop(observation_keys.index(f"cam_{WRIST_SERIAL}_extrinsics"))
print(observation_keys)
action_keys = ["command"]

QUEUE_SIZE_IMAGE = 1
QUEUE_SIZE_INFO  = 10
LATCH_INFO = True

MAX_CONSECUTIVE_NONE = 30
RECONNECT_AFTER_NONE = 100


# ── Helpers ───────────────────────────────────────────────────────────────────
MODEL_TO_ROS = {
    "plumb_bob":            "plumb_bob",
    "rational_polynomial":  "rational_polynomial",
    # ZED factory uses brown-conrady, but our server normalizes to plumb_bob
    "distortion.brown_conrady":         "plumb_bob",
    "distortion.inverse_brown_conrady": "plumb_bob",
}


def model_to_ros(model_str: str) -> str:
    return MODEL_TO_ROS.get(model_str, "plumb_bob")


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
    ci.distortion_model = model_to_ros(info.get("model", ""))
    return ci


def extrinsics_to_tf(ext, parent_frame, child_frame, stamp):
    """Build a TransformStamped from the server's extrinsics dict."""
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
        role = CAMERA_ROLES.get(s, str(s))
        base = f"/zed/{role}"
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
    parser.add_argument("--port", type=int, default=6380)
    parser.add_argument("--base-frame", default="panda_link0",
                        help="TF parent frame for SCENE extrinsics "
                             "(extrinsics.frame == 'base')")
    parser.add_argument("--ee-frame", default="panda_hand",
                        help="TF parent frame for WRIST extrinsics "
                             "(extrinsics.frame == 'ee')")
    parser.add_argument("--scene-frame",
                        default="zed_scene_left_optical_frame",
                        help="TF child frame for scene (ZED 2i) extrinsics")
    parser.add_argument("--wrist-frame",
                        default="zed_wrist_left_optical_frame",
                        help="TF child frame for wrist (ZED-Mini) extrinsics")
    args, _ = parser.parse_known_args()

    rospy.init_node("agentlace_zed_client", anonymous=False)
    rospy.loginfo(f"Connecting to agentlace server at {args.ip}:{args.port}")

    client = make_client(args.ip, args.port)
    pubs   = make_publishers(SERIALS)

    # Per-camera TF child frame
    child_frames = {
        SCENE_SERIAL: args.scene_frame,
        WRIST_SERIAL: args.wrist_frame,
    }

    tf_broadcaster = tf2_ros.StaticTransformBroadcaster()
    extrinsics_published = {s: False for s in SERIALS}

    info_cache = {s: {"color": None, "depth": None} for s in SERIALS}

    none_count  = 0
    frame_count = 0
    t_last_log  = time.time()
    LOG_INTERVAL = 10.0

    while not rospy.is_shutdown():
        obs = client.obs()

        # ── Handle None (server down or warming up) ──────────────────────────
        if obs is None:
            none_count += 1
            if none_count == MAX_CONSECUTIVE_NONE:
                rospy.logwarn(
                    f"Received {none_count} consecutive None observations — "
                    f"is the server running at {args.ip}:{args.port}?")
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

        # ── Extrinsics → static TF, once per camera ──────────────────────────
        # Scene cam: frame='base'  -> parent = panda_link0  (fixed)
        # Wrist cam: frame='ee'    -> parent = panda_hand   (also fixed,
        #                              but the parent itself moves via FK)
        for serial in SERIALS:
            if extrinsics_published[serial]:
                continue
            ext = obs.get(f"cam_{serial}_extrinsics")
            if ext is None or not isinstance(ext, dict):
                continue

            frame = ext.get("frame", "base")
            if frame == "base":
                parent = args.base_frame
            elif frame == "ee":
                parent = args.ee_frame
            else:
                rospy.logwarn(
                    f"[{serial}] unknown extrinsics frame '{frame}', "
                    f"defaulting parent to base ({args.base_frame})")
                parent = args.base_frame

            child = child_frames[serial]
            tf_msg = extrinsics_to_tf(ext, parent, child, stamp)
            tf_broadcaster.sendTransform(tf_msg)
            pos = ext["position_xyz"]
            rospy.loginfo(
                f"Published static TF: {parent} → {child}  "
                f"pos=[{pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f}]  "
                f"(frame={frame})")
            extrinsics_published[serial] = True

        # ── Per-camera image publish ─────────────────────────────────────────
        for serial in SERIALS:
            p = pubs[serial]
            role = CAMERA_ROLES[serial]

            c_info = obs.get(f"cam_{serial}_color_info")
            d_info = obs.get(f"cam_{serial}_depth_info")
            if c_info: info_cache[serial]["color"] = c_info
            if d_info: info_cache[serial]["depth"] = d_info

            # ZED depth is already aligned to the left eye, so depth and
            # color share the same optical frame.
            color_fid = f"zed_{role}_left_optical_frame"
            depth_fid = color_fid

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

            # Raw depth (same frame as color on ZED)
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

            # Aligned depth (same data as raw depth on ZED, kept for API parity
            # with the RealSense client so downstream code doesn't have to fork)
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
                rospy.set_param(f"/zed/{meta['role']}/depth_min_mm",
                                meta["depth_min_mm"])
                rospy.set_param(f"/zed/{meta['role']}/depth_max_mm",
                                meta["depth_max_mm"])
                if info_cache[serial]["depth"] and \
                        "depth_scale" in info_cache[serial]["depth"]:
                    rospy.set_param(
                        f"/zed/{meta['role']}/depth_scale",
                        info_cache[serial]["depth"]["depth_scale"])

            # Wrist depth-range sanity warning
            if role == "wrist" and depth_aligned is not None and \
                    isinstance(depth_aligned, np.ndarray) and meta:
                # depth_aligned is uint16 mm already (server casts ZED's
                # float-mm output to uint16 mm). depth_scale = 0.001 (mm->m).
                valid = depth_aligned[depth_aligned > 0]
                if valid.size > 0:
                    median_mm = float(np.median(valid))
                    if not (meta["depth_min_mm"] < median_mm
                            < meta["depth_max_mm"]):
                        rospy.logwarn_throttle(5.0,
                            f"[{serial}] Median depth {median_mm:.0f}mm outside "
                            f"[{meta['depth_min_mm']}, {meta['depth_max_mm']}]mm "
                            f"— is the camera aimed correctly?")
                    else:
                        rospy.logdebug(
                            f"[{serial}] Median depth {median_mm:.0f}mm ok")


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass