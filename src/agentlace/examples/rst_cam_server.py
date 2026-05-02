"""
RealSense camera server with calibrated intrinsics and extrinsics.

Streams RGB-D data from all connected RealSense cameras via agentlace.
For the calibrated camera (--serial), publishes the ChArUco-calibrated
intrinsics and extrinsics instead of factory defaults.

Usage:
    python ~/archit/ssa_ws/src/agentlace/examples/rst_cam_server.py --intrinsics ~/archit/ssa_ws/src/agentlace/examples/config/intrinsics.npz --extrinsics ~/archit/ssa_ws/src/agentlace/examples/config/T_base_camera.npz

Observation keys published per camera:
    cam_{serial}_color         : (H, W, 3) uint8 BGR
    cam_{serial}_depth         : (Hd, Wd) uint16 raw depth at native depth resolution
    cam_{serial}_depth_aligned : (H, W) uint16 depth aligned to color frame
                                 (same resolution as color; pixel (u,v) here
                                 corresponds to pixel (u,v) in color)
    cam_{serial}_color_info    : dict with fx, fy, cx, cy, dist_coeffs, width, height
    cam_{serial}_depth_info    : dict with fx, fy, cx, cy, depth_scale, ...
    cam_{serial}_extrinsics    : dict with T_base_camera (4x4 list), pos, quat_xyzw
                                 (only for calibrated camera)
"""

import argparse
import time
import json

import numpy as np
import pyrealsense2 as rs
from agentlace.action import ActionServer, ActionConfig


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--intrinsics",
                   help="Path to intrinsics.npz from ChArUco calibration")
    p.add_argument("--extrinsics",
                   help="Path to T_base_camera.npz from extrinsics calibration")
    p.add_argument("--serial", default="032522250211",
                   help="Serial of the calibrated camera (default: 032522250211)")
    p.add_argument("--use_d455", action="store_true",
                   help="Use D455 camera (default: use D415)")
    p.add_argument("--width", type=int, default=1280,
                   help="Color stream width (default: 1280; MUST match the "
                        "resolution used during intrinsics calibration)")
    p.add_argument("--height", type=int, default=720,
                   help="Color stream height (default: 720)")
    p.add_argument("--depth-width", type=int, default=640)
    p.add_argument("--depth-height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--port", type=int, default=6379,
                   help="agentlace server port")
    return p.parse_args()


# ── Load calibration data ──────────────────────────────────────────────────────
def load_calibrated_intrinsics(npz_path, expected_w, expected_h):
    """Load K + distortion from intrinsics.npz.

    Returns a dict matching the color_info schema, or None on failure.
    """
    data = np.load(npz_path, allow_pickle=True)
    K = data["K"]
    dist = data["dist"].ravel()
    calib_size = data["image_size"]   # [width, height]

    if int(calib_size[0]) != expected_w or int(calib_size[1]) != expected_h:
        print(f"WARNING: calibration was done at {int(calib_size[0])}x"
              f"{int(calib_size[1])}, but streaming at {expected_w}x"
              f"{expected_h}.  Intrinsics will NOT match — either change "
              f"--width/--height or re-calibrate at this resolution.")
        return None

    return {
        "width":  expected_w,
        "height": expected_h,
        "fx": float(K[0, 0]),
        "fy": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
        "dist_coeffs": dist.tolist(),
        "model": "rational_polynomial" if len(dist) > 5 else "plumb_bob",
        "source": "charuco_calibration",
    }


def _rotation_matrix_to_quat_xyzw(R):
    """Convert 3x3 rotation matrix to quaternion [x, y, z, w] (numpy only)."""
    # Shepperd's method - numerically stable, no scipy needed
    tr = np.trace(R)
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
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
    return [float(x), float(y), float(z), float(w)]


def load_extrinsics(npz_path):
    """Load T_base_camera from T_base_camera.npz.

    Returns a dict with the 4x4 matrix and convenience fields.
    """
    data = np.load(npz_path, allow_pickle=True)
    T = data["T_base_camera"]
    pos = T[:3, 3].tolist()
    q = _rotation_matrix_to_quat_xyzw(T[:3, :3])

    return {
        "T_base_camera": T.tolist(),       # 4x4 nested list
        "position_xyz":  pos,              # [x, y, z] meters
        "quat_xyzw":     q,                # [x, y, z, w]
        "method":        str(data.get("method", "unknown")),
    }


# ── Build factory intrinsics dict from pyrealsense2 intrinsics ─────────────
def factory_intrinsics_dict(intr, extra=None):
    d = {
        "width":  intr.width,
        "height": intr.height,
        "fx":     intr.fx,
        "fy":     intr.fy,
        "cx":     intr.ppx,
        "cy":     intr.ppy,
        "dist_coeffs": list(intr.coeffs),
        "model":  str(intr.model),
        "source": "factory",
    }
    if extra:
        d.update(extra)
    return d

# ── Camera role config ─────────────────────────────────────────────────────
# Add this near the top of the file, after parse_args()

CAMERA_ROLES = {
    "123622270802": "wrist",      # D405 — close range, end-effector
    "947122060531": "scene",      # D415 — mid range, fixed/calibrated
    "032522250211": "scene",      # D455 — mid range, fixed/calibrated
}

# Valid depth range per role (mm) — used as ROS param / metadata
DEPTH_RANGE_MM = {
    "wrist": (50,   2000),  # generous — warn only on clearly bogus values
    "scene": (300,  4000),
}

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    if args.use_d455:
        print("Using D455 as calibrated camera")
        args.serial = "123622270802"
    else:
        print("Using D415 as calibrated camera")
        args.serial = "947122060531"

    # ── Load calibration files ──────────────────────────────────────────────
    calib_color_info = None
    if args.intrinsics:
        print(f"Loading calibrated intrinsics from {args.intrinsics}")
        calib_color_info = load_calibrated_intrinsics(
            args.intrinsics, args.width, args.height)
        if calib_color_info:
            print(f"  fx={calib_color_info['fx']:.2f}  "
                  f"fy={calib_color_info['fy']:.2f}  "
                  f"cx={calib_color_info['cx']:.2f}  "
                  f"cy={calib_color_info['cy']:.2f}  "
                  f"dist_coeffs={len(calib_color_info['dist_coeffs'])} params")

    calib_extrinsics = None
    if args.extrinsics:
        print(f"Loading extrinsics from {args.extrinsics}")
        calib_extrinsics = load_extrinsics(args.extrinsics)
        pos = calib_extrinsics["position_xyz"]
        print(f"  T_base_camera pos = [{pos[0]:+.4f}, {pos[1]:+.4f}, "
              f"{pos[2]:+.4f}] m")

    # ── Detect all connected RealSense cameras ──────────────────────────────
    ctx = rs.context()
    serials = [d.get_info(rs.camera_info.serial_number) for d in ctx.devices]
    print(f"\nFound {len(serials)} camera(s): {serials}")

    if args.serial and args.serial not in serials:
        print(f"WARNING: calibrated camera {args.serial} is not connected.")

    # ── Start a pipeline per camera + cache intrinsics ──────────────────────
    pipelines = {}
    aligners = {}           # serial -> rs.align  (depth → color frame)
    intrinsics_cache = {}   # serial -> {"color": dict, "depth": dict}

    for serial in serials:
        pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(serial)

        # Only the calibrated camera needs the calibration resolution (1280x720).
        # Other cameras default to 640x480 to stay within USB 3.0 bandwidth
        # when multiple cameras share the same controller.
        role = CAMERA_ROLES.get(serial, "scene")
        if serial == args.serial:
            # Calibrated scene camera — use calibration resolution
            c_w, c_h = args.width, args.height
        elif role == "wrist":
            # D405 wrist cam — 640x480 is fine, it's close range
            c_w, c_h = 640, 480
        else:
            c_w, c_h = 640, 480

        cfg.enable_stream(rs.stream.color, c_w, c_h,
                          rs.format.bgr8, args.fps)
        cfg.enable_stream(rs.stream.depth, args.depth_width, args.depth_height,
                          rs.format.z16, args.fps)
        try:
            profile = pipeline.start(cfg)
        except RuntimeError as e:
            print(f"  {serial}: FAILED to start ({e})")
            print(f"    Try putting cameras on separate USB controllers, or "
                  f"reduce --fps.")
            continue
        pipelines[serial] = pipeline
        print(f"  {serial}: streaming color {c_w}x{c_h}, "
              f"depth {args.depth_width}x{args.depth_height} @ {args.fps}fps")

        # Align depth to color frame so each (u, v) in the aligned depth
        # corresponds to the same (u, v) in the color image.
        aligners[serial] = rs.align(rs.stream.color)

        # Color intrinsics — use calibrated if this is the calibrated camera
        color_stream = (profile.get_stream(rs.stream.color)
                        .as_video_stream_profile())
        ci = color_stream.get_intrinsics()

        if serial == args.serial and calib_color_info is not None:
            color_info = calib_color_info
            print(f"  {serial}: using CALIBRATED color intrinsics")
        else:
            color_info = factory_intrinsics_dict(ci)
            print(f"  {serial}: using factory color intrinsics")

        # Depth intrinsics — always factory (we only calibrated the color cam)
        depth_stream = (profile.get_stream(rs.stream.depth)
                        .as_video_stream_profile())
        di = depth_stream.get_intrinsics()
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()

        depth_info = factory_intrinsics_dict(
            di, extra={"depth_scale": depth_scale})

        intrinsics_cache[serial] = {
            "color": color_info,
            "depth": depth_info,
        }
        print(f"  {serial}: depth_scale={depth_scale:.6f}")

    # ── agentlace callbacks ─────────────────────────────────────────────────
    def observation_callback(keys):
        obs = {}
        for serial, pipeline in pipelines.items():
            frames = pipeline.wait_for_frames()

            # Align depth → color viewport (resamples depth to color resolution
            # so pixel (u,v) in aligned_depth == pixel (u,v) in color image)
            aligned_frames = aligners[serial].process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = frames.get_depth_frame()           # raw (native res)
            depth_aligned = aligned_frames.get_depth_frame() # aligned to color
            if not color_frame or not depth_frame or not depth_aligned:
                continue

            obs[f"cam_{serial}_color"] = np.asanyarray(color_frame.get_data())
            obs[f"cam_{serial}_depth"] = np.asanyarray(depth_frame.get_data())
            obs[f"cam_{serial}_depth_aligned"] = np.asanyarray(
                depth_aligned.get_data())
            obs[f"cam_{serial}_color_info"] = intrinsics_cache[serial]["color"]
            obs[f"cam_{serial}_depth_info"] = intrinsics_cache[serial]["depth"]

            role = CAMERA_ROLES.get(serial, "scene")
            depth_min, depth_max = DEPTH_RANGE_MM[role]

            obs[f"cam_{serial}_meta"] = {
                "role":         role,
                "depth_min_mm": depth_min,
                "depth_max_mm": depth_max,
                "serial":       serial,
            }
            obs[f"cam_{serial}_meta"] = {
                "role":      role,
                "depth_min_mm": depth_min,
                "depth_max_mm": depth_max,
                "serial":    serial,
            }

            if serial == args.serial and calib_extrinsics is not None:
                obs[f"cam_{serial}_extrinsics"] = calib_extrinsics
        return obs

    def action_callback(key, action):
        print(f"Received action '{key}': {action}")
        return {"status": "ok"}

    # ── Build observation + action key lists ────────────────────────────────
    obs_keys = []
    for s in pipelines:     # only cameras that started successfully
        obs_keys += [
            f"cam_{s}_color", f"cam_{s}_depth", f"cam_{s}_depth_aligned",
            f"cam_{s}_color_info", f"cam_{s}_depth_info",
            f"cam_{s}_meta",        # metadata
        ]
        if s == args.serial and calib_extrinsics is not None:
            obs_keys.append(f"cam_{s}_extrinsics")

    act_keys = ["command"]

    print(f"\nObservation keys: {obs_keys}")
    print(f"Server starting on port {args.port} ...")

    config = ActionConfig(port_number=args.port,
                          action_keys=act_keys,
                          observation_keys=obs_keys)
    server = ActionServer(config, observation_callback, action_callback)
    server.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down")
        for p in pipelines.values():
            p.stop()


if __name__ == "__main__":
    main()