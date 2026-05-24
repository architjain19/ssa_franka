"""
RealSense camera server with calibrated intrinsics and extrinsics.

Streams RGB-D from D415 (scene) + D405 (wrist) and publishes calibrated
intrinsics and extrinsics per camera if calibration files are provided.

Usage:
    python rst_cam_server.py \
        --scene-intrinsics  config/scene_intrinsics.npz \
        --scene-extrinsics  config/T_base_scene.npz \
        --wrist-intrinsics  config/wrist_intrinsics.npz \
        --wrist-extrinsics  config/T_base_wrist.npz

You can pass any subset; cameras without calibration fall back to factory
intrinsics and publish no extrinsics (no TF on the client side).

Observation keys per camera:
    cam_{serial}_color         : (H, W, 3) uint8 BGR
    cam_{serial}_depth         : (Hd, Wd) uint16 raw depth
    cam_{serial}_depth_aligned : (H, W) uint16 depth aligned to color frame
    cam_{serial}_color_info    : dict (fx, fy, cx, cy, dist_coeffs, ...)
    cam_{serial}_depth_info    : dict (..., depth_scale)
    cam_{serial}_meta          : dict (role, depth_min_mm, depth_max_mm, serial)
    cam_{serial}_extrinsics    : dict (T_base_camera, position_xyz, quat_xyzw)
                                 — only published if extrinsics file given
"""

import argparse
import time

import numpy as np
import pyrealsense2 as rs
from agentlace.action import ActionServer, ActionConfig

# ── Hard-coded serials ─────────────────────────────────────────────────────
SCENE_SERIAL = "947122060531"   # D415 — fixed scene camera
# WRIST_SERIAL = "123622270802"   # D405 — end-effector wrist camera

CAMERA_ROLES = {
    # WRIST_SERIAL: "wrist",
    SCENE_SERIAL: "scene",
}

# Valid depth range per role (mm)
DEPTH_RANGE_MM = {
    # "wrist": (50,   2000),
    "scene": (300,  4000),
}


# ── CLI ────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # Per-camera calibration files
    p.add_argument("--scene-intrinsics",
                   help="intrinsics.npz for scene (D415)")
    p.add_argument("--scene-extrinsics",
                   help="T_base_camera.npz for scene (D415)")
    p.add_argument("--wrist-intrinsics",
                   help="intrinsics.npz for wrist (D405)")
    p.add_argument("--wrist-extrinsics",
                   help="T_base_camera.npz for wrist (D405)")

    # Per-camera resolutions (must match the calibration resolution!)
    p.add_argument("--scene-width",  type=int, default=1280)
    p.add_argument("--scene-height", type=int, default=720)
    p.add_argument("--wrist-width",  type=int, default=640)
    p.add_argument("--wrist-height", type=int, default=480)

    p.add_argument("--depth-width",  type=int, default=640)
    p.add_argument("--depth-height", type=int, default=480)
    p.add_argument("--fps",  type=int, default=15)
    p.add_argument("--port", type=int, default=6379)
    p.add_argument("--reset_cam", type=bool, default=False,
                   help="hardware reset cameras on startup to clear stale state")
    return p.parse_args()


# ── Calibration loaders ────────────────────────────────────────────────────
def load_calibrated_intrinsics(npz_path, expected_w, expected_h, label=""):
    """Load K + distortion from an intrinsics.npz file."""
    data = np.load(npz_path, allow_pickle=True)
    K = data["K"]
    dist = data["dist"].ravel()
    calib_size = data["image_size"]   # [width, height]

    if int(calib_size[0]) != expected_w or int(calib_size[1]) != expected_h:
        print(f"WARNING [{label}]: calibration was done at "
              f"{int(calib_size[0])}x{int(calib_size[1])}, but streaming at "
              f"{expected_w}x{expected_h}. Intrinsics will NOT match — "
              f"either change the streaming resolution or re-calibrate.")
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
    """3x3 rotation matrix → [x, y, z, w] (Shepperd's method)."""
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
    """Load T_base_camera and convert to a serializable dict."""
    data = np.load(npz_path, allow_pickle=True)
    T = data["T_base_camera"]
    return {
        "T_base_camera": T.tolist(),
        "position_xyz":  T[:3, 3].tolist(),
        "quat_xyzw":     _rotation_matrix_to_quat_xyzw(T[:3, :3]),
        "method":        str(data.get("method", "unknown")),
    }


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


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # ── Load calibration per role ──────────────────────────────────────────
    # calib[role] = {"intrinsics": dict|None, "extrinsics": dict|None}
    calib = {
        "scene": {"intrinsics": None, "extrinsics": None},
        "wrist": {"intrinsics": None, "extrinsics": None},
    }

    if args.scene_intrinsics:
        print(f"Loading SCENE intrinsics from {args.scene_intrinsics}")
        calib["scene"]["intrinsics"] = load_calibrated_intrinsics(
            args.scene_intrinsics, args.scene_width, args.scene_height, "scene")
        if calib["scene"]["intrinsics"]:
            ci = calib["scene"]["intrinsics"]
            print(f"  scene  fx={ci['fx']:.2f}  fy={ci['fy']:.2f}  "
                  f"cx={ci['cx']:.2f}  cy={ci['cy']:.2f}  "
                  f"dist={len(ci['dist_coeffs'])} params")

    if args.scene_extrinsics:
        print(f"Loading SCENE extrinsics from {args.scene_extrinsics}")
        calib["scene"]["extrinsics"] = load_extrinsics(args.scene_extrinsics)
        pos = calib["scene"]["extrinsics"]["position_xyz"]
        print(f"  scene T_base_camera pos = "
              f"[{pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f}] m")

    if args.wrist_intrinsics:
        print(f"Loading WRIST intrinsics from {args.wrist_intrinsics}")
        calib["wrist"]["intrinsics"] = load_calibrated_intrinsics(
            args.wrist_intrinsics, args.wrist_width, args.wrist_height, "wrist")
        if calib["wrist"]["intrinsics"]:
            ci = calib["wrist"]["intrinsics"]
            print(f"  wrist  fx={ci['fx']:.2f}  fy={ci['fy']:.2f}  "
                  f"cx={ci['cx']:.2f}  cy={ci['cy']:.2f}  "
                  f"dist={len(ci['dist_coeffs'])} params")

    if args.wrist_extrinsics:
        print(f"Loading WRIST extrinsics from {args.wrist_extrinsics}")
        calib["wrist"]["extrinsics"] = load_extrinsics(args.wrist_extrinsics)
        pos = calib["wrist"]["extrinsics"]["position_xyz"]
        print(f"  wrist T_base_camera pos = "
              f"[{pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f}] m")

    # ── Detect connected cameras ───────────────────────────────────────────
    ctx = rs.context()

    # =======================================================================
    # RESET ALL cameras on startup to clear any stale state
    if args.reset_cam:
        print("=======================================================================")
        print("Resetting cameras to clear stale state...")
        devices = ctx.query_devices()
        for dev in devices:
            dev.hardware_reset()
        print("=======================================================================")
    # =======================================================================
    
    serials = [d.get_info(rs.camera_info.serial_number) for d in ctx.devices]
    print(f"\nFound {len(serials)} camera(s): {serials}")

    for expected, role in CAMERA_ROLES.items():
        if expected not in serials:
            print(f"WARNING: expected {role} camera {expected} not connected.")

    # ── Start a pipeline per known camera ──────────────────────────────────
    pipelines = {}
    aligners = {}
    intrinsics_cache = {}

    for serial in serials:
        if serial not in CAMERA_ROLES:
            print(f"  {serial}: unknown serial — skipping")
            continue

        role = CAMERA_ROLES[serial]
        if role == "scene":
            c_w, c_h = args.scene_width, args.scene_height
        else:  # wrist
            c_w, c_h = args.wrist_width, args.wrist_height

        pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, c_w, c_h,
                          rs.format.bgr8, args.fps)
        cfg.enable_stream(rs.stream.depth, args.depth_width, args.depth_height,
                          rs.format.z16, args.fps)
        try:
            profile = pipeline.start(cfg)
        except RuntimeError as e:
            print(f"  {serial} ({role}): FAILED to start ({e})")
            print(f"    Try separate USB controllers or reduce --fps.")
            continue

        pipelines[serial] = pipeline
        aligners[serial] = rs.align(rs.stream.color)
        print(f"  {serial} ({role}): color {c_w}x{c_h}, "
              f"depth {args.depth_width}x{args.depth_height} @ {args.fps}fps")

        # Color intrinsics — calibrated if available
        color_stream = (profile.get_stream(rs.stream.color)
                        .as_video_stream_profile())
        ci = color_stream.get_intrinsics()
        if calib[role]["intrinsics"] is not None:
            color_info = calib[role]["intrinsics"]
            print(f"    using CALIBRATED color intrinsics")
        else:
            color_info = factory_intrinsics_dict(ci)
            print(f"    using factory color intrinsics")

        # Depth intrinsics — always factory
        depth_stream = (profile.get_stream(rs.stream.depth)
                        .as_video_stream_profile())
        di = depth_stream.get_intrinsics()
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        depth_info = factory_intrinsics_dict(
            di, extra={"depth_scale": depth_scale})

        intrinsics_cache[serial] = {"color": color_info, "depth": depth_info}
        print(f"    depth_scale={depth_scale:.6f}")

    # ── agentlace callbacks ────────────────────────────────────────────────
    def observation_callback(keys):
        obs = {}
        for serial, pipeline in pipelines.items():
            frames = pipeline.wait_for_frames()
            aligned = aligners[serial].process(frames)

            color_frame  = aligned.get_color_frame()
            depth_frame  = frames.get_depth_frame()
            depth_a_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame or not depth_a_frame:
                continue

            obs[f"cam_{serial}_color"] = np.asanyarray(color_frame.get_data())
            obs[f"cam_{serial}_depth"] = np.asanyarray(depth_frame.get_data())
            obs[f"cam_{serial}_depth_aligned"] = np.asanyarray(
                depth_a_frame.get_data())
            obs[f"cam_{serial}_color_info"] = intrinsics_cache[serial]["color"]
            obs[f"cam_{serial}_depth_info"] = intrinsics_cache[serial]["depth"]

            role = CAMERA_ROLES[serial]
            depth_min, depth_max = DEPTH_RANGE_MM[role]
            obs[f"cam_{serial}_meta"] = {
                "role":         role,
                "depth_min_mm": depth_min,
                "depth_max_mm": depth_max,
                "serial":       serial,
            }

            # Per-camera extrinsics (only if loaded for this role)
            if calib[role]["extrinsics"] is not None:
                obs[f"cam_{serial}_extrinsics"] = calib[role]["extrinsics"]
        return obs

    def action_callback(key, action):
        print(f"Received action '{key}': {action}")
        return {"status": "ok"}

    # ── Build observation keys ─────────────────────────────────────────────
    obs_keys = []
    for s in pipelines:
        role = CAMERA_ROLES[s]
        obs_keys += [
            f"cam_{s}_color", f"cam_{s}_depth", f"cam_{s}_depth_aligned",
            f"cam_{s}_color_info", f"cam_{s}_depth_info", f"cam_{s}_meta",
        ]
        if calib[role]["extrinsics"] is not None:
            obs_keys.append(f"cam_{s}_extrinsics")

    print(f"\nObservation keys: {obs_keys}")
    print(f"Server starting on port {args.port} ...")

    config = ActionConfig(
        port_number=args.port,
        action_keys=["command"],
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