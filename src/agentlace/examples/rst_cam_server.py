"""
RealSense camera server with calibrated intrinsics and extrinsics.

Streams RGB-D + IR data from all connected RealSense cameras via agentlace.
For the calibrated camera (--serial), publishes the ChArUco-calibrated
intrinsics and extrinsics instead of factory defaults.

Observation keys published per camera:
    cam_{serial}_color         : (H, W, 3) uint8 BGR
    cam_{serial}_depth         : (Hd, Wd) uint16 raw depth at native depth resolution
    cam_{serial}_depth_aligned : (H, W) uint16 depth aligned to color frame
    cam_{serial}_ir_left       : (Hir, Wir) uint8  rectified left IR (mono)
    cam_{serial}_ir_right      : (Hir, Wir) uint8  rectified right IR (mono)
    cam_{serial}_color_info    : dict (fx, fy, cx, cy, dist_coeffs, w, h)
    cam_{serial}_depth_info    : dict (fx, fy, cx, cy, depth_scale, ...)
    cam_{serial}_ir_info       : dict (fx, fy, cx, cy, w, h, baseline_m,
                                       distortion_model, dist_coeffs)
    cam_{serial}_extrinsics    : dict (T_base_camera 4x4, pos, quat_xyzw)
                                 — only for the calibrated camera
    cam_{serial}_meta          : dict (role, depth range, serial)

NOTE: The IR emitter is DISABLED by default on this server. This produces
clean IR images for learning-based stereo (e.g. FoundationStereo) at the
cost of slightly degraded factory depth on textureless surfaces.

Usage:
    python rst_cam_server.py \\
        --intrinsics ~/.../config/intrinsics.npz \\
        --extrinsics ~/.../config/T_base_camera.npz
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
                   help="Color stream width (MUST match calibration resolution)")
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--depth-width", type=int, default=640)
    p.add_argument("--depth-height", type=int, default=480)
    p.add_argument("--ir-width", type=int, default=848,
                   help="IR stream width. 848x480 @ 15fps is a good balance "
                        "between bandwidth and FoundationStereo quality.")
    p.add_argument("--ir-height", type=int, default=480)
    p.add_argument("--ir-fps", type=int, default=15,
                   help="IR fps. Keep this low (15) since FoundationStereo "
                        "can't process faster than ~5fps anyway.")
    p.add_argument("--fps", type=int, default=30, help="Color/depth fps")
    p.add_argument("--port", type=int, default=6379)
    p.add_argument("--enable-ir", action="store_true", default=True,
                   help="Stream IR1/IR2 (default: True)")
    p.add_argument("--no-ir", dest="enable_ir", action="store_false",
                   help="Disable IR streams (saves USB bandwidth)")
    return p.parse_args()


# ── Calibration loaders ────────────────────────────────────────────────────────
def load_calibrated_intrinsics(npz_path, expected_w, expected_h):
    data = np.load(npz_path, allow_pickle=True)
    K = data["K"]
    dist = data["dist"].ravel()
    calib_size = data["image_size"]
    if int(calib_size[0]) != expected_w or int(calib_size[1]) != expected_h:
        print(f"WARNING: calibration was at {int(calib_size[0])}x"
              f"{int(calib_size[1])}, streaming at {expected_w}x{expected_h}.")
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
    data = np.load(npz_path, allow_pickle=True)
    T = data["T_base_camera"]
    pos = T[:3, 3].tolist()
    q = _rotation_matrix_to_quat_xyzw(T[:3, :3])
    return {
        "T_base_camera": T.tolist(),
        "position_xyz":  pos,
        "quat_xyzw":     q,
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


# ── Camera role config ─────────────────────────────────────────────────────────
CAMERA_ROLES = {
    "123622270802": "wrist",
    "947122060531": "scene",
    "032522250211": "scene",
}
DEPTH_RANGE_MM = {
    "wrist": (50,   2000),
    "scene": (300,  4000),
}


def main():
    args = parse_args()

    if args.use_d455:
        print("Using D455 as calibrated camera")
        args.serial = "123622270802"
    else:
        print("Using D415 as calibrated camera")
        args.serial = "947122060531"

    # ── Load calibration ────────────────────────────────────────────────────
    calib_color_info = None
    if args.intrinsics:
        print(f"Loading calibrated intrinsics from {args.intrinsics}")
        calib_color_info = load_calibrated_intrinsics(
            args.intrinsics, args.width, args.height)
        if calib_color_info:
            print(f"  fx={calib_color_info['fx']:.2f}  "
                  f"fy={calib_color_info['fy']:.2f}  "
                  f"cx={calib_color_info['cx']:.2f}  "
                  f"cy={calib_color_info['cy']:.2f}")

    calib_extrinsics = None
    if args.extrinsics:
        print(f"Loading extrinsics from {args.extrinsics}")
        calib_extrinsics = load_extrinsics(args.extrinsics)
        pos = calib_extrinsics["position_xyz"]
        print(f"  T_base_camera pos = [{pos[0]:+.4f}, {pos[1]:+.4f}, "
              f"{pos[2]:+.4f}] m")

    # ── Detect cameras ──────────────────────────────────────────────────────
    ctx = rs.context()
    serials = [d.get_info(rs.camera_info.serial_number) for d in ctx.devices]
    print(f"\nFound {len(serials)} camera(s): {serials}")
    if args.serial and args.serial not in serials:
        print(f"WARNING: calibrated camera {args.serial} is not connected.")

    # ── Start pipelines ─────────────────────────────────────────────────────
    pipelines = {}
    aligners = {}
    intrinsics_cache = {}
    ir_enabled = {}        # serial -> bool — whether IR streams started OK

    for serial in serials:
        pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(serial)

        role = CAMERA_ROLES.get(serial, "scene")
        if serial == args.serial:
            c_w, c_h = args.width, args.height
        else:
            c_w, c_h = 640, 480

        cfg.enable_stream(rs.stream.color, c_w, c_h,
                          rs.format.bgr8, args.fps)
        cfg.enable_stream(rs.stream.depth, args.depth_width, args.depth_height,
                          rs.format.z16, args.fps)

        # Add IR streams (Y8 mono). On D415 family devices, IR resolution
        # MUST match depth resolution because depth is computed from these
        # IR imagers. We auto-snap IR to depth resolution.
        # Skip for D405 wrist cam — known USB 2 IR enumeration issues.
        ir_active = False
        if args.enable_ir and role != "wrist":
            ir_w = args.depth_width
            ir_h = args.depth_height
            ir_fps = args.ir_fps if args.ir_fps <= args.fps else args.fps
            if (ir_w, ir_h) != (args.ir_width, args.ir_height):
                print(f"  {serial}: snapping IR resolution to depth "
                      f"({args.ir_width}x{args.ir_height} -> {ir_w}x{ir_h}) "
                      f"because D415 requires depth and IR to match.")
            cfg.enable_stream(rs.stream.infrared, 1,
                              ir_w, ir_h, rs.format.y8, ir_fps)
            cfg.enable_stream(rs.stream.infrared, 2,
                              ir_w, ir_h, rs.format.y8, ir_fps)
            ir_active = True
            actual_ir_w, actual_ir_h, actual_ir_fps = ir_w, ir_h, ir_fps

        try:
            profile = pipeline.start(cfg)
        except RuntimeError as e:
            print(f"  {serial}: FAILED to start with IR ({e})")
            if ir_active:
                print(f"    Retrying without IR streams...")
                cfg = rs.config()
                cfg.enable_device(serial)
                cfg.enable_stream(rs.stream.color, c_w, c_h,
                                  rs.format.bgr8, args.fps)
                cfg.enable_stream(rs.stream.depth,
                                  args.depth_width, args.depth_height,
                                  rs.format.z16, args.fps)
                try:
                    profile = pipeline.start(cfg)
                    ir_active = False
                    print(f"    Started without IR.")
                except RuntimeError as e2:
                    print(f"    Still failed: {e2}")
                    continue
            else:
                continue

        pipelines[serial] = pipeline
        ir_enabled[serial] = ir_active
        ir_msg = (f", IR {actual_ir_w}x{actual_ir_h}@{actual_ir_fps}fps"
                  if ir_active else ", no IR")
        print(f"  {serial}: color {c_w}x{c_h}, "
              f"depth {args.depth_width}x{args.depth_height}@{args.fps}fps"
              f"{ir_msg}")

        aligners[serial] = rs.align(rs.stream.color)

        # IR emitter OFF — chosen to prioritize FoundationStereo quality
        depth_sensor = profile.get_device().first_depth_sensor()
        if depth_sensor.supports(rs.option.emitter_enabled):
            depth_sensor.set_option(rs.option.emitter_enabled, 0.0)
            print(f"  {serial}: IR emitter DISABLED (FoundationStereo mode)")

        # Color intrinsics
        color_stream = (profile.get_stream(rs.stream.color)
                        .as_video_stream_profile())
        ci = color_stream.get_intrinsics()
        if serial == args.serial and calib_color_info is not None:
            color_info = calib_color_info
            print(f"  {serial}: using CALIBRATED color intrinsics")
        else:
            color_info = factory_intrinsics_dict(ci)

        # Depth intrinsics
        depth_stream = (profile.get_stream(rs.stream.depth)
                        .as_video_stream_profile())
        di = depth_stream.get_intrinsics()
        depth_scale = depth_sensor.get_depth_scale()
        depth_info = factory_intrinsics_dict(
            di, extra={"depth_scale": depth_scale})

        # IR intrinsics + baseline + color->IR1 extrinsic
        # (only if IR streams started successfully)
        ir_info = None
        if ir_active:
            try:
                ir1_stream = (profile.get_stream(rs.stream.infrared, 1)
                              .as_video_stream_profile())
                ir2_stream = (profile.get_stream(rs.stream.infrared, 2)
                              .as_video_stream_profile())
                ir1_intr = ir1_stream.get_intrinsics()

                # Stereo baseline (IR1 -> IR2)
                stereo_extr = ir1_stream.get_extrinsics_to(ir2_stream)
                baseline_m = float(np.linalg.norm(stereo_extr.translation))

                # Color -> IR1 extrinsic (factory).
                # Downstream consumers can compose this with the ChArUco
                # T_base_color extrinsic to get T_base_ir1, which is the
                # frame FoundationStereo's depth lives in.
                T_color_ir1 = np.eye(4)
                try:
                    color_to_ir1 = (color_stream
                                    .get_extrinsics_to(ir1_stream))
                    T_color_ir1[:3, :3] = np.array(color_to_ir1.rotation).reshape(3, 3).T
                    T_color_ir1[:3, 3] = np.array(color_to_ir1.translation)
                except Exception as e:
                    print(f"  {serial}: failed to query color->IR1 "
                          f"extrinsic ({e}); using identity.")

                ir_info = {
                    "width":       ir1_intr.width,
                    "height":      ir1_intr.height,
                    "fx":          ir1_intr.fx,
                    "fy":          ir1_intr.fy,
                    "cx":          ir1_intr.ppx,
                    "cy":          ir1_intr.ppy,
                    "dist_coeffs": list(ir1_intr.coeffs),
                    "model":       str(ir1_intr.model),
                    "baseline_m":  baseline_m,
                    "T_color_ir1": T_color_ir1.tolist(),
                    "source":      "factory",
                    "note":        "Rectified stereo pair. K applies to "
                                   "ir_left. T_color_ir1 is the rigid "
                                   "transform from color frame to IR1 "
                                   "frame (factory calibration).",
                }
                t = T_color_ir1[:3, 3]
                print(f"  {serial}: IR fx={ir1_intr.fx:.2f}  "
                      f"baseline={baseline_m*1000:.2f}mm  "
                      f"color->IR1 t=[{t[0]*1000:+.1f},{t[1]*1000:+.1f},"
                      f"{t[2]*1000:+.1f}]mm")
            except Exception as e:
                print(f"  {serial}: failed to query IR intrinsics ({e})")
                ir_enabled[serial] = False

        intrinsics_cache[serial] = {
            "color": color_info,
            "depth": depth_info,
            "ir":    ir_info,
        }
        print(f"  {serial}: depth_scale={depth_scale:.6f}")

    # ── Observation callback ────────────────────────────────────────────────
    def observation_callback(keys):
        obs = {}
        for serial, pipeline in pipelines.items():
            try:
                frames = pipeline.wait_for_frames(timeout_ms=2000)
            except RuntimeError as e:
                print(f"[{serial}] frame timeout: {e}")
                continue

            aligned_frames = aligners[serial].process(frames)
            color_frame   = aligned_frames.get_color_frame()
            depth_frame   = frames.get_depth_frame()
            depth_aligned = aligned_frames.get_depth_frame()
            if not color_frame or not depth_frame or not depth_aligned:
                continue

            obs[f"cam_{serial}_color"]         = np.asanyarray(color_frame.get_data())
            obs[f"cam_{serial}_depth"]         = np.asanyarray(depth_frame.get_data())
            obs[f"cam_{serial}_depth_aligned"] = np.asanyarray(depth_aligned.get_data())
            obs[f"cam_{serial}_color_info"]    = intrinsics_cache[serial]["color"]
            obs[f"cam_{serial}_depth_info"]    = intrinsics_cache[serial]["depth"]

            # IR — both frames pulled from the SAME frameset, so they're
            # hardware-synchronized. Critical for FoundationStereo.
            if ir_enabled.get(serial, False):
                ir_l = frames.get_infrared_frame(1)
                ir_r = frames.get_infrared_frame(2)
                if ir_l and ir_r:
                    obs[f"cam_{serial}_ir_left"]  = np.asanyarray(ir_l.get_data())
                    obs[f"cam_{serial}_ir_right"] = np.asanyarray(ir_r.get_data())
                    obs[f"cam_{serial}_ir_info"]  = intrinsics_cache[serial]["ir"]

            role = CAMERA_ROLES.get(serial, "scene")
            depth_min, depth_max = DEPTH_RANGE_MM[role]
            obs[f"cam_{serial}_meta"] = {
                "role":         role,
                "depth_min_mm": depth_min,
                "depth_max_mm": depth_max,
                "serial":       serial,
                "ir_available": ir_enabled.get(serial, False),
            }

            if serial == args.serial and calib_extrinsics is not None:
                obs[f"cam_{serial}_extrinsics"] = calib_extrinsics
        return obs

    def action_callback(key, action):
        print(f"Received action '{key}': {action}")
        return {"status": "ok"}

    # ── Build keys list ─────────────────────────────────────────────────────
    obs_keys = []
    for s in pipelines:
        obs_keys += [
            f"cam_{s}_color", f"cam_{s}_depth", f"cam_{s}_depth_aligned",
            f"cam_{s}_color_info", f"cam_{s}_depth_info",
            f"cam_{s}_meta",
        ]
        if ir_enabled.get(s, False):
            obs_keys += [
                f"cam_{s}_ir_left", f"cam_{s}_ir_right", f"cam_{s}_ir_info",
            ]
        if s == args.serial and calib_extrinsics is not None:
            obs_keys.append(f"cam_{s}_extrinsics")

    act_keys = ["command"]

    print(f"\nObservation keys ({len(obs_keys)}):")
    for k in obs_keys:
        print(f"  {k}")
    print(f"\nServer starting on port {args.port} ...")

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