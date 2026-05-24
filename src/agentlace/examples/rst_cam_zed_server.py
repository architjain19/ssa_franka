"""
ZED camera server with calibrated intrinsics and extrinsics.

Streams left RGB + depth from a ZED-2i (scene) + ZED-Mini (wrist) and
publishes calibrated intrinsics and extrinsics per camera if calibration
files are provided.

Usage:

    python /home/daphne/archit/ssa_ws/src/agentlace/examples/rst_cam_zed_server.py \
        --scene-intrinsics /home/daphne/archit/ssa_ws/src/agentlace/examples/config/zed/scene/intrinsics.npz \
        --scene-extrinsics /home/daphne/archit/ssa_ws/src/agentlace/examples/config/zed/scene/T_base_camera.npz \
        --scene-resolution HD1080 \
        --use-rectified-stream 0 \
        --depth-mode ULTRA
        
    python zed_cam_server.py \
        --scene-intrinsics  config/scene/intrinsics.npz \
        --scene-extrinsics  config/scene/T_base_camera.npz \
        --wrist-intrinsics  config/wrist/intrinsics.npz \
        --wrist-extrinsics  config/wrist/T_ee_camera.npz

Extrinsics file can contain either:
    T_base_camera   : fixed camera in base frame    (scene/eye-to-hand)
    T_ee_camera     : camera in end-effector frame  (wrist/eye-in-hand)
The server publishes whichever it finds; the client is responsible for
chaining T_base_ee @ T_ee_camera in the wrist case.

You can pass any subset; cameras without calibration fall back to factory
intrinsics and publish no extrinsics.

IMPORTANT: the streaming resolution MUST match the calibration resolution,
since intrinsics are resolution-specific. The server warns and falls back
to factory intrinsics if they don't match.

By default, this server captures the SDK-rectified left image (sl.VIEW.LEFT)
and reports near-zero distortion. If you calibrated the UNRECTIFIED stream
(the script default), pass --use-rectified-stream=False so the calibrated K
matches the actual pixels you're streaming.

Observation keys per camera:
    cam_{serial}_color         : (H, W, 3) uint8 BGR (left eye)
    cam_{serial}_depth         : (H, W) uint16 depth in mm
    cam_{serial}_depth_aligned : same as depth (ZED already aligns to left)
    cam_{serial}_color_info    : dict (fx, fy, cx, cy, dist_coeffs, ...)
    cam_{serial}_depth_info    : dict (..., depth_scale)
    cam_{serial}_meta          : dict (role, depth_min_mm, depth_max_mm, serial)
    cam_{serial}_extrinsics    : dict (T_base_camera or T_ee_camera, ...)
                                 — only published if extrinsics file given
"""

import argparse
import time

import numpy as np
import pyzed.sl as sl
from agentlace.action import ActionServer, ActionConfig

# ── Hard-coded serials ─────────────────────────────────────────────────────
# EDIT these to your actual ZED serial numbers. Find them by running:
#   python zed_charuco_calib.py devices
# SCENE_SERIAL = 39668372     # ZED 2i — fixed scene camera
WRIST_SERIAL = 16744838     # ZED-Mini — end-effector wrist camera

CAMERA_ROLES = {
    WRIST_SERIAL: "wrist",
    # SCENE_SERIAL: "scene",
}

# Valid depth range per role (mm)
# ZED-Mini min depth is ~10 cm, ZED-2i min is ~30 cm.
DEPTH_RANGE_MM = {
    "wrist": (100,  2000),
    # "scene": (300,  4000),
}

# Resolution strings accepted by sl.RESOLUTION
ZED_RESOLUTIONS = ["HD2K", "HD1080", "HD1200", "HD1536", "HD720", "SVGA", "VGA"]


# ── CLI ────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # Per-camera calibration files
    p.add_argument("--scene-intrinsics",
                   help="intrinsics.npz for scene (ZED 2i)")
    p.add_argument("--scene-extrinsics",
                   help="T_base_camera.npz for scene (ZED 2i)")
    p.add_argument("--wrist-intrinsics",
                   help="intrinsics.npz for wrist (ZED-Mini)")
    p.add_argument("--wrist-extrinsics",
                   help="T_ee_camera.npz (eye-in-hand) OR T_base_camera.npz "
                        "for wrist (ZED-Mini)")

    # Per-camera resolution (must match the resolution used during calibration)
    p.add_argument("--scene-resolution", default="HD720",
                   choices=ZED_RESOLUTIONS)
    p.add_argument("--wrist-resolution", default="HD720",
                   choices=ZED_RESOLUTIONS)

    p.add_argument("--fps", type=int, default=7)
    p.add_argument("--port", type=int, default=6380)

    # ZED depth settings (shared across both cameras)
    p.add_argument("--depth-mode", default="NEURAL",
                   choices=["NONE", "PERFORMANCE", "QUALITY", "ULTRA",
                            "NEURAL", "NEURAL_PLUS"],
                   help="ZED depth quality (default NEURAL). Use NONE to "
                        "disable depth.")
    p.add_argument("--depth-min", type=float, default=0.1,
                   help="ZED depth minimum (m), default 0.1")
    p.add_argument("--depth-max", type=float, default=5.0,
                   help="ZED depth maximum (m), default 5.0")

    p.add_argument("--use-rectified-stream", type=int, default=1,
                   help="1 = stream sl.VIEW.LEFT (SDK-rectified, recommended "
                        "for downstream use). 0 = stream LEFT_UNRECTIFIED. "
                        "MUST match whatever you calibrated. Default 1.")
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
    """Load T_base_camera (eye-to-hand) OR T_ee_camera (eye-in-hand).

    Returns a dict with a `frame` field telling the client which is which:
      frame="base"  →  T is camera in base frame   (apply directly)
      frame="ee"    →  T is camera in EE frame    (client must chain
                                                   T_base_camera = T_base_ee @ T)
    """
    data = np.load(npz_path, allow_pickle=True)
    keys = list(data.keys())

    if "T_base_camera" in keys:
        T = data["T_base_camera"]
        frame = "base"
        t_key = "T_base_camera"
    elif "T_ee_camera" in keys:
        T = data["T_ee_camera"]
        frame = "ee"
        t_key = "T_ee_camera"
    else:
        raise KeyError(
            f"{npz_path}: expected either 'T_base_camera' or 'T_ee_camera' "
            f"key in npz, found {keys}")

    return {
        t_key:           T.tolist(),
        "frame":         frame,     # "base" or "ee"
        "position_xyz":  T[:3, 3].tolist(),
        "quat_xyzw":     _rotation_matrix_to_quat_xyzw(T[:3, :3]),
        "method":        str(data.get("method", "unknown")),
    }


# ── ZED helpers ────────────────────────────────────────────────────────────
def list_zed_devices():
    """Return list of (serial:int, model:str) for connected ZEDs."""
    out = []
    for d in sl.Camera.get_device_list():
        try:
            serial = int(d.serial_number)
        except Exception:
            serial = 0
            print(f"WARNING: failed to parse serial number from device {d.serial_number}")
        model = str(d.camera_model).replace("MODEL.", "")
        out.append((serial, model))
    return out


def open_zed_for_role(serial, resolution_str, fps, depth_mode_str,
                      depth_min_m, depth_max_m, label=""):
    """Open a ZED with depth enabled, ready for streaming."""
    init = sl.InitParameters()
    init.set_from_serial_number(int(serial))
    init.camera_resolution = getattr(sl.RESOLUTION, resolution_str)
    init.camera_fps = fps
    init.depth_mode = getattr(sl.DEPTH_MODE, depth_mode_str)
    init.coordinate_units = sl.UNIT.MILLIMETER       # depth in mm (uint16-ish)
    init.depth_minimum_distance = float(depth_min_m * 1000.0) \
        if depth_mode_str != "NONE" else -1.0
    init.depth_maximum_distance = float(depth_max_m * 1000.0) \
        if depth_mode_str != "NONE" else -1.0
    init.camera_image_flip = sl.FLIP_MODE.OFF
    init.sdk_verbose = 0

    zed = sl.Camera()
    err = zed.open(init)
    if err != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(
            f"[{label}] zed.open(serial={serial}) failed: {err}")

    cam_info = zed.get_camera_information()
    try:
        res = cam_info.camera_configuration.resolution
    except AttributeError:
        res = cam_info.camera_resolution
    image_size_wh = (int(res.width), int(res.height))

    try:
        cp = cam_info.camera_configuration.calibration_parameters
    except AttributeError:
        cp = cam_info.calibration_parameters
    lc = cp.left_cam
    factory_color = {
        "width":  image_size_wh[0],
        "height": image_size_wh[1],
        "fx":     float(lc.fx),
        "fy":     float(lc.fy),
        "cx":     float(lc.cx),
        "cy":     float(lc.cy),
        "dist_coeffs": [float(x) for x in lc.disto],   # k1,k2,p1,p2,k3
        "model":  "plumb_bob",
        "source": "factory",
    }

    # Manual exposure example
    zed.set_camera_settings(sl.VIDEO_SETTINGS.AEC_AGC, 0)        # disable auto
    zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, 5)      # 0-100
    zed.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, 70)          # 0-100
    return zed, image_size_wh, factory_color


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    use_rectified = bool(args.use_rectified_stream)

    # ── Load calibration per role ──────────────────────────────────────────
    calib = {
        "scene": {"intrinsics": None, "extrinsics": None,
                  "resolution": args.scene_resolution},
        "wrist": {"intrinsics": None, "extrinsics": None,
                  "resolution": args.wrist_resolution},
    }

    # We don't know the exact image_size yet (depends on resolution string),
    # so we'll load intrinsics AFTER each camera opens. Just stash paths now.
    intrinsics_paths = {
        "scene": args.scene_intrinsics,
        "wrist": args.wrist_intrinsics,
    }

    # Extrinsics are resolution-independent, load them right away
    if args.scene_extrinsics:
        print(f"Loading SCENE extrinsics from {args.scene_extrinsics}")
        calib["scene"]["extrinsics"] = load_extrinsics(args.scene_extrinsics)
        ex = calib["scene"]["extrinsics"]
        pos = ex["position_xyz"]
        print(f"  scene frame={ex['frame']:>4}  pos = "
              f"[{pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f}] m")
        if ex["frame"] != "base":
            print(f"  WARNING: scene extrinsics has frame='{ex['frame']}', "
                  f"expected 'base'. The scene camera is fixed, so this should "
                  f"be a T_base_camera npz.")

    if args.wrist_extrinsics:
        print(f"Loading WRIST extrinsics from {args.wrist_extrinsics}")
        calib["wrist"]["extrinsics"] = load_extrinsics(args.wrist_extrinsics)
        ex = calib["wrist"]["extrinsics"]
        pos = ex["position_xyz"]
        print(f"  wrist frame={ex['frame']:>4}  pos = "
              f"[{pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f}] m")
        if ex["frame"] == "base":
            print(f"  NOTE: wrist extrinsics is a T_base_camera (eye-to-hand) "
                  f"file. The wrist camera moves with the gripper, so unless "
                  f"the robot is parked, this is probably the wrong file. "
                  f"Most likely you want T_ee_camera.npz from "
                  f"`handeye-solve --eye-in-hand`.")

    # ── Detect connected ZEDs ──────────────────────────────────────────────
    devs = list_zed_devices()
    print(f"\nFound {len(devs)} ZED device(s):")
    for s, m in devs:
        role = CAMERA_ROLES.get(s, "(unknown)")
        print(f"  serial={s}  model={m}  role={role}")

    serials_present = {s for s, _ in devs}
    for expected, role in CAMERA_ROLES.items():
        if expected not in serials_present:
            print(f"WARNING: expected {role} camera "
                  f"(serial {expected}) not connected.")

    # ── Start a pipeline per known camera ──────────────────────────────────
    zeds = {}                # serial -> sl.Camera
    intrinsics_cache = {}    # serial -> {"color": dict, "depth": dict}
    image_holders = {}       # serial -> {"img": sl.Mat, "depth": sl.Mat}

    ordered = sorted(devs, key=lambda d: 0 if CAMERA_ROLES.get(d[0]) == "scene" else 1)
    for serial, model in ordered:
        if serial not in CAMERA_ROLES:
            print(f"  {serial}: unknown serial — skipping")
            continue

        role = CAMERA_ROLES[serial]
        res_str = calib[role]["resolution"]

        try:
            zed, image_size_wh, factory_color = open_zed_for_role(
                serial, res_str, args.fps, args.depth_mode,
                args.depth_min, args.depth_max, label=role)
        except RuntimeError as e:
            print(f"  {serial} ({role}): FAILED to open ({e})")
            continue

        zeds[serial] = zed
        c_w, c_h = image_size_wh
        print(f"  {serial} ({role}, {model}): "
              f"{c_w}x{c_h} @ {args.fps}fps, "
              f"depth={args.depth_mode}, view={'RECTIFIED' if use_rectified else 'UNRECTIFIED'}")

        # Resolve color intrinsics: calibrated if file + size match, else factory
        color_info = None
        if intrinsics_paths[role]:
            print(f"    loading {role} intrinsics from {intrinsics_paths[role]}")
            color_info = load_calibrated_intrinsics(
                intrinsics_paths[role], c_w, c_h, role)

        if color_info is None:
            color_info = factory_color
            print(f"    using FACTORY color intrinsics")
        else:
            print(f"    using CALIBRATED color intrinsics "
                  f"(fx={color_info['fx']:.2f}, fy={color_info['fy']:.2f}, "
                  f"cx={color_info['cx']:.2f}, cy={color_info['cy']:.2f}, "
                  f"dist={len(color_info['dist_coeffs'])} params)")

        # Depth intrinsics on ZED = left-camera intrinsics (depth is aligned).
        # depth_scale: convert raw uint16 mm -> meters by multiplying by 0.001.
        depth_info = dict(factory_color)
        depth_info["source"] = "factory"
        depth_info["depth_scale"] = 0.001  # mm -> meters

        intrinsics_cache[serial] = {"color": color_info, "depth": depth_info}

        # Pre-allocate retrieval buffers
        image_holders[serial] = {
            "img":   sl.Mat(),
            "depth": sl.Mat(),
        }

    if not zeds:
        raise SystemExit("No ZED cameras successfully opened. Exiting.")

    # Pre-resolve enums
    view_enum = sl.VIEW.LEFT if use_rectified else sl.VIEW.LEFT_UNRECTIFIED
    depth_enabled = (args.depth_mode != "NONE")
    runtime = sl.RuntimeParameters()
    runtime.confidence_threshold = 75          # lower = stricter = fewer wrong pixels
    runtime.texture_confidence_threshold = 100 # default 100; lower if too sparse
    runtime.enable_fill_mode = False           # True only if you want holes filled

    # ── agentlace callbacks ────────────────────────────────────────────────
    def observation_callback(keys):
        import cv2
        obs = {}
        for serial, zed in zeds.items():
            # if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
            err = zed.grab(runtime)
            if err != sl.ERROR_CODE.SUCCESS:
                print(f"[{serial}] grab failed: {err}")
                continue
            holders = image_holders[serial]

            # Color (left eye)
            zed.retrieve_image(holders["img"], view_enum)
            color_bgra = holders["img"].get_data()
            # ZED returns BGRA; convert to BGR to match RealSense convention
            if color_bgra.shape[-1] == 4:
                color = cv2.cvtColor(color_bgra, cv2.COLOR_BGRA2BGR)
            else:
                color = color_bgra

            obs[f"cam_{serial}_color"] = color

            # Depth (already aligned to left view, units = mm because we set
            # UNIT.MILLIMETER). Stored as float32 from ZED → cast to uint16
            # to match the RealSense raw depth format downstream code expects.
            if depth_enabled:
                zed.retrieve_measure(holders["depth"], sl.MEASURE.DEPTH)
                depth_f = holders["depth"].get_data()      # H x W float32, mm
                # Replace nan/inf with 0 then clip to uint16 range
                depth_f = np.nan_to_num(depth_f, nan=0.0,
                                        posinf=0.0, neginf=0.0)
                np.clip(depth_f, 0, 65535, out=depth_f)
                depth_u16 = depth_f.astype(np.uint16)
            else:
                # Publish an empty depth map so downstream keys stay consistent
                depth_u16 = np.zeros(color.shape[:2], dtype=np.uint16)

            obs[f"cam_{serial}_depth"] = depth_u16
            # ZED aligns depth to the left eye already, so "aligned" == "depth"
            obs[f"cam_{serial}_depth_aligned"] = depth_u16

            obs[f"cam_{serial}_color_info"] = intrinsics_cache[serial]["color"]
            obs[f"cam_{serial}_depth_info"] = intrinsics_cache[serial]["depth"]

            role = CAMERA_ROLES[serial]
            depth_min_mm, depth_max_mm = DEPTH_RANGE_MM[role]
            obs[f"cam_{serial}_meta"] = {
                "role":         role,
                "depth_min_mm": depth_min_mm,
                "depth_max_mm": depth_max_mm,
                "serial":       serial,
                "model":        "ZED",
            }

            if calib[role]["extrinsics"] is not None:
                obs[f"cam_{serial}_extrinsics"] = calib[role]["extrinsics"]
        return obs

    def action_callback(key, action):
        print(f"Received action '{key}': {action}")
        return {"status": "ok"}

    # ── Build observation keys ─────────────────────────────────────────────
    obs_keys = []
    for s in zeds:
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
        for z in zeds.values():
            z.close()


if __name__ == "__main__":
    main()