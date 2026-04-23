import argparse
import time
import numpy as np
import pyrealsense2 as rs
from agentlace.action import ActionServer, ActionConfig

# ── Detect all connected RealSense cameras ──────────────────────────────────
ctx = rs.context()
serials = [d.get_info(rs.camera_info.serial_number) for d in ctx.devices]
print(f"Found {len(serials)} camera(s): {serials}")

# ── Start a pipeline per camera + cache intrinsics ───────────────────────────
pipelines = {}
intrinsics_cache = {}  # serial -> dict with fx, fy, cx, cy, coeffs, depth_scale

for serial in serials:
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(cfg)
    pipelines[serial] = pipeline

    # Color intrinsics
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    ci = color_stream.get_intrinsics()

    # Depth intrinsics
    depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    di = depth_stream.get_intrinsics()

    # Depth scale (meters per unit)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    intrinsics_cache[serial] = {
        "color": {
            "width": ci.width, "height": ci.height,
            "fx": ci.fx, "fy": ci.fy,
            "cx": ci.ppx, "cy": ci.ppy,
            "coeffs": list(ci.coeffs),   # [k1, k2, p1, p2, k3]
            "model": str(ci.model),
        },
        "depth": {
            "width": di.width, "height": di.height,
            "fx": di.fx, "fy": di.fy,
            "cx": di.ppx, "cy": di.ppy,
            "coeffs": list(di.coeffs),
            "model": str(di.model),
            "depth_scale": depth_scale,
        },
    }
    print(f"  Started pipeline for {serial}, depth_scale={depth_scale:.6f}")

# ── agentlace callbacks ───────────────────────────────────────────────────────
def observation_callback(keys):
    obs = {}
    for serial, pipeline in pipelines.items():
        frames = pipeline.wait_for_frames()
        color = np.asanyarray(frames.get_color_frame().get_data())
        depth = np.asanyarray(frames.get_depth_frame().get_data())
        obs[f"cam_{serial}_color"] = color
        obs[f"cam_{serial}_depth"] = depth
        # Send intrinsics on every frame (cheap — small dicts, already cached)
        obs[f"cam_{serial}_color_info"] = intrinsics_cache[serial]["color"]
        obs[f"cam_{serial}_depth_info"] = intrinsics_cache[serial]["depth"]
    return obs

def action_callback(key, action):
    print(f"Received action '{key}': {action}")
    return {"status": "ok"}

# ── Build observation + action key lists ──────────────────────────────────────
obs_keys = []
for s in serials:
    obs_keys += [
        f"cam_{s}_color", f"cam_{s}_depth",
        f"cam_{s}_color_info", f"cam_{s}_depth_info",
    ]
act_keys = ["command"]
port_number = 6379

print(f"Observation keys: {obs_keys}")
print(f"Server starting on port {port_number} …")

config = ActionConfig(port_number=port_number, action_keys=act_keys, observation_keys=obs_keys)
server = ActionServer(config, observation_callback, action_callback)
server.start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Shutting down")
    for p in pipelines.values():
        p.stop()




# import argparse
# import numpy as np
# import pyrealsense2 as rs
# from agentlace.action import ActionServer, ActionConfig

# # ── Detect all connected RealSense cameras ──────────────────────────────────
# ctx = rs.context()
# serials = [d.get_info(rs.camera_info.serial_number) for d in ctx.devices]
# print(f"Found {len(serials)} camera(s): {serials}")

# # ── Start a pipeline per camera ──────────────────────────────────────────────
# pipelines = {}
# for serial in serials:
#     pipeline = rs.pipeline()
#     cfg = rs.config()
#     cfg.enable_device(serial)
#     cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
#     cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
#     pipeline.start(cfg)
#     pipelines[serial] = pipeline
#     print(f"  Started pipeline for {serial}")

# # ── agentlace callbacks ───────────────────────────────────────────────────────
# def observation_callback(keys):
#     obs = {}
#     for serial, pipeline in pipelines.items():
#         frames = pipeline.wait_for_frames()
#         color = np.asanyarray(frames.get_color_frame().get_data())
#         depth = np.asanyarray(frames.get_depth_frame().get_data())
#         obs[f"cam_{serial}_color"] = color
#         obs[f"cam_{serial}_depth"] = depth
#     return obs

# def action_callback(key, action):
#     print(f"Received action '{key}': {action}")
#     return {"status": "ok"}

# # ── Build observation + action key lists ──────────────────────────────────────
# obs_keys = [f"cam_{s}_color" for s in serials] + [f"cam_{s}_depth" for s in serials]
# act_keys = ["command"]
# port_number = 6379

# print(f"Observation keys: {obs_keys}")
# print(f"Action keys: {act_keys}")
# print(f"Server Port {port_number} …")

# config = ActionConfig(port_number=port_number, action_keys=act_keys, observation_keys=obs_keys)
# server = ActionServer(config, observation_callback, action_callback)

# print(f"Server starting on port {port_number} …")
# server.start()

# # Keep alive
# import time
# try:
#     while True:
#         time.sleep(1)
# except KeyboardInterrupt:
#     print("Shutting down")
#     for p in pipelines.values():
#         p.stop()