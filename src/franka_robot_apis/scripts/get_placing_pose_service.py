#!/usr/bin/env python3
"""
ROS1 Noetic service node: AnyPlace Placement-Pose Pipeline
-----------------------------------------------------------
Service: /robot/perception/get_placement_pose  (robot_api_interfaces/RobotCommand)

Workflow:
  1. Calls /robot/perception/get_object_mask twice (base + target) sequentially.
  2. Builds two point clouds in camera_wrist_depth_optical_frame from each
     (rgb, depth, mask, intrinsics, depth_scale) NPZ payload.
  3. Sends both PCs to the AnyPlace WebSocket server.
  4. Looks up the TF panda_link0 <- camera_wrist_depth_optical_frame.
  5. Transforms every predicted placement pose:
        camera_frame -> target_obj  (T_cam_target, current pose of target)
        camera_frame -> placement   (T_cam_placement = T_relative @ T_cam_target)
        panda_link0   -> placement   (T_base_placement = T_base_cam @ T_cam_placement)

Request (JSON string in .req field):
{
    "base_text_prompt":   "white plate",
    "target_text_prompt": "green cup",
    "init_k_val":         20,        // optional, # AnyPlace pose hypotheses
    "n_refine_iters":     50,        // optional, # diffusion refine steps
    "max_pc_points":      4096       // optional, downsample cap per cloud
}

Response (JSON string in .data field):
{
    "status": "success",
    "message": "...",
    "best_pose_base":   [[...4x4 in panda_link0...]],
    "best_pose_camera": [[...4x4 in camera frame...]],
    "all_poses_base":   [K x 4 x 4],
    "T_base_cam":       [[...4x4...]],
    "T_cam_target":     [[...4x4...]],
    "num_poses":        K,
    "base_detection":   {...labels, confidences, mask_pixels...},
    "target_detection": {...},
    "base_pc_size":     N1,
    "target_pc_size":   N2,
    "anyplace_latency_sec": 1.234
}

Usage:
    rosrun franka_robot_apis get_placement_pose_service.py

    rosservice call /robot/perception/get_placement_pose \
        '{"req": "{\"base_text_prompt\": \"white plate\", \"target_text_prompt\": \"green cup\"}"}'
"""

import json
import asyncio
import base64
import io
import time
import threading
import traceback
from urllib.parse import urlparse

import numpy as np
import aiohttp
from scipy.spatial.transform import Rotation as R

import rospy
import tf2_ros
import tf.transformations as tft
import tf

from robot_api_interfaces.srv import RobotCommand, RobotCommandResponse
from robot_api_interfaces.msg import ResultCode


# ---------------------------------------------------------------------------
# Helpers (module-level — pure numpy, easy to unit test)
# ---------------------------------------------------------------------------
def decode_npz_b64(npz_b64):
    """base64 -> dict[str, ndarray]."""
    raw = base64.b64decode(npz_b64)
    with np.load(io.BytesIO(raw), allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def encode_pc_b64(pc):
    """(N,3) ndarray -> base64-encoded NPZ blob with field 'pc'."""
    buf = io.BytesIO()
    np.savez_compressed(buf, pc=pc.astype(np.float32))
    return base64.b64encode(buf.getvalue()).decode("ascii")


def depth_mask_to_pointcloud(depth, mask, intrinsics, depth_scale,
                              z_min=0.15, z_max=1.5):
    """
    Project masked depth pixels into 3-D points in the camera optical frame.

    Args:
        depth        (H,W) uint16 or float32, raw depth values
        mask         (H,W) uint8 / bool, non-zero where the object is
        intrinsics   (3,3) float, K = [[fx,0,cx],[0,fy,cy],[0,0,1]]
        depth_scale  scalar, multiply raw depth -> metres (e.g. 0.001 for mm)
        z_min/z_max  metres, drop points outside this range

    Returns:
        (N,3) float32 array of XYZ points in camera frame.
    """
    if mask.ndim != 2 or depth.ndim != 2:
        raise ValueError("depth and mask must be 2-D arrays.")
    if mask.shape != depth.shape:
        raise ValueError(f"mask {mask.shape} and depth {depth.shape} shape mismatch.")

    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])

    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    z = depth[ys, xs].astype(np.float32) * float(depth_scale)
    valid = (z > z_min) & (z < z_max)
    xs, ys, z = xs[valid], ys[valid], z[valid]
    if xs.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    X = (xs.astype(np.float32) - cx) * z / fx
    Y = (ys.astype(np.float32) - cy) * z / fy
    Z = z
    return np.stack([X, Y, Z], axis=1).astype(np.float32)


def downsample_pc(pc, max_points, rng=None):
    """Random subsample to at most max_points."""
    if pc.shape[0] <= max_points:
        return pc
    if rng is None:
        rng = np.random.default_rng(0)
    idx = rng.choice(pc.shape[0], size=max_points, replace=False)
    return pc[idx]

def transform_pc(pc, T):
    """Apply 4x4 transform to (N,3) point cloud."""
    if pc.shape[0] == 0:
        return pc
    homo = np.hstack([pc, np.ones((pc.shape[0], 1), dtype=pc.dtype)])
    return (homo @ T.T)[:, :3].astype(np.float32)

def rank_for_drop_in(placements_base, base_pc):
    """Heuristic: prefer placements near the support's XY center, just above its top."""
    base_top_z   = float(np.percentile(base_pc[:, 2], 95))
    base_xy_mean = base_pc[:, :2].mean(axis=0)

    pos = placements_base[:, :3, 3]
    xy_err = np.linalg.norm(pos[:, :2] - base_xy_mean, axis=1)   # closer = better
    z_err  = np.abs(pos[:, 2] - (base_top_z + 0.02))             # 2cm above top
    score  = -(xy_err + 2.0 * z_err)
    return int(np.argmax(score))

def crop_to_workspace(pc, x_range=(0.10, 0.90), y_range=(-0.6, 0.6), z_range=(-0.02, 0.6)):
    """
    Hard workspace bounds in panda_link0. Anything below z=-2cm is floor/junk;
    anything above ~60cm is ceiling/arm. Tune x/y to your actual reachable area.
    """
    if pc.shape[0] == 0:
        return pc
    m = ((pc[:, 0] >= x_range[0]) & (pc[:, 0] <= x_range[1]) &
         (pc[:, 1] >= y_range[0]) & (pc[:, 1] <= y_range[1]) &
         (pc[:, 2] >= z_range[0]) & (pc[:, 2] <= z_range[1]))
    return pc[m]


def remove_statistical_outliers(pc, nb_neighbors=20, std_ratio=2.0):
    """Removes scattered outliers (depth-noise flyers) using open3d's SOR."""
    if pc.shape[0] < nb_neighbors + 1:
        return pc
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc.astype(np.float64))
    pcd_clean, _ = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio
    )
    return np.asarray(pcd_clean.points, dtype=np.float32)


def keep_top_slab(pc, slab_thickness=0.04):
    """
    For flat-top supports (plate, table, shelf): keep only points within
    `slab_thickness` of the max z. Forces the model to see the support as a
    thin surface rather than a deep volume.
    Skip this for containers (bowl, tube rack, vial holder) — you want depth there.
    """
    if pc.shape[0] == 0:
        return pc
    z_max = float(pc[:, 2].max())
    return pc[pc[:, 2] >= (z_max - slab_thickness)]

def transform_stamped_to_matrix(ts):
    """geometry_msgs/TransformStamped -> 4x4 numpy."""
    t = ts.transform.translation
    q = ts.transform.rotation
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    T[:3, 3]  = [t.x, t.y, t.z]
    return T


def centroid_pose(pc):
    """
    Build a pose whose translation is the PC centroid and rotation = identity.
    This is the 'current pose' of the target in the camera frame, used as the
    composition anchor for the relative transform predicted by AnyPlace.
    """
    T = np.eye(4, dtype=np.float64)
    if pc.shape[0] > 0:
        T[:3, 3] = pc.mean(axis=0)
    return T


# ---------------------------------------------------------------------------
# Service node
# ---------------------------------------------------------------------------
class GetPlacementPoseNode:

    def __init__(self):
        # ------------------------------------------------------------------ #
        #  Parameters                                                          #
        # ------------------------------------------------------------------ #
        # Same SAM2 server IP as the get_object_mask node — we just swap the
        # port. Keep both overridable for flexibility.
        self.sam2_url = rospy.get_param(
            "~sam2_url", "ws://10.158.54.164:8766/get_object_mask"
        )
        sam2_host = urlparse(self.sam2_url).hostname or "10.158.54.164"
        self.anyplace_url = rospy.get_param(
            "~anyplace_url",
            f"ws://{sam2_host}:8768/predict_placement_pose",
        )

        # ROS service we call to get masks (the existing one you already run)
        self.mask_service_name = rospy.get_param(
            "~mask_service_name", "/robot/perception/get_object_mask"
        )

        # TF frames
        self.base_frame   = rospy.get_param("~base_frame",   "panda_link0")
        self.camera_frame = rospy.get_param("~camera_frame", "zed_scene_left_optical_frame")

        # Defaults
        self.default_init_k       = int(rospy.get_param("~init_k_val",      20))
        self.default_refine_iters = int(rospy.get_param("~n_refine_iters",  50))
        self.default_max_pc_pts   = int(rospy.get_param("~max_pc_points",   4096))

        # Timeouts
        self.mask_call_timeout_sec = float(rospy.get_param("~mask_call_timeout_sec", 30.0))
        self.anyplace_timeout_sec  = float(rospy.get_param("~anyplace_timeout_sec",  90.0))
        self.tf_lookup_timeout_sec = float(rospy.get_param("~tf_lookup_timeout_sec", 3.0))
        self.tf_timeout_sec= float(rospy.get_param("~tf_timeout_sec", 2.0))

        # ------------------------------------------------------------------ #
        #  TF                                                                  #
        # ------------------------------------------------------------------ #
        self.tf_buffer   = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self._tf_listener = tf.TransformListener()
        self._tf_broadcaster = tf.TransformBroadcaster()


        # Serialize service handler invocations — AnyPlace server is single-GPU,
        # and rospy.Service spawns one thread per call. Two simultaneous calls
        # would also try to issue overlapping mask requests.
        self._handler_lock = threading.Lock()

        # ------------------------------------------------------------------ #
        #  Wait for the upstream mask service                                  #
        # ------------------------------------------------------------------ #
        rospy.loginfo(f"Waiting for upstream mask service: {self.mask_service_name}")
        try:
            rospy.wait_for_service(self.mask_service_name, timeout=10.0)
            rospy.loginfo("Upstream mask service is up.")
        except rospy.ROSException:
            rospy.logwarn(
                f"Timed out waiting for {self.mask_service_name}. "
                "Will retry on each request."
            )
        self._mask_proxy = rospy.ServiceProxy(self.mask_service_name, RobotCommand)

        # ------------------------------------------------------------------ #
        #  Service                                                             #
        # ------------------------------------------------------------------ #
        self._service = rospy.Service(
            "/robot/perception/get_placement_pose",
            RobotCommand,
            self._handle_request,
        )

        rospy.loginfo(
            "\nGetPlacementPoseNode (ROS1) ready.\n"
            f"  Service       : /robot/perception/get_placement_pose\n"
            f"  Mask service  : {self.mask_service_name}\n"
            f"  AnyPlace WS   : {self.anyplace_url}\n"
            f"  Base frame    : {self.base_frame}\n"
            f"  Camera frame  : {self.camera_frame}"
        )

    # ------------------------------------------------------------------ #
    #  Service handler                                                     #
    # ------------------------------------------------------------------ #
    def _handle_request(self, request):
        rospy.loginfo(f"get_placement_pose request: {request.req}")
        response = RobotCommandResponse()

        with self._handler_lock:
            # --- 1. Parse request ---------------------------------------
            try:
                req = json.loads(request.req)
                base_prompt   = (req.get("base_text_prompt")   or "").strip()
                target_prompt = (req.get("target_text_prompt") or "").strip()
                if not base_prompt or not target_prompt:
                    return self._fail(
                        response,
                        "Both 'base_text_prompt' and 'target_text_prompt' are required."
                    )
                init_k         = int(req.get("init_k_val",     self.default_init_k))
                refine_iters   = int(req.get("n_refine_iters", self.default_refine_iters))
                max_pc_points  = int(req.get("max_pc_points",  self.default_max_pc_pts))
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                return self._fail(response, f"Bad request: {e}")

            # --- 2. Get mask + frame for BASE object --------------------
            rospy.loginfo(f"[1/4] Requesting mask for BASE object: '{base_prompt}'")
            base_npz, base_meta, err = self._call_mask_service(base_prompt)
            if err:
                return self._fail(response, f"Base mask call failed: {err}")

            # --- 3. Get mask + frame for TARGET object ------------------
            rospy.loginfo(f"[2/4] Requesting mask for TARGET object: '{target_prompt}'")
            target_npz, target_meta, err = self._call_mask_service(target_prompt)
            if err:
                return self._fail(response, f"Target mask call failed: {err}")

            # --- 4. Build point clouds in CAMERA frame ------------------
            try:
                base_pc_cam   = self._build_pc(base_npz)
                target_pc_cam = self._build_pc(target_npz)
            except Exception as e:
                return self._fail(response, f"Point-cloud construction failed: {e}")

            rospy.loginfo(
                f"Built PCs (camera frame): "
                f"base={base_pc_cam.shape[0]}, target={target_pc_cam.shape[0]}"
            )
            if base_pc_cam.shape[0] < 50 or target_pc_cam.shape[0] < 50:
                return self._fail(response,
                    f"PCs too small (base={base_pc_cam.shape[0]}, "
                    f"target={target_pc_cam.shape[0]}).")

            # --- 5. Lookup TF and transform PCs into base frame ---------
            try:
                T_base_cam = self._lookup_T_base_cam()
            except Exception as e:
                return self._fail(response, f"TF lookup failed: {e}")
            rospy.loginfo(f"T_base_cam translation: {np.round(T_base_cam[:3, 3], 4).tolist()}")

            base_pc   = transform_pc(base_pc_cam,   T_base_cam)
            target_pc = transform_pc(target_pc_cam, T_base_cam)

            # ---- NEW: clean PCs in base frame ----
            n0_base, n0_tgt = base_pc.shape[0], target_pc.shape[0]

            base_pc   = crop_to_workspace(base_pc)
            target_pc = crop_to_workspace(target_pc)

            base_pc   = remove_statistical_outliers(base_pc)
            target_pc = remove_statistical_outliers(target_pc)

            # Optional, task-dependent: only for flat supports.
            # Pass `flat_support=True` in the request to enable.
            if bool(req.get("flat_support", False)):
                base_pc = keep_top_slab(base_pc, slab_thickness=0.04)

            rospy.loginfo(
                f"Cleaned PCs: base {n0_base}->{base_pc.shape[0]} "
                f"(z=[{base_pc[:,2].min():.3f},{base_pc[:,2].max():.3f}]), "
                f"target {n0_tgt}->{target_pc.shape[0]} "
                f"(z=[{target_pc[:,2].min():.3f},{target_pc[:,2].max():.3f}])"
            )
            if base_pc.shape[0] < 50 or target_pc.shape[0] < 50:
                return self._fail(response,
                    f"PCs empty after cleaning "
                    f"(base={base_pc.shape[0]}, target={target_pc.shape[0]}). "
                    "Workspace bounds may be too tight or the mask is bad.")

            base_pc   = downsample_pc(base_pc,   max_pc_points)
            target_pc = downsample_pc(target_pc, max_pc_points)

            try:
                import open3d as o3d
                o3d.io.write_point_cloud("/tmp/base_pc_raw.ply", o3d.geometry.PointCloud(o3d.utility.Vector3dVector(base_pc.astype(np.float64))))
                o3d.io.write_point_cloud("/tmp/target_pc_raw.ply", o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target_pc.astype(np.float64))))
            except Exception as e:
                rospy.logwarn(f"Could not save debug PCs: {e}")

            # --- 6. Call AnyPlace WebSocket -----------------------------
            rospy.loginfo(f"[3/4] Calling AnyPlace | init_k={init_k}, refine_iters={refine_iters}")
            t0 = time.time()
            try:
                ap_result = asyncio.run(
                    self._call_anyplace(base_pc, target_pc, init_k, refine_iters)
                )
            except aiohttp.ClientConnectorError:
                return self._fail(response, f"Could not connect to AnyPlace at {self.anyplace_url}")
            except asyncio.TimeoutError:
                return self._fail(response, f"AnyPlace timed out after {self.anyplace_timeout_sec}s")
            except Exception as e:
                return self._fail(response, f"AnyPlace error: {e}\n{traceback.format_exc()}")
            anyplace_latency = time.time() - t0
            rospy.loginfo(f"AnyPlace returned in {anyplace_latency:.2f}s")

            if ap_result.get("status") != "success":
                return self._fail(response,
                    f"AnyPlace returned non-success: {ap_result.get('message', ap_result)}")

            try:
                poses_base = decode_npz_b64(ap_result["poses_npz"])["poses"]   # (K,4,4) in base frame
            except Exception as e:
                return self._fail(response, f"Could not decode AnyPlace poses_npz: {e}")
            poses_base = np.asarray(poses_base, dtype=np.float64)
            if poses_base.ndim != 3 or poses_base.shape[1:] != (4, 4):
                return self._fail(response, f"AnyPlace poses unexpected shape: {poses_base.shape}")

            # --- 7. Compose placements directly in base frame -----------
            T_base_target  = centroid_pose(target_pc)            # in panda_link0 now
            placements_base = np.einsum("kij,jl->kil", poses_base, T_base_target)

            best_idx       = 0
            # best_idx = rank_for_drop_in(placements_base, base_pc)
            best_pose_base = placements_base[best_idx]

            rospy.loginfo("[4/4] Composed placement pose in panda_link0:")
            rospy.loginfo(f"  translation: {np.round(best_pose_base[:3, 3], 4).tolist()}")
            rospy.loginfo(f"  rotation (quat xyzw): "
                          f"{np.round(R.from_matrix(best_pose_base[:3,:3]).as_quat(), 4).tolist()}")
            
            # --- 8. Build response (compact: best pose only) ------------
            best_quat_xyzw = R.from_matrix(best_pose_base[:3, :3]).as_quat().tolist()
            # best_translation = best_pose_base[:3, 3].tolist()

            # Score for the best pose. AnyPlace returns ranked poses with
            # return_top=True, so index 0 is the highest-ranked candidate.
            # If the server ever exposes per-pose scores, surface them here.
            # best_score = None
            # if isinstance(ap_result.get("scores"), list) and ap_result["scores"]:
            #     best_score = float(ap_result["scores"][best_idx])

            # --- 9. Get position and orientation in base frame, and return response -----------
            def _quaternion_multiply(q_a, q_b):
                """
                Hamilton product of two quaternion dicts {x, y, z, w}.

                Args:
                    q_a (dict): first quaternion
                    q_b (dict): second quaternion

                Returns:
                    dict: composed quaternion
                """
                ax, ay, az, aw = q_a["x"], q_a["y"], q_a["z"], q_a["w"]
                bx, by, bz, bw = q_b["x"], q_b["y"], q_b["z"], q_b["w"]
                return {
                    "x": float(aw * bx + ax * bw + ay * bz - az * by),
                    "y": float(aw * by - ax * bz + ay * bw + az * bx),
                    "z": float(aw * bz + ax * by - ay * bx + az * bw),
                    "w": float(aw * bw - ax * bx - ay * by - az * bz),
                }
            
            q_base = {
                "x": best_quat_xyzw[0],
                "y": best_quat_xyzw[1],
                "z": best_quat_xyzw[2],
                "w": best_quat_xyzw[3]
            }
            t_base = [float(x) for x in best_pose_base[:3, 3]]

            rospy.loginfo(
                f"Original pose in '{self.base_frame}' | "
                f"xyz=({t_base[0]:.4f}, {t_base[1]:.4f}, {t_base[2]:.4f})  "
                f"quat=({q_base['x']:.4f}, {q_base['y']:.4f}, "
                f"{q_base['z']:.4f}, {q_base['w']:.4f})"
            )

            # target object centroid height
            target_obj_height = float(target_pc[:, 2].max() - target_pc[:, 2].min())
            rospy.loginfo(f"Target object height: {target_obj_height:.4f}m")
            # base object height
            base_obj_height = float(base_pc[:, 2].max() - base_pc[:, 2].min())
            rospy.loginfo(f"Base object height: {base_obj_height:.4f}m")

            # if t_base[2] < 0.0:
            #     t_base[2] = 0.0
            t_base[2] = target_obj_height + base_obj_height*0.5 + 0.02
            rospy.loginfo(f"Adjusted Z for safety: {t_base[2]:.4f}m")
            
            # Post-multiply by 180° around X to fix AnyGrasp EE convention
            q_180x = {"x": 1.0, "y": 0.0, "z": 0.0, "w": 0.0}
            q_base = _quaternion_multiply(q_base, q_180x)
            q_180y = {"x": 0.0, "y": -1.0, "z": 0.0, "w": 0.0}
            q_base = _quaternion_multiply(q_base, q_180y)
            # q_90yz = {"x": -0.7071068, "y": 0.0, "z": 0.0, "w": 0.7071068}
            # q_base = _quaternion_multiply(q_base, q_90yz)

            # # Apply a backward shift of 0.18m along the grasp pose's local X
            # try:
            #     shift_m = 0.18
            #     # rotation matrix from base-frame quaternion
            #     Q = [q_base['x'], q_base['y'], q_base['z'], q_base['w']]
            #     BR = tft.quaternion_matrix(Q)[0:3, 0:3]
            #     # local backward along +Z -> negative Z in local coordinates
            #     shift_global = BR.dot(np.array([0.0, 0.0, -shift_m]))
            #     t_shift = [
            #         float(t_base[0] + shift_global[0]),
            #         float(t_base[1] + shift_global[1]),
            #         float(t_base[2] + shift_global[2]),
            #     ]

            #     # Broadcast the shifted pose as frame 'shifted_placement' (parent = base_frame)
            #     now = rospy.Time.now()
            #     self._tf_broadcaster.sendTransform(
            #         (t_shift[0], t_shift[1], t_shift[2]),
            #         (q_base['x'], q_base['y'], q_base['z'], q_base['w']),
            #         now,
            #         "shifted_placement",
            #         self.base_frame,
            #     )

            #     # small pause to allow TF to propagate, then lookup shifted pose
            #     rospy.sleep(0.05)
            #     self._tf_listener.waitForTransform(
            #         self.base_frame,
            #         "shifted_placement",
            #         rospy.Time(0),
            #         rospy.Duration(self.tf_timeout_sec),
            #     )
            #     trans_s, rot_s = self._tf_listener.lookupTransform(
            #         self.base_frame,
            #         "shifted_placement",
            #         rospy.Time(0),
            #     )

            #     t_base = [float(trans_s[0]), float(trans_s[1]), float(trans_s[2])]
            #     q_base = {"x": float(rot_s[0]), "y": float(rot_s[1]),
            #               "z": float(rot_s[2]), "w": float(rot_s[3])}

            #     rospy.loginfo(
            #         f"Shifted pose in '{self.base_frame}' | "
            #         f"xyz=({t_base[0]:.4f}, {t_base[1]:.4f}, {t_base[2]:.4f})  "
            #         f"quat=({q_base['x']:.4f}, {q_base['y']:.4f}, "
            #         f"{q_base['z']:.4f}, {q_base['w']:.4f})"
            #     )

            # except Exception as e:
            #     rospy.logwarn(f"Failed to apply/lookup shifted placement TF: {e}")
            #     # keep original t_base/q_base if shifting fails


            rospy.loginfo(
                f"Shifted pose in '{self.base_frame}' | "
                f"xyz=({t_base[0]:.4f}, {t_base[1]:.4f}, {t_base[2]:.4f})  "
                f"quat=({q_base['x']:.4f}, {q_base['y']:.4f}, "
                f"{q_base['z']:.4f}, {q_base['w']:.4f})"
            )

            payload = {
                "status":  "success",
                "message": (
                    f"Placement pose predicted for '{target_prompt}' "
                    f"onto '{base_prompt}'."
                ),

                "placement_pose_base": {
                    "position": {
                        "x": t_base[0],
                        "y": t_base[1],
                        "z": t_base[2],
                    },
                    "orientation": {
                        "x": q_base["x"],
                        "y": q_base["y"],
                        "z": q_base["z"],
                        "w": q_base["w"],
                    },
                },

                # The thing you actually use downstream
                # "best_pose_base":     best_pose_base.tolist(),     # 4x4 in panda_link0
                # "best_pose_position": best_translation,            # [x, y, z]
                # "best_pose_quat_xyzw": best_quat_xyzw,             # [qx, qy, qz, qw]
                # "frame_id":           self.base_frame,             # "panda_link0"

                # # Metadata / confidence
                # "best_pose_index":     int(best_idx),
                # "best_pose_score":     best_score,
                # "num_pose_candidates": int(poses_cam.shape[0]),
                # "anyplace_latency_sec": round(anyplace_latency, 3),

                # "base_detection": {
                #     "prompt":      base_prompt,
                #     "labels":      base_meta.get("detected_labels", []),
                #     "confidences": base_meta.get("confidences", []),
                #     "mask_pixels": base_meta.get("mask_pixels"),
                # },
                # "target_detection": {
                #     "prompt":      target_prompt,
                #     "labels":      target_meta.get("detected_labels", []),
                #     "confidences": target_meta.get("confidences", []),
                #     "mask_pixels": target_meta.get("mask_pixels"),
                # },
                # "base_pc_size":   int(base_pc.shape[0]),
                # "target_pc_size": int(target_pc.shape[0]),
            }

            response.result_code.result_code = ResultCode.SUCCESS
            response.result_code.message     = "Placement-pose prediction succeeded."
            response.data                    = json.dumps(payload)
            return response

    # ------------------------------------------------------------------ #
    #  Upstream mask service call                                          #
    # ------------------------------------------------------------------ #
    def _call_mask_service(self, text_prompt):
        """
        Returns:
            (npz_dict, meta_dict, error_string)
            On success -> (dict of arrays, dict of metadata, None)
            On failure -> (None, None, "...")
        """
        try:
            req = RobotCommand._request_class()
            req.req = json.dumps({"text_prompt": text_prompt})

            # rospy.ServiceProxy is blocking; rely on the upstream node's own
            # timeouts. We add a soft wall-clock check via threading watchdog
            # only if needed — for now just call straight through.
            resp = self._mask_proxy(req)
            if resp.result_code.result_code != ResultCode.SUCCESS:
                return None, None, (
                    f"Upstream mask service failure: "
                    f"{resp.result_code.message}"
                )

            payload = json.loads(resp.data)
            if payload.get("status") != "success":
                return None, None, (
                    f"Mask service status != success: "
                    f"{payload.get('message', payload)}"
                )

            npz_b64 = payload.get("npz_base64")
            if payload.get("data_encoding") != "npz_base64" or not npz_b64:
                return None, None, "Mask response missing 'npz_base64'."

            arrays = decode_npz_b64(npz_b64)
            meta = {
                "detected_labels": payload.get("detected_labels", []),
                "confidences":     payload.get("confidences", []),
                "mask_pixels":     payload.get("mask_pixels"),
                "num_detections":  payload.get("num_detections"),
            }
            rospy.loginfo(
                f"  -> mask received | labels={meta['detected_labels']} "
                f"| conf={meta['confidences']} "
                f"| mask_pixels={meta['mask_pixels']}"
            )
            return arrays, meta, None
        except rospy.ServiceException as e:
            return None, None, f"ServiceException: {e}"
        except json.JSONDecodeError as e:
            return None, None, f"Bad JSON from mask service: {e}"
        except Exception as e:
            return None, None, f"Unexpected error: {e}\n{traceback.format_exc()}"

    # ------------------------------------------------------------------ #
    #  Build point cloud from mask service NPZ                             #
    # ------------------------------------------------------------------ #
    def _build_pc(self, npz):
        for k in ("depth", "mask", "intrinsics"):
            if k not in npz:
                raise KeyError(f"Mask NPZ missing required field '{k}'")

        depth      = npz["depth"]
        mask       = npz["mask"]
        intrinsics = np.asarray(npz["intrinsics"], dtype=np.float64)
        # depth_scale is optional but expected; default to 0.001 (mm -> m)
        depth_scale = float(np.asarray(npz["depth_scale"]).item()) \
            if "depth_scale" in npz else 0.001

        return depth_mask_to_pointcloud(depth, mask, intrinsics, depth_scale)

    # ------------------------------------------------------------------ #
    #  TF lookup                                                           #
    # ------------------------------------------------------------------ #
    def _lookup_T_base_cam(self):
        """Return 4x4 transform that maps points in camera_frame -> base_frame."""
        ts = self.tf_buffer.lookup_transform(
            self.base_frame,
            self.camera_frame,
            rospy.Time(0),
            rospy.Duration(self.tf_lookup_timeout_sec),
        )
        return transform_stamped_to_matrix(ts)

    # ------------------------------------------------------------------ #
    #  AnyPlace WebSocket call                                             #
    # ------------------------------------------------------------------ #
    async def _call_anyplace(self, base_pc, target_pc, init_k, refine_iters):
        payload = {
            "base_pc_npz":    encode_pc_b64(base_pc),
            "target_pc_npz":  encode_pc_b64(target_pc),
            "init_k_val":     int(init_k),
            "n_refine_iters": int(refine_iters),
            "return_top":     True,
            "with_coll":      False,
        }
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(
                self.anyplace_url,
                max_msg_size=200 * 1024 * 1024,
                heartbeat=20.0,
            ) as ws:
                await ws.send_str(json.dumps(payload))
                msg = await asyncio.wait_for(
                    ws.receive(), timeout=self.anyplace_timeout_sec
                )
                if msg.type == aiohttp.WSMsgType.TEXT:
                    return json.loads(msg.data)
                if msg.type == aiohttp.WSMsgType.ERROR:
                    raise RuntimeError(f"WebSocket error frame from {self.anyplace_url}")
                if msg.type in (aiohttp.WSMsgType.CLOSE,
                                aiohttp.WSMsgType.CLOSING,
                                aiohttp.WSMsgType.CLOSED):
                    raise RuntimeError(f"WebSocket closed unexpectedly by {self.anyplace_url}")
                raise RuntimeError(f"Unexpected WS message type {msg.type}")

    # ------------------------------------------------------------------ #
    #  Failure helper                                                      #
    # ------------------------------------------------------------------ #
    def _fail(self, response, msg):
        rospy.logerr(f"get_placement_pose service error: {msg}")
        response.result_code.result_code = ResultCode.FAILURE
        response.result_code.message     = msg
        response.data = json.dumps({"status": "error", "message": msg})
        return response

    def spin(self):
        rospy.spin()


def main():
    rospy.init_node("get_placement_pose_service", anonymous=False)
    try:
        rospy.loginfo("Creating GetPlacementPoseNode ...")
        node = GetPlacementPoseNode()
        rospy.loginfo("GetPlacementPoseNode spinning ...")
        node.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS interrupt — shutting down GetPlacementPoseNode.")
    except Exception as e:
        rospy.logerr(f"Fatal error: {e}\n{traceback.format_exc()}")
    finally:
        rospy.loginfo("GetPlacementPoseNode shutdown complete.")


if __name__ == "__main__":
    main()