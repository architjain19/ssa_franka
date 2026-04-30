#!/usr/bin/env python3
"""
compute_camera_tf_link8.py  (ROS 1 Noetic)
==========================================

Loads a saved T_base_camera (from `extrinsics-from-board` or `handeye-solve`)
and the current TF for `panda_link0 -> panda_link8`, then computes:

    T_link8_camera = inverse(T_base_link8) @ T_base_camera

and prints a ready-to-paste launch file XML + CLI command for
tf2_ros's static_transform_publisher.

PHYSICAL ASSUMPTION (READ THIS)
-------------------------------
This is only valid as a STATIC transform if the camera is rigidly attached
to panda_link8 (eye-in-hand). For a scene camera fixed in the world
(eye-to-hand -- which is what `handeye-solve` and `extrinsics-from-board`
were originally written for), the link8->camera transform changes whenever
the arm moves and should NOT be published as a static transform. Use
panda_link0 as the parent in that case.

If you DO have an eye-in-hand setup: the robot must be at the same pose
that was used when T_base_camera was computed, otherwise the answer is
wrong by however much link8 has moved between then and now.

USAGE
-----
  source /opt/ros/noetic/setup.bash
  source ~/<your_ws>/devel/setup.sh
  # franka_ros must already be running so /tf is populated.

  python3 compute_camera_tf_link8.py \
      --extrinsics /home/daphne/camera_calibration/handeye_result/T_base_camera.npz

  # To verify by also publishing the static TF live:
  python3 compute_camera_tf_link8.py --extrinsics ... --publish

Requires: rospy, tf2_ros, numpy, scipy, geometry_msgs.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def transform_to_matrix(t):
    """geometry_msgs/Transform -> 4x4 homogeneous numpy matrix."""
    from scipy.spatial.transform import Rotation as R
    T = np.eye(4)
    T[0, 3] = t.translation.x
    T[1, 3] = t.translation.y
    T[2, 3] = t.translation.z
    q = [t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w]
    T[:3, :3] = R.from_quat(q).as_matrix()
    return T


def average_se3(samples):
    """Average a list of 4x4 SE(3) matrices. Translations are arithmetically
    averaged; rotations are averaged then re-orthonormalized via SVD (the
    closest rotation to the mean of the rotation matrices)."""
    if len(samples) == 1:
        return samples[0]
    stack = np.stack(samples)
    T_avg = np.mean(stack, axis=0)
    U, _, Vt = np.linalg.svd(T_avg[:3, :3])
    R_avg = U @ Vt
    if np.linalg.det(R_avg) < 0:
        R_avg = U @ np.diag([1.0, 1.0, -1.0]) @ Vt
    T_avg[:3, :3] = R_avg
    T_avg[3, :] = [0.0, 0.0, 0.0, 1.0]
    return T_avg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extrinsics",   required=True,
                    help="Path to T_base_camera.npz from "
                         "extrinsics-from-board or handeye-solve")
    ap.add_argument("--parent-frame", default="panda_link0",
                    help="Robot base frame (default: panda_link0)")
    ap.add_argument("--link-frame",   default="panda_link8",
                    help="Frame the camera is rigidly attached to "
                         "(default: panda_link8). This is the PARENT of "
                         "the published static TF.")
    ap.add_argument("--camera-frame", default="camera_color_optical_frame",
                    help="Camera frame name (the CHILD of the static TF). "
                         "Default matches realsense2_camera ROS driver.")
    ap.add_argument("--publish", action="store_true",
                    help="Also broadcast the static TF live (for verification "
                         "with rviz / `ros2 run tf2_tools view_frames`). "
                         "Ctrl-C to exit.")
    ap.add_argument("--average", type=int, default=1,
                    help="Average TF over N samples to reduce noise "
                         "(default: 1; robot must be stationary)")
    ap.add_argument("--timeout", type=float, default=5.0,
                    help="TF lookup timeout in seconds (default: 5)")
    ap.add_argument("--save", action="store_true",
                    help="Save T_link8_camera.npz next to the input file")
    args = ap.parse_args()

    # ---- Load T_base_camera ----
    extr_path = Path(args.extrinsics)
    if not extr_path.exists():
        sys.exit(f"Extrinsics file not found: {extr_path}")
    data = np.load(extr_path, allow_pickle=True)
    if "T_base_camera" not in data.files:
        sys.exit(f"'{extr_path}' has no 'T_base_camera' key. "
                 f"Found keys: {list(data.files)}")
    T_base_camera = np.asarray(data["T_base_camera"], dtype=np.float64)
    if T_base_camera.shape != (4, 4):
        sys.exit(f"T_base_camera has shape {T_base_camera.shape}; "
                 f"expected (4, 4)")
    print(f"[load] T_base_camera from {extr_path}")
    if "method" in data.files:
        try:
            method = str(data["method"])
        except Exception:
            method = "?"
        print(f"[load]   method: {method}")
    print(f"[load]   T_base_camera =\n{T_base_camera}")

    # ---- ROS init + TF listener ----
    rospy.init_node("compute_camera_tf_link8", anonymous=True,
                    disable_signals=True)
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)

    # ---- Look up T_base_link8 ----
    print(f"\n[tf] looking up {args.parent_frame} -> {args.link_frame}"
          f"  (waiting up to {args.timeout:.1f} s)")
    samples = []
    rate = rospy.Rate(20)  # 20 Hz between samples
    sample_deadline = rospy.Time.now() + rospy.Duration(
        args.timeout + max(0, args.average - 1) * 0.05)

    # First lookup waits for the TF tree to populate.
    try:
        trans = tf_buffer.lookup_transform(
            args.parent_frame,
            args.link_frame,
            rospy.Time(0),
            rospy.Duration(args.timeout))
    except (tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException) as e:
        sys.exit(
            f"[tf] lookup failed: {e}\n"
            f"  Check that /tf is publishing and that {args.parent_frame} "
            f"and {args.link_frame} are connected:\n"
            f"      rostopic hz /tf\n"
            f"      rosrun tf2_tools view_frames.py\n"
            f"      rosrun tf tf_echo {args.parent_frame} {args.link_frame}")
    samples.append(transform_to_matrix(trans.transform))
    if args.average > 1:
        print(f"[tf]   sample 1/{args.average}")

    # Additional samples (averaging) — short timeout, just need a fresh one.
    while len(samples) < args.average and rospy.Time.now() < sample_deadline:
        rate.sleep()
        try:
            trans = tf_buffer.lookup_transform(
                args.parent_frame, args.link_frame,
                rospy.Time(0), rospy.Duration(0.5))
            samples.append(transform_to_matrix(trans.transform))
            print(f"[tf]   sample {len(samples)}/{args.average}")
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            continue

    if len(samples) < args.average:
        print(f"[tf] WARNING: only collected {len(samples)}/{args.average} "
              f"samples within timeout; proceeding with what we have.")

    T_base_link8 = average_se3(samples)
    print(f"\n[tf] T_base_link8 (current pose, averaged over {len(samples)} "
          f"sample{'s' if len(samples) > 1 else ''}) =")
    print(T_base_link8)

    # ---- Compute T_link8_camera ----
    T_link8_base = np.linalg.inv(T_base_link8)
    T_link8_camera = T_link8_base @ T_base_camera

    pos = T_link8_camera[:3, 3]
    from scipy.spatial.transform import Rotation as R
    rot = R.from_matrix(T_link8_camera[:3, :3])
    q = rot.as_quat()                   # xyzw
    rpy = rot.as_euler("xyz", degrees=True)

    print(f"\n=== T_{args.link_frame}_{args.camera_frame} (computed) ===")
    print(T_link8_camera)
    print(f"\nPosition (xyz, m):  [{pos[0]:+.6f}, {pos[1]:+.6f}, {pos[2]:+.6f}]")
    print(f"Quaternion (xyzw):   [{q[0]:+.6f}, {q[1]:+.6f}, "
          f"{q[2]:+.6f}, {q[3]:+.6f}]")
    print(f"RPY (xyz, deg):      [{rpy[0]:+.2f}, {rpy[1]:+.2f}, {rpy[2]:+.2f}]")

    # Sanity check: distance from link8 to camera should be physically
    # reasonable (a few cm to maybe 30 cm for a wrist-mounted camera).
    dist = np.linalg.norm(pos)
    if dist > 0.5:
        print(f"\nWARNING: |T_link8_camera| = {dist*100:.1f} cm. That's a "
              f"long way from link8 -- check that the robot was at the same\n"
              f"pose during extrinsics calibration as it is right now, and\n"
              f"that the camera is actually mounted on link8 (not fixed in\n"
              f"the world). For a scene camera, use --link-frame "
              f"{args.parent_frame} instead.")

    # ---- ROS 1 launch + CLI snippets ----
    # tf2_ros static_transform_publisher (ROS 1) takes 9 positional args:
    #   x y z qx qy qz qw frame_id child_frame_id
    print("\n=== ROS 1 CLI command (for quick test) ===")
    print(f"rosrun tf2_ros static_transform_publisher \\")
    print(f"  {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} \\")
    print(f"  {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f} \\")
    print(f"  {args.link_frame} {args.camera_frame}")

    print("\n=== ROS 1 launch file XML snippet ===")
    node_name = f"static_tf_{args.link_frame}_to_{args.camera_frame}"
    print(f'<node pkg="tf2_ros" type="static_transform_publisher"')
    print(f'      name="{node_name}"')
    print(f'      args="{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} '
          f'{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f} '
          f'{args.link_frame} {args.camera_frame}" />')

    # ---- Save ----
    if args.save:
        out_path = extr_path.with_name(
            extr_path.stem + f"_{args.link_frame}.npz")
        np.savez(out_path,
                 T_link8_camera=T_link8_camera,
                 T_base_link8_at_capture=T_base_link8,
                 T_base_camera=T_base_camera,
                 link_frame=args.link_frame,
                 camera_frame=args.camera_frame,
                 parent_frame=args.parent_frame,
                 n_tf_samples=len(samples))
        print(f"\nSaved: {out_path}")

    # ---- Optional: publish live for verification ----
    if args.publish:
        broadcaster = tf2_ros.StaticTransformBroadcaster()
        msg = TransformStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = args.link_frame
        msg.child_frame_id = args.camera_frame
        msg.transform.translation.x = float(pos[0])
        msg.transform.translation.y = float(pos[1])
        msg.transform.translation.z = float(pos[2])
        msg.transform.rotation.x = float(q[0])
        msg.transform.rotation.y = float(q[1])
        msg.transform.rotation.z = float(q[2])
        msg.transform.rotation.w = float(q[3])
        broadcaster.sendTransform(msg)
        print(f"\n[publish] broadcasting static TF "
              f"{args.link_frame} -> {args.camera_frame}.")
        print("[publish] verify with:")
        print(f"    rosrun tf tf_echo {args.link_frame} {args.camera_frame}")
        print("[publish] Ctrl-C to exit.")
        try:
            rospy.spin()
        except KeyboardInterrupt:
            print("\n[publish] exiting")


if __name__ == "__main__":
    main()