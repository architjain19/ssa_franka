#!/usr/bin/env python3
"""
ROS1 Noetic services that move the Franka EE to a target pose by:
  1. Calling /robot/proprioception/get_current_ee_pose to get current EE pose
  2. Sending start + target poses to a motion-generation WebSocket server
  3. Receiving a trajectory JSON (panda_joint1…7 waypoints at dt=0.02 s)
  4. Publishing the full JointTrajectory to
     /position_joint_trajectory_controller/command  (blocking until done)

Services
--------
  /robot/control/move_ee_to_pose      (RobotCommand) — absolute target pose
  /robot/control/move_ee_to_rel_pose  (RobotCommand) — delta position, keep orientation
  /robot/control/reset_robot          (RobotQuery)   — move to hardcoded home pose

WebSocket message sent TO server
---------------------------------
  "<j1> <j2> ... <j7> <gf> <gf>  <tx> <ty> <tz> <tqw> <tqx> <tqy> <tqz>"
  (9 joint values: 7 arm + 2 gripper fingers, then target pose)

WebSocket response FROM server (JSON)
---------------------------------------
  {
    "metadata": {"dt": 0.02, "total_waypoints": 62, ...},
    "joint_names": ["panda_joint1", ..., "panda_joint7"],
    "waypoints":   [{"idx": 0, "position": [j1..j7],
                     "velocity": [...], "acceleration": [...]}, ...],
    "current_state": {"position": [j1..j7]}
  }

Trajectory safety & execution notes
-------------------------------------
  CuRobo / Isaac Sim returns trajectories at dt=0.02 s (50 Hz).  With 62
  waypoints that is only ~1.24 s of raw motion — far too fast for a physical
  Franka.  This node applies a configurable TIME_SCALE factor (default 8×)
  which stretches the timing to ~10 s while simultaneously dividing all
  velocities by TIME_SCALE and all accelerations by TIME_SCALE² so that the
  kinematic profile remains consistent.

  Before publishing the trajectory the node runs three safety checks:
    1. Start-state alignment  — waypoint[0] must match live joint state
    2. Inter-waypoint spike   — no consecutive jump > spike_tol radians
    3. Scaled velocity clamp  — per-joint speed capped at max_joint_vel rad/s

  After publishing the node blocks: it sleeps for the total trajectory
  duration (+ traj_buffer), then polls /joint_states until all 7 arm joints
  are within joint_tol of the final waypoint or converge_timeout expires.

Parameters (all ROS private params, set in launch file or via _param:=value)
------------
  ~time_scale          float  8.0     slow-down multiplier (raw_dt × time_scale)
  ~max_joint_vel       float  0.4     per-joint velocity cap after scaling (rad/s)
  ~start_state_tol     float  0.15    max allowed deviation of traj[0] from live
                                      joints before execution is aborted (rad)
  ~spike_tol           float  0.3     max allowed position jump between consecutive
                                      waypoints (rad) — catches planner artefacts
  ~ws_host             str    10.158.54.164
  ~ws_port             int    8765
  ~ws_timeout          float  20.0
  ~traj_topic          str    /position_joint_trajectory_controller/command
  ~traj_buffer         float  1.0     extra seconds after traj duration before polling
  ~converge_timeout    float  10.0    max poll time after trajectory finishes (s)
  ~joint_tol           float  0.05    convergence tolerance (rad)
  ~ee_pose_svc_timeout float  5.0
  ~current_joints_svc_timeout float 5.0

Request JSON (move_ee_to_pose / move_ee_to_rel_pose):
  { "target_pose": {"position": {"x":..,"y":..,"z":..},
                    "orientation": {"x":..,"y":..,"z":..,"w":..}} }
  { "delta_position": {"x": 0.0, "y": 0.05, "z": 0.1} }

Response JSON:
  { "success": true/false, "message": "...", "elapsed_time": 4.2 }

Example calls:
  rosservice call /robot/control/move_ee_to_pose \
    "req: '{\"target_pose\": {\"position\": {\"x\": 0.4, \"y\": 0.0, \"z\": 0.5},
    \"orientation\": {\"x\": 0.0, \"y\": 1.0, \"z\": 0.0, \"w\": 0.0}}}'"

  rosservice call /robot/control/move_ee_to_rel_pose \
    "req: '{\"delta_position\": {\"x\": 0.0, \"y\": 0.0, \"z\": -0.1}}'"

  rosservice call /robot/control/reset_robot "{}"
"""

import json
import math
import time
import asyncio
import threading
import traceback

import rospy

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg    import JointState
from robot_api_interfaces.srv import (
    RobotCommand, RobotCommandResponse,
    RobotQuery,   RobotQueryResponse,
)
from robot_api_interfaces.msg import ResultCode

try:
    import aiohttp
    _WEBSOCKETS_OK = True
except ImportError:
    _WEBSOCKETS_OK = False
    rospy.logwarn_once(
        "aiohttp library not found. Install with: pip install aiohttp"
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PANDA_JOINTS = [
    "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
    "panda_joint5", "panda_joint6", "panda_joint7",
]

# Hardcoded home pose for reset_robot
RESET_POSE = {
    "position":    {"x": 0.40,     "y":  0.0,      "z": 0.5},
    "orientation": {"x": 0.8722,     "y":  -0.4867,      "z": -0.0424, "w": 0.0264},
}

# ---------------------------------------------------------------------------
# Main node class
# ---------------------------------------------------------------------------
class MoveEEControllerNode:
    """
    ROS1 service node that moves the Franka EE via WebSocket motion planning
    + position_joint_trajectory_controller.

    Architecture
    ------------
    Service call
      → get_current_joints (current joint state)
      → WebSocket send (start joints + target pose)
      → WebSocket recv (trajectory JSON)
      → Safety checks (start alignment, spike detection, velocity clamp)
      → Build time-scaled JointTrajectory
      → Publish JointTrajectory (all waypoints, timed)
      → Block until joints converge OR timeout
      → Return result
    """

    def __init__(self):
        # ------------------------------------------------------------------ #
        #  Parameters                                                          #
        # ------------------------------------------------------------------ #
        self.ws_host       = rospy.get_param("~ws_host",        "10.158.54.164")
        self.ws_port       = int(rospy.get_param("~ws_port",    8765))
        self.ws_timeout    = float(rospy.get_param("~ws_timeout",     20.0))

        self.traj_topic    = rospy.get_param(
            "~traj_topic",
            "/position_joint_trajectory_controller/command",
        )

        # --- Trajectory timing & safety parameters ---

        # Slow-down multiplier applied to the CuRobo trajectory.
        # CuRobo returns dt=0.02 s (50 Hz).  time_scale=8 → effective
        # dt=0.16 s per waypoint (~10 s for 62 waypoints).
        # Increase for even slower / safer motion.
        self.time_scale    = float(rospy.get_param("~time_scale",       1.5))

        # Per-joint velocity ceiling AFTER time-scaling (rad/s).
        # Franka's joint speed limits vary by joint (1.5–3 rad/s max hardware).
        # 0.4 rad/s is very conservative and safe for general operation.
        self.max_joint_vel = float(rospy.get_param("~max_joint_vel",    0.4))

        # Maximum allowed deviation between the trajectory's first waypoint
        # and the live robot joint state (rad).  Catches the case where the
        # robot has moved since the plan was generated.
        self.start_state_tol = float(rospy.get_param("~start_state_tol", 0.15))     # if delay in data receving, increase this value to 0.25

        # Maximum allowed position jump between consecutive waypoints (rad).
        # Anything larger is treated as a planner spike / corruption.
        self.spike_tol     = float(rospy.get_param("~spike_tol",        0.3))

        # Extra buffer (seconds) added after trajectory total time before
        # switching from time-sleep to joint-state polling
        self.traj_buffer   = float(rospy.get_param("~traj_buffer",     1.0))

        # Maximum time to poll for convergence after trajectory finishes
        self.converge_timeout = float(rospy.get_param("~converge_timeout", 10.0))

        # Joint position tolerance (radians) to declare "reached"
        self.joint_tol     = float(rospy.get_param("~joint_tol",        0.05))

        # Timeout (seconds) when waiting for the ee_pose service to become available
        self.ee_pose_svc_timeout = float(rospy.get_param("~ee_pose_svc_timeout", 5.0))
        self.current_joints_svc_timeout = float(rospy.get_param("~current_joints_svc_timeout", 5.0))

        # ------------------------------------------------------------------ #
        #  EE pose service proxy                                               #
        # ------------------------------------------------------------------ #
        _ee_svc = "/robot/proprioception/get_current_ee_pose"
        rospy.loginfo(f"Waiting for service {_ee_svc} ...")
        try:
            rospy.wait_for_service(_ee_svc, timeout=self.ee_pose_svc_timeout)
        except rospy.ROSException:
            rospy.logwarn(
                f"Service {_ee_svc} not yet available — will retry on each call."
            )
        self._ee_pose_proxy = rospy.ServiceProxy(_ee_svc, RobotQuery)

        _joint_svc = "/robot/proprioception/get_current_joints"
        rospy.loginfo(f"Waiting for service {_joint_svc} ...")
        try:
            rospy.wait_for_service(_joint_svc, timeout=self.current_joints_svc_timeout)
        except rospy.ROSException:
            rospy.logwarn(
                f"Service {_joint_svc} not yet available — will retry on each call."
            )
        self._current_joints_proxy = rospy.ServiceProxy(_joint_svc, RobotQuery)

        # ------------------------------------------------------------------ #
        #  Joint state cache                                                   #
        # ------------------------------------------------------------------ #
        self._js_lock         = threading.Lock()
        self._latest_js       = None   # sensor_msgs/JointState
        self._js_name_to_idx  = {}     # populated on first message

        rospy.Subscriber(
            "/joint_states", JointState,
            self._joint_state_cb, queue_size=5,
        )

        # ------------------------------------------------------------------ #
        #  Trajectory publisher                                                #
        # ------------------------------------------------------------------ #
        self._traj_pub = rospy.Publisher(
            self.traj_topic, JointTrajectory, queue_size=1, latch=False
        )

        # One request at a time — prevents concurrent trajectory execution
        self._traj_lock = threading.Lock()

        # ------------------------------------------------------------------ #
        #  Services                                                            #
        # ------------------------------------------------------------------ #
        self._svc_abs = rospy.Service(
            "/robot/control/move_ee_to_pose",
            RobotCommand,
            self._handle_move_abs,
        )
        self._svc_rel = rospy.Service(
            "/robot/control/move_ee_to_rel_pose",
            RobotCommand,
            self._handle_move_rel,
        )
        self._svc_reset = rospy.Service(
            "/robot/control/reset_robot",
            RobotQuery,
            self._handle_reset,
        )

        rospy.loginfo(
            "\nMoveEEControllerNode (ROS1) ready.\n"
            f"  /robot/control/move_ee_to_pose\n"
            f"  /robot/control/move_ee_to_rel_pose\n"
            f"  /robot/control/reset_robot\n"
            f"  ee_pose source   : /robot/proprioception/get_current_ee_pose\n"
            f"  ws               : ws://{self.ws_host}:{self.ws_port}/ws\n"
            f"  traj_topic       : {self.traj_topic}\n"
            f"  time_scale       : {self.time_scale}×  "
            f"(raw dt=0.02s → {0.02 * self.time_scale:.3f}s per step)\n"
            f"  max_joint_vel    : {self.max_joint_vel} rad/s\n"
            f"  start_state_tol  : {self.start_state_tol} rad\n"
            f"  spike_tol        : {self.spike_tol} rad\n"
            f"  joint_tol        : {self.joint_tol} rad\n"
            f"  websockets       : {'OK' if _WEBSOCKETS_OK else 'MISSING — pip install aiohttp'}"
        )

    # ------------------------------------------------------------------ #
    #  Joint-state callback                                                #
    # ------------------------------------------------------------------ #

    def _joint_state_cb(self, msg):
        with self._js_lock:
            self._latest_js = msg
            if not self._js_name_to_idx:
                self._js_name_to_idx = {
                    n: i for i, n in enumerate(msg.name)
                }

    def _get_panda_positions(self):
        """
        Return current panda joint positions [j1..j7] or None.
        """
        with self._js_lock:
            js   = self._latest_js
            idx  = self._js_name_to_idx

        if js is None:
            return None
        try:
            return [js.position[idx[j]] for j in PANDA_JOINTS]
        except KeyError:
            return None

    # ------------------------------------------------------------------ #
    #  EE / joint helpers                                                  #
    # ------------------------------------------------------------------ #

    def _get_current_joints(self):
        """
        Return the latest joint state (9 values: 7 arm + 2 gripper) via
        /robot/proprioception/get_current_joints, or None on failure.
        """
        try:
            resp = self._current_joints_proxy()
        except rospy.ServiceException as e:
            rospy.logerr(f"get_current_joints service call failed: {e}")
            return None

        if resp.result_code.result_code != ResultCode.SUCCESS:
            rospy.logerr(
                f"get_current_joints returned non-success: "
                f"{resp.result_code.message}"
            )
            return None

        try:
            data     = json.loads(resp.data)
            joints_dict = data["joints"]
            joint_positions_list = [joints_dict[j]["position"] for j in PANDA_JOINTS]
            return joint_positions_list
        except (json.JSONDecodeError, KeyError) as e:
            rospy.logerr(f"Failed to parse get_current_joints response: {e}")
            return None

    def _get_ee_pose(self):
        """
        Retrieve current EE pose by calling /robot/proprioception/get_current_ee_pose.

        That service reads O_T_EE from /franka_state_controller/franka_states
        and converts it to position + quaternion — no TF dependency needed here.

        Returns:
            dict {"position": {"x","y","z"}, "orientation": {"x","y","z","w"}}
            or None on failure.
        """
        try:
            resp = self._ee_pose_proxy()
        except rospy.ServiceException as e:
            rospy.logerr(f"get_current_ee_pose service call failed: {e}")
            return None

        if resp.result_code.result_code != ResultCode.SUCCESS:
            rospy.logerr(
                f"get_current_ee_pose returned non-success: "
                f"{resp.result_code.message}"
            )
            return None

        try:
            data     = json.loads(resp.data)
            ee_pose  = data["ee_pose"]
            _ = ee_pose["position"]["x"]
            _ = ee_pose["orientation"]["w"]
            return ee_pose
        except (json.JSONDecodeError, KeyError) as e:
            rospy.logerr(f"Failed to parse get_current_ee_pose response: {e}")
            return None

    # ------------------------------------------------------------------ #
    #  WebSocket communication                                             #
    # ------------------------------------------------------------------ #

    def _build_ws_message(self, current_joints, target_pose):
        """
        Build the space-separated string the motion server expects.

        Format:
          "<j1> ... <j7> <gf> <gf>  <tx> <ty> <tz> <tqw> <tqx> <tqy> <tqz>"
          (9 joint values: 7 arm joints + 2 gripper fingers, then target pose)
        """
        sp = current_joints   # 9 values: j1..j7, finger1, finger2
        tp = target_pose["position"]
        to = target_pose["orientation"]
        grpl = 0.04
        grpr = 0.04
        return (
            f"{sp[0]} {sp[1]} {sp[2]} {sp[3]} {sp[4]} {sp[5]} {sp[6]} {grpl} {grpr} "  # current joints + gripper
            f"{tp['x']} {tp['y']} {tp['z']} "
            f"{to['w']} {to['x']} {to['y']} {to['z']}"
        )

    async def _ws_communicate(self, message):
        uri = f"ws://{self.ws_host}:{self.ws_port}/ws"
        rospy.loginfo(f"WebSocket connecting to {uri} ...")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(uri, heartbeat=None) as ws:
                    rospy.logdebug(f"WebSocket connected. Sending: {message}")
                    await ws.send_str(message)
                    msg = await asyncio.wait_for(ws.receive(), timeout=self.ws_timeout)
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        rospy.logdebug(f"WebSocket received {len(msg.data)} chars.")
                        return msg.data
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        raise RuntimeError("WebSocket error frame received")
                    else:
                        raise RuntimeError(f"Unexpected WebSocket message type: {msg.type}")
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"WebSocket server at {uri} did not respond within {self.ws_timeout}s"
            )
        except aiohttp.ClientConnectorError as e:
            raise RuntimeError(f"WebSocket connection refused at {uri}: {e}")
        except Exception as e:
            raise RuntimeError(f"WebSocket error at {uri}: {e}")

    def _send_ws(self, message):
        """
        Synchronous wrapper around the async WebSocket call.

        Returns:
            (str response, None) on success
            (None, error_string) on failure
        """
        if not _WEBSOCKETS_OK:
            return None, "aiohttp library not installed (pip install aiohttp)"
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                response = loop.run_until_complete(self._ws_communicate(message))
            finally:
                loop.close()
            return response, None
        except (TimeoutError, RuntimeError, Exception) as e:
            return None, str(e)

    # ------------------------------------------------------------------ #
    #  Trajectory parsing                                                  #
    # ------------------------------------------------------------------ #

    def _parse_trajectory(self, ws_response_str):
        """
        Parse the WebSocket JSON response into a trajectory dict.

        Accepts two shapes:
          • Response with top-level 'trajectory' key wrapping the data
          • Response that IS the trajectory data directly

        Returns:
            (traj_dict, None) or (None, error_string)
        """
        try:
            data = json.loads(ws_response_str)
        except json.JSONDecodeError as e:
            return None, f"WebSocket response is not valid JSON: {e}"

        # Unwrap if nested under 'trajectory'
        if "trajectory" in data:
            data = data["trajectory"]

        if "waypoints" not in data:
            return None, "Trajectory JSON missing 'waypoints' field"
        if "joint_names" not in data:
            return None, "Trajectory JSON missing 'joint_names' field"

        return data, None

    # ------------------------------------------------------------------ #
    #  Trajectory building — safety checks + time scaling                 #
    # ------------------------------------------------------------------ #

    def _build_joint_trajectory(self, traj_data, current_joints):
        try:
            import numpy as np
            from scipy.interpolate import CubicSpline, make_interp_spline
        except ImportError:
            return None, "scipy not installed — run: pip install scipy"

        waypoints = traj_data.get("waypoints", [])
        if not waypoints:
            return None, "Trajectory contains no waypoints"

        raw_dt    = float(traj_data.get("metadata", {}).get("dt", 0.02))
        scaled_dt = raw_dt * self.time_scale
        nj        = len(PANDA_JOINTS)
        n         = len(waypoints)

        # ---------------------------------------------------------------- #
        #  Safety check 1 — start-state alignment                          #
        # ---------------------------------------------------------------- #
        traj_start_pos = waypoints[0]["position"]
        robot_arm      = current_joints[:7]
        deviations     = [abs(float(traj_start_pos[i]) - float(robot_arm[i])) for i in range(nj)]
        max_dev        = max(deviations)
        max_dev_joint  = PANDA_JOINTS[deviations.index(max_dev)]

        if max_dev > self.start_state_tol:
            return None, (
                f"Start-state alignment FAILED: {max_dev_joint} deviates "
                f"{max_dev:.4f} rad (tolerance={self.start_state_tol:.3f} rad). "
                + ", ".join(f"{PANDA_JOINTS[i]}={deviations[i]:.4f}" for i in range(nj))
            )
        rospy.loginfo(f"[Safety 1/3] Start-state OK (max_dev={max_dev:.5f} rad on {max_dev_joint})")

        # ---------------------------------------------------------------- #
        #  Safety check 2 — inter-waypoint spike detection (on raw data)   #
        # ---------------------------------------------------------------- #
        for i in range(1, n):
            prev_pos = waypoints[i - 1]["position"]
            curr_pos = waypoints[i]["position"]
            for j, jname in enumerate(PANDA_JOINTS):
                jump = abs(float(curr_pos[j]) - float(prev_pos[j]))
                if jump > self.spike_tol:
                    return None, (
                        f"Spike on {jname} wp{i-1}→{i}: "
                        f"Δ={jump:.4f} rad (limit={self.spike_tol:.3f} rad)"
                    )
        rospy.loginfo("[Safety 2/3] Spike detection OK")

        # ---------------------------------------------------------------- #
        #  Smooth positions via cubic spline at CuRobo's original          #
        #  control-point resolution                                         #
        #                                                                   #
        #  CuRobo internally plans at ~10x coarser resolution and          #
        #  resamples to dt=0.02 by LINEAR position interpolation.          #
        #  This creates a "staircase": 10 waypoints with identical         #
        #  positions, then a sudden step — which central-diff converts      #
        #  into a velocity spike (0→0.014 rad/s in one dt at ts=1).       #
        #                                                                   #
        #  Fix: identify the original control points (every RESAMPLE_STEP  #
        #  waypoints) and fit a cubic spline through them, clamped to      #
        #  v=0 at start and end. This eliminates the staircase while       #
        #  faithfully preserving the overall trajectory shape.             #
        #  Max position error vs original: ~1.5e-4 rad (negligible).      #
        # ---------------------------------------------------------------- #

        # Detect the resampling stride: find first nonzero position step
        RESAMPLE_STEP = 10   # CuRobo's internal dt / raw_dt — usually 10 for 0.02→0.2s
        # Auto-detect: find first step with meaningful position change
        for stride_check in [5, 10, 20]:
            if n > stride_check:
                test_delta = max(
                    abs(float(waypoints[stride_check]["position"][j]) -
                        float(waypoints[0]["position"][j]))
                    for j in range(nj)
                )
                if test_delta > 1e-5:
                    RESAMPLE_STEP = stride_check
                    break

        rospy.loginfo(f"Detected CuRobo resample stride: {RESAMPLE_STEP} (every {RESAMPLE_STEP} waypoints = 1 original step)")

        # Build control point indices
        ctrl_indices = list(range(0, n, RESAMPLE_STEP))
        if ctrl_indices[-1] != n - 1:
            ctrl_indices.append(n - 1)

        # times_full = np.array([i * scaled_dt for i in range(n)])
        # times_ctrl = times_full[ctrl_indices]
        # Fit spline on RAW time grid — derivatives are in raw-time units
        # then scale vels/accs down afterward to match the stretched timeline.
        # If you fit on scaled_dt, the spline derivatives are already "slow"
        # and the controller sees inconsistent pos/vel at time_scale=1.
        times_full = np.array([i * raw_dt for i in range(n)])   # <-- raw_dt, not scaled_dt
        times_ctrl = times_full[ctrl_indices]

        positions_all  = np.array([[float(waypoints[i]["position"][j]) for j in range(nj)] for i in range(n)])
        positions_ctrl = positions_all[ctrl_indices]

        # Override first control point with robot's actual current position
        # to eliminate any residual float32 gap at t=0
        positions_ctrl[0] = np.array([float(robot_arm[j]) for j in range(nj)])

        # # After building ctrl_indices, find where motion actually starts
        # MIN_MOTION_THRESH = 1e-4  # rad — anything below this is staircase noise

        # first_motion_idx = ctrl_indices[1]  # default
        # for ci in ctrl_indices[1:]:
        #     delta = max(
        #         abs(float(waypoints[ci]["position"][j]) - float(positions_ctrl[0][j]))
        #         for j in range(nj)
        #     )
        #     if delta > MIN_MOTION_THRESH:
        #         first_motion_idx = ci
        #         break

        # rospy.loginfo(f"First real motion at ctrl index: {first_motion_idx} (wp {ctrl_indices.index(first_motion_idx)})")

        # # Replace ctrl_indices[1] with first_motion_idx if staircase detected
        # # (ctrl_indices[0] stays as wp0, pinned to robot actual position)
        # if first_motion_idx != ctrl_indices[1]:
        #     ctrl_indices[1] = first_motion_idx
        #     rospy.loginfo(f"Staircase detected — skipping to wp{first_motion_idx} as second control point")

        # rospy.loginfo(f"RESAMPLE_STEP={RESAMPLE_STEP}, ctrl_indices[:5]={ctrl_indices[:5]}")

        # Fit clamped cubic spline: v=0 at both endpoints
        smooth_pos  = np.zeros((n, nj))
        smooth_vels = np.zeros((n, nj))
        smooth_accs = np.zeros((n, nj))

        # # Cubic spline with clamped boundary conditions (v=0 at end points)
        # for j in range(nj):
        #     cs = CubicSpline(
        #         times_ctrl, positions_ctrl[:, j],
        #         bc_type=((1, 0.0), (1, 0.0))   # clamp: v=0 at start and end
        #     )
        #     smooth_pos[:, j]  = cs(times_full, 0)
        #     smooth_vels[:, j] = cs(times_full, 1)
        #     smooth_accs[:, j] = cs(times_full, 2)
        
        # Quintic spline with clamped boundary conditions (v=0 AND acc=0 at endpoints):
        bc_zero = ([(1, 0.0), (2, 0.0)], [(1, 0.0), (2, 0.0)])  # vel=0, acc=0 at both ends
        for j in range(nj):
            bspl = make_interp_spline(
                times_ctrl, positions_ctrl[:, j],
                k=5,            # quintic — needs both vel and acc conditions
                bc_type=bc_zero
            )
            smooth_pos[:, j]  = bspl(times_full, 0)
            smooth_vels[:, j] = bspl(times_full, 1)
            smooth_accs[:, j] = bspl(times_full, 2)

        rospy.loginfo(
            f"Spline fit: {len(ctrl_indices)} control points, "
            f"max_pos_err={np.max(np.abs(smooth_pos - positions_all)):.2e} rad, "
            f"peak_vel={np.max(np.abs(smooth_vels)):.4f} rad/s (unscaled)"
        )

        # Scale derivatives to match stretched time_from_start.
        # Velocity in raw-time units → divide by time_scale.
        # Acceleration in raw-time units → divide by time_scale².
        smooth_vels /= self.time_scale
        smooth_accs /= (self.time_scale ** 2)

        # ---------------------------------------------------------------- #
        #  Build JointTrajectory                                            #
        # ---------------------------------------------------------------- #
        traj_msg = JointTrajectory()
        traj_msg.header.stamp = rospy.Time.now()
        traj_msg.joint_names  = PANDA_JOINTS

        max_vel_seen   = 0.0
        clamp_warnings = 0

        for i in range(n):
            pt = JointTrajectoryPoint()
            pt.positions = smooth_pos[i].tolist()

            clamped_vels = []
            for v in smooth_vels[i]:
                max_vel_seen = max(max_vel_seen, abs(v))
                if abs(v) > self.max_joint_vel:
                    clamp_warnings += 1
                    v = math.copysign(self.max_joint_vel, v)
                clamped_vels.append(v)

            pt.velocities    = clamped_vels
            pt.accelerations = smooth_accs[i].tolist()
            pt.effort        = []
            pt.time_from_start = rospy.Duration.from_sec(i * scaled_dt)
            traj_msg.points.append(pt)

        total_dur = traj_msg.points[-1].time_from_start.to_sec()

        if clamp_warnings:
            rospy.logwarn(
                f"[Safety 3/3] Velocity clamped on {clamp_warnings} fields "
                f"to {self.max_joint_vel} rad/s (max before clamping={max_vel_seen:.4f} rad/s)"
            )
        else:
            rospy.loginfo(f"[Safety 3/3] Velocity OK (max={max_vel_seen:.4f} rad/s)")

        rospy.loginfo(
            f"JointTrajectory built: {n} waypoints, "
            f"dt={scaled_dt:.4f}s, total_duration={total_dur:.2f}s"
        )
        return traj_msg, None

    # ------------------------------------------------------------------ #
    #  Blocking convergence wait                                           #
    # ------------------------------------------------------------------ #

    def _wait_for_convergence(self, goal_positions, traj_duration):
        """
        Block the calling thread until the robot arm reaches goal_positions
        or the overall timeout is exceeded.

        Phase 1 — trajectory sleep:
            Sleep for traj_duration + traj_buffer seconds.  During this time
            the position_joint_trajectory_controller is actively tracking the
            published trajectory.  We do not poll here — polling at high rate
            while the controller is mid-execution adds unnecessary load.

        Phase 2 — convergence polling:
            After the trajectory duration, poll /joint_states at 10 Hz.
            Declare success when every arm joint is within self.joint_tol of
            goal_positions.  Give up after self.converge_timeout seconds.

        Args:
            goal_positions (list[float]): 7 target joint positions (rad)
            traj_duration  (float):       total trajectory time in seconds

        Returns:
            (reached: bool, elapsed: float)
        """
        t_start = time.time()

        # Phase 1 — let the controller run
        sleep_dur = traj_duration + self.traj_buffer
        rospy.loginfo(
            f"Trajectory published — sleeping {sleep_dur:.2f}s "
            f"({traj_duration:.2f}s traj + {self.traj_buffer:.2f}s buffer) ..."
        )
        rospy.sleep(sleep_dur)

        # Phase 2 — poll for convergence
        rospy.loginfo("Polling joint states for convergence ...")
        poll_rate  = rospy.Rate(10)   # 10 Hz
        poll_start = time.time()
        max_err    = float("inf")

        while not rospy.is_shutdown():
            current = self._get_panda_positions()
            elapsed = time.time() - t_start

            if current is not None:
                errors  = [
                    abs(current[i] - goal_positions[i])
                    for i in range(len(PANDA_JOINTS))
                ]
                max_err     = max(errors)
                max_err_idx = errors.index(max_err)

                if max_err <= self.joint_tol:
                    rospy.loginfo(
                        f"Convergence reached in {elapsed:.2f}s "
                        f"(max_err={max_err:.4f} rad on "
                        f"{PANDA_JOINTS[max_err_idx]})"
                    )
                    return True, elapsed

            poll_elapsed = time.time() - poll_start
            if poll_elapsed >= self.converge_timeout:
                if current is not None:
                    rospy.logwarn(
                        f"Convergence timeout after {elapsed:.2f}s — "
                        f"max_err={max_err:.4f} rad on "
                        f"{PANDA_JOINTS[errors.index(max_err)]} "
                        f"(tolerance={self.joint_tol} rad). "
                        f"Per-joint errors: "
                        + ", ".join(
                            f"{PANDA_JOINTS[i]}={errors[i]:.4f}"
                            for i in range(len(PANDA_JOINTS))
                        )
                    )
                else:
                    rospy.logwarn(
                        f"Convergence timeout after {elapsed:.2f}s — "
                        f"no joint state data available."
                    )
                return False, elapsed

            poll_rate.sleep()

        # ROS is shutting down
        elapsed = time.time() - t_start
        return False, elapsed

    # ------------------------------------------------------------------ #
    #  Core execution pipeline                                             #
    # ------------------------------------------------------------------ #

    def _execute(self, current_joints, target_pose):
        """
        Full pipeline: WebSocket → safety-checked trajectory → execute → block.

        Steps
        -----
        1. Acquire the trajectory lock (one motion at a time).
        2. Build and send the WebSocket message.
        3. Parse the JSON trajectory from the server response.
        4. Run safety checks and build a time-scaled JointTrajectory message.
        5. Publish the trajectory to the controller.
        6. Block until the robot converges or times out.
        7. Return the populated RobotCommandResponse.

        Args:
            current_joints (list[float]): 9 DOF joint values (arm[7] + gripper[2])
            target_pose    (dict): desired EE pose with "position" and "orientation"

        Returns:
            RobotCommandResponse
        """
        response = RobotCommandResponse()

        if not self._traj_lock.acquire(blocking=False):
            return self._cmd_fail(
                response,
                "Another trajectory is already in progress. Please wait.",
            )

        try:
            # ------------------------------------------------------------ #
            #  Step 1 — Build and send WebSocket message                    #
            # ------------------------------------------------------------ #
            ws_msg = self._build_ws_message(current_joints, target_pose)
            rospy.loginfo(f"Sending to WebSocket: {ws_msg}")
            ws_raw, ws_err = self._send_ws(ws_msg)

            if ws_err:
                return self._cmd_fail(response, f"WebSocket error: {ws_err}")

            # ------------------------------------------------------------ #
            #  Step 2 — Parse trajectory JSON                               #
            # ------------------------------------------------------------ #
            traj_data, parse_err = self._parse_trajectory(ws_raw)
            if parse_err:
                return self._cmd_fail(response, f"Trajectory parse error: {parse_err}")

            # ------------------------------------------------------------ #
            #  Step 3 — Safety checks + build time-scaled JointTrajectory  #
            #                                                               #
            #  _build_joint_trajectory() performs:                         #
            #    • Start-state alignment check vs. live robot joints        #
            #    • Inter-waypoint spike detection                           #
            #    • Velocity scaling (÷ time_scale) + clamping              #
            #    • Acceleration scaling (÷ time_scale²)                    #
            #    • time_from_start stretching (× time_scale)               #
            # ------------------------------------------------------------ #
            traj_msg, build_err = self._build_joint_trajectory(
                traj_data, current_joints
            )
            if build_err:
                return self._cmd_fail(response, f"Trajectory safety check failed: {build_err}")

            # ------------------------------------------------------------ #
            #  Step 4 — Publish to position_joint_trajectory_controller     #
            # ------------------------------------------------------------ #
            traj_duration = traj_msg.points[-1].time_from_start.to_sec()
            rospy.loginfo(
                f"Publishing JointTrajectory: "
                f"{len(traj_msg.points)} waypoints, "
                f"duration={traj_duration:.2f}s"
            )
            # Re-stamp right before publishing so time_from_start is fresh
            traj_msg.header.stamp = rospy.Time.now()
            self._traj_pub.publish(traj_msg)
            t_publish = time.time()

            # ------------------------------------------------------------ #
            #  Step 5 — Block until joints converge or timeout             #
            # ------------------------------------------------------------ #
            goal_positions = [float(p) for p in traj_data["waypoints"][-1]["position"]]

            reached, elapsed = self._wait_for_convergence(
                goal_positions, traj_duration
            )

            if reached:
                response.result_code.result_code = ResultCode.SUCCESS
                response.result_code.message     = (
                    "Trajectory execution completed successfully"
                )
                response.data = json.dumps({
                    "success":      True,
                    "message":      "Trajectory execution completed successfully",
                    "elapsed_time": round(elapsed, 3),
                })
                rospy.loginfo(
                    f"move_ee SUCCESS — elapsed={elapsed:.2f}s, "
                    f"target={[round(p, 4) for p in goal_positions]}"
                )
            else:
                response.result_code.result_code = ResultCode.TIMEOUT
                response.result_code.message     = (
                    f"Joints did not converge within "
                    f"{traj_duration + self.traj_buffer + self.converge_timeout:.1f}s"
                )
                response.data = json.dumps({
                    "success":      False,
                    "error":        "Timeout — joints did not converge",
                    "elapsed_time": round(elapsed, 3),
                })
                rospy.logwarn(
                    f"move_ee TIMEOUT — elapsed={elapsed:.2f}s"
                )

            return response

        except Exception as e:
            rospy.logerr(traceback.format_exc())
            return self._cmd_fail(response, f"Unexpected error: {e}")

        finally:
            self._traj_lock.release()

    # ------------------------------------------------------------------ #
    #  Service handlers                                                    #
    # ------------------------------------------------------------------ #

    def _handle_move_abs(self, request):
        """
        /robot/control/move_ee_to_pose — absolute target pose.

        Expected request.req JSON:
        {
            "target_pose": {
                "position":    {"x": 0.4, "y": 0.0,  "z": 0.5},
                "orientation": {"x": 0.0, "y": 1.0, "z": 0.0, "w": 0.0}
            }
        }
        """
        rospy.loginfo(f"move_ee_to_pose request: {request.req}")
        response = RobotCommandResponse()

        # --- Parse ---
        try:
            data        = json.loads(request.req)
            target_pose = self._extract_pose(data, "target_pose")
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            return self._cmd_fail(response, f"Bad request: {e}")

        # --- Current joints ---
        current_joints = self._get_current_joints()
        if current_joints is None:
            return self._cmd_fail(
                response, "get_current_joints service failed or returned invalid data"
            )
        rospy.loginfo(
            f"Current joints (arm): {[round(j, 4) for j in current_joints[:7]]}\n"
            f"Target pose: pos=({target_pose['position']['x']:.3f}, "
            f"{target_pose['position']['y']:.3f}, "
            f"{target_pose['position']['z']:.3f}) "
            f"ori=({target_pose['orientation']['x']:.3f}, "
            f"{target_pose['orientation']['y']:.3f}, "
            f"{target_pose['orientation']['z']:.3f}, "
            f"{target_pose['orientation']['w']:.3f})"
        )

        return self._execute(current_joints, target_pose)

    def _handle_move_rel(self, request):
        """
        /robot/control/move_ee_to_rel_pose — delta position, orientation unchanged.

        Expected request.req JSON:
        {
            "delta_position": {"x": 0.0, "y": 0.0, "z": -0.1}
        }
        """
        rospy.loginfo(f"move_ee_to_rel_pose request: {request.req}")
        response = RobotCommandResponse()

        # --- Parse ---
        try:
            data  = json.loads(request.req)
            delta = self._extract_delta(data)
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            return self._cmd_fail(response, f"Bad request: {e}")

        # --- Current EE pose ---
        start_pose = self._get_ee_pose()
        if start_pose is None:
            return self._cmd_fail(
                response, "get_current_ee_pose service failed or returned invalid data"
            )

        # --- Current joints ---
        current_joints = self._get_current_joints()
        if current_joints is None:
            return self._cmd_fail(
                response, "get_current_joints service failed or returned invalid data"
            )

        # Build absolute target by adding delta to current position
        target_pose = {
            "position": {
                "x": start_pose["position"]["x"] + float(delta["x"]),
                "y": start_pose["position"]["y"] + float(delta["y"]),
                "z": start_pose["position"]["z"] + float(delta["z"]),
            },
            "orientation": start_pose["orientation"],   # unchanged
        }

        rospy.loginfo(
            f"Rel move: delta=({delta['x']:.3f}, {delta['y']:.3f}, {delta['z']:.3f}) | "
            f"start=({start_pose['position']['x']:.3f}, "
            f"{start_pose['position']['y']:.3f}, "
            f"{start_pose['position']['z']:.3f}) | "
            f"current joints (arm): {[round(j, 4) for j in current_joints[:7]]} | "
            f"target=({target_pose['position']['x']:.3f}, "
            f"{target_pose['position']['y']:.3f}, "
            f"{target_pose['position']['z']:.3f})"
        )

        return self._execute(current_joints, target_pose)

    def _handle_reset(self, request):
        """
        /robot/control/reset_robot — move to fixed home pose.
        """
        rospy.loginfo("reset_robot request received.")
        query_response = RobotQueryResponse()

        current_joints = self._get_current_joints()
        if current_joints is None:
            query_response.result_code.result_code = ResultCode.FAILURE
            query_response.result_code.message     = (
                "get_current_joints service failed or returned invalid data"
            )
            query_response.data = json.dumps({"success": False})
            return query_response

        cmd_response = self._execute(current_joints, RESET_POSE)

        # Mirror RobotCommandResponse → RobotQueryResponse
        query_response.result_code = cmd_response.result_code
        query_response.data        = cmd_response.data
        return query_response

    # ------------------------------------------------------------------ #
    #  Parse helpers                                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_pose(data, key):
        """
        Extract and validate a pose dict from *data[key]*.

        Returns:
            dict with "position" and "orientation" sub-dicts

        Raises:
            ValueError: on missing or invalid fields
        """
        if key not in data:
            raise ValueError(f"Missing '{key}' in request JSON")

        tp = data[key]
        pos = tp.get("position", {})
        ori = tp.get("orientation", {})

        for k in ("x", "y", "z"):
            if k not in pos:
                raise ValueError(f"Missing position.{k}")
            float(pos[k])
        for k in ("x", "y", "z", "w"):
            if k not in ori:
                raise ValueError(f"Missing orientation.{k}")
            float(ori[k])

        return {
            "position":    {k: float(pos[k]) for k in ("x", "y", "z")},
            "orientation": {k: float(ori[k]) for k in ("x", "y", "z", "w")},
        }

    @staticmethod
    def _extract_delta(data):
        """
        Extract and validate a delta_position dict.

        Returns:
            dict {"x": float, "y": float, "z": float}

        Raises:
            ValueError: on missing or invalid fields
        """
        if "delta_position" not in data:
            raise ValueError("Missing 'delta_position' in request JSON")

        dp = data["delta_position"]
        for k in ("x", "y", "z"):
            if k not in dp:
                raise ValueError(f"Missing delta_position.{k}")
            float(dp[k])

        return {k: float(dp[k]) for k in ("x", "y", "z")}

    # ------------------------------------------------------------------ #
    #  Response helpers                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _cmd_fail(response, msg):
        rospy.logerr(f"move_ee error: {msg}")
        response.result_code.result_code = ResultCode.FAILURE
        response.result_code.message     = msg
        response.data = json.dumps({"success": False, "error": msg})
        return response

    # ------------------------------------------------------------------ #
    #  Spin                                                                #
    # ------------------------------------------------------------------ #

    def spin(self):
        rospy.spin()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    rospy.init_node("move_ee_controller_node", anonymous=False)

    try:
        rospy.loginfo("Creating MoveEEControllerNode ...")
        node = MoveEEControllerNode()
        rospy.loginfo("MoveEEControllerNode spinning ...")
        node.spin()

    except rospy.ROSInterruptException:
        rospy.loginfo("ROS interrupt — shutting down.")
    except Exception as e:
        rospy.logerr(f"Fatal error: {e}\n{traceback.format_exc()}")
    finally:
        rospy.loginfo("MoveEEControllerNode shutdown complete.")


if __name__ == "__main__":
    main()