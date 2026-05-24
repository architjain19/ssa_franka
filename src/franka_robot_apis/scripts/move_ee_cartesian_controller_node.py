#!/usr/bin/env python3
"""
ROS1 services that move robot EE to target pose and BLOCK until completion.

Services:
  /robot/control/move_ee_to_pose       - Move to absolute pose (WebSocket trajectory)
  /robot/control/move_ee_to_rel_pose   - Move by delta position (orientation unchanged)
  /robot/control/move_ee_guarded       - Guarded relative move along a single axis
                                         (cancels trajectory on contact)
  /robot/control/reset_robot           - Return to home pose

Type: RobotCommand.srv / RobotQuery.srv

Example service requests:
  rosservice call /robot/control/move_ee_to_pose \
    "req: '{\"target_pose\": {\"position\": {\"x\": 0.5, \"y\": 0.0, \"z\": 0.5}, \
    \"orientation\": {\"x\": 0.8722, \"y\": -0.4867, \"z\": -0.0424, \"w\": 0.0264}}}'"

  rosservice call /robot/control/move_ee_to_rel_pose \
    "req: '{\"delta_position\": {\"x\": 0.0, \"y\": 0.1, \"z\": 0.3}}'"

  rosservice call /robot/control/move_ee_guarded \
    "req: '{\"axis\": \"z\", \"distance\": -0.05}'"

  rosservice call /robot/control/reset_robot "{}"
"""

import json
import math
import time
import asyncio
import threading
import traceback
import rospy
from geometry_msgs.msg import PoseStamped, WrenchStamped
from robot_api_interfaces.srv import RobotCommand, RobotCommandResponse, RobotQuery, RobotQueryResponse
from robot_api_interfaces.msg import ResultCode
from franka_msgs.msg import FrankaState

try:
    import aiohttp
    _WEBSOCKETS_OK = True
except ImportError:
    _WEBSOCKETS_OK = False
    rospy.logwarn_once(
        "aiohttp library not found. Install with: pip install aiohttp"
    )

AXIS_TO_IDX = {"x": 0, "y": 1, "z": 2}

class MoveEEControllerNode:
    def __init__(self):
        rospy.init_node("move_ee_controller_node")

        # ===== Cartesian Control Parameters =====
        self.equilibrium_pose_topic = rospy.get_param(
            "~equilibrium_pose_topic",
            "/cartesian_impedance_controller/equilibrium_pose"
        )
        self.publish_rate       = rospy.get_param("~publish_rate", 20)
        self.execution_timeout  = rospy.get_param("~execution_timeout", 20.0)
        self.position_tolerance = rospy.get_param("~position_tolerance", 0.01)

        # ===== WebSocket & Trajectory Parameters =====
        self.ws_host    = rospy.get_param("~ws_host", "10.158.54.164")
        self.ws_port    = int(rospy.get_param("~ws_port", 8765))
        self.ws_timeout = float(rospy.get_param("~ws_timeout", 20.0))

        # Trajectory timing & safety
        self.time_scale             = float(rospy.get_param("~time_scale", 1.5))
        self.position_jump_tolerance = float(rospy.get_param("~position_jump_tolerance", 0.3))
        self.ee_convergence_timeout  = float(rospy.get_param("~ee_convergence_timeout", 10.0))
        self.traj_buffer             = float(rospy.get_param("~traj_buffer", 1.0))
        # Convergence / settling
        self.settle_velocity_threshold = float(rospy.get_param("~settle_velocity_threshold", 0.005))   # m/s
        self.settle_position_tolerance = float(rospy.get_param("~settle_position_tolerance", 0.025))   # m, looser
        self.settle_samples            = int(rospy.get_param("~settle_samples", 5))
        
        # Service timeouts
        self.ee_pose_svc_timeout      = float(rospy.get_param("~ee_pose_svc_timeout", 5.0))
        self.current_joints_svc_timeout = float(rospy.get_param("~current_joints_svc_timeout", 5.0))

        # ===== Guarded Move Parameters =====
        self.default_force_threshold = {
            "x": float(rospy.get_param("~default_force_threshold_x", 4.0)),
            "y": float(rospy.get_param("~default_force_threshold_y", 4.0)),
            "z": float(rospy.get_param("~default_force_threshold_z", 4.0)),
        }
        self.guard_force_frame = str(
            rospy.get_param("~guard_force_frame", "ee")
        ).lower().strip()
        if self.guard_force_frame not in ("ee", "base"):
            rospy.logwarn(
                f"Invalid guard_force_frame '{self.guard_force_frame}', defaulting to 'ee'"
            )
            self.guard_force_frame = "ee"

        # Publisher
        self.pose_pub = rospy.Publisher(
            self.equilibrium_pose_topic, PoseStamped, queue_size=1)

        # Subscriber for monitoring
        self.robot_state_sub = rospy.Subscriber(
            "/franka_state_controller/franka_states", FrankaState,
            self._robot_state_callback, queue_size=1)

        # Subscriber for external forces (guarded move)
        self.latest_force      = None  # [fx, fy, fz] in base frame
        self.has_force_data    = False
        self.fext_sub = rospy.Subscriber(
            "/franka_state_controller/F_ext", WrenchStamped,
            self._fext_callback, queue_size=1,
        )

        # ===== Service Proxies =====
        _ee_svc = "/robot/proprioception/get_current_ee_pose"
        rospy.loginfo(f"Waiting for service {_ee_svc} ...")
        try:
            rospy.wait_for_service(_ee_svc, timeout=self.ee_pose_svc_timeout)
        except rospy.ROSException:
            rospy.logwarn(f"Service {_ee_svc} not yet available - will retry on each call.")
        self._ee_pose_proxy = rospy.ServiceProxy(_ee_svc, RobotQuery)

        _joint_svc = "/robot/proprioception/get_current_joints"
        rospy.loginfo(f"Waiting for service {_joint_svc} ...")
        try:
            rospy.wait_for_service(_joint_svc, timeout=self.current_joints_svc_timeout)
        except rospy.ROSException:
            rospy.logwarn(f"Service {_joint_svc} not yet available - will retry on each call.")
        self._current_joints_proxy = rospy.ServiceProxy(_joint_svc, RobotQuery)

        _set_gripper_width_svc = "/robot/control/set_gripper_width"
        rospy.loginfo(f"Waiting for service {_set_gripper_width_svc} ...")
        try:
            rospy.wait_for_service(_set_gripper_width_svc, timeout=self.ee_pose_svc_timeout)
        except rospy.ROSException:
            rospy.logwarn(f"Service {_set_gripper_width_svc} not yet available - will retry on each call.")
        self._set_gripper_width_proxy = rospy.ServiceProxy(_set_gripper_width_svc, RobotCommand)

        # ===== Shared State =====
        self.latest_o_tee    = None
        self.has_received_data = False

        # One motion at a time — used by both absolute and relative services
        self._traj_lock = threading.Lock()

        # Cancellation primitives for guarded move: the guard callback can set the event to signal cancellation, and the main loop checks it between waypoints and during sleeps.
        self._cancel_event = threading.Event()
        self._cancel_hook = None

        self.reset_robot_pose_config = {
            "position":    {"x": 0.4, "y": 0.0, "z": 0.45},
            "orientation": {"x": 1.0, "y": 0.0, "z": 0.0, "w": 0.0},
        }

        # ===== ROS Services =====
        rospy.Service("/robot/control/move_ee_to_pose",     RobotCommand, self._handle_move_ee_to_pose)
        rospy.Service("/robot/control/move_ee_to_rel_pose", RobotCommand, self._handle_move_ee_to_rel_pose)
        rospy.Service("/robot/control/move_ee_guarded",     RobotCommand, self._handle_move_ee_guarded)
        rospy.Service("/robot/control/reset_robot",         RobotQuery,   self._handle_reset_robot)

        rospy.loginfo(
            f"MoveEEControllerNode ready.\n"
            f"  /robot/control/move_ee_to_pose\n"
            f"  /robot/control/move_ee_to_rel_pose\n"
            f"  /robot/control/move_ee_guarded\n"
            f"  /robot/control/reset_robot\n"
            f"  equilibrium_pose_topic : {self.equilibrium_pose_topic}\n"
            f"  ws                     : ws://{self.ws_host}:{self.ws_port}/ws\n"
            f"  time_scale             : {self.time_scale}x\n"
            f"  position_tolerance     : {self.position_tolerance} m\n"
            f"  position_jump_tol      : {self.position_jump_tolerance} m\n"
            f"  websockets             : {'OK' if _WEBSOCKETS_OK else 'MISSING - pip install aiohttp'}\n"
            f"  settle_vel_thresh      : {self.settle_velocity_threshold} m/s\n"
            f"  settle_pos_tol         : {self.settle_position_tolerance} m\n"
            f"  settle_samples         : {self.settle_samples}\n"
            f"  guard_force_thresholds : {self.default_force_threshold} N\n"
            f"  guard_force_frame      : {self.guard_force_frame}"
        )

    # =========================================================================
    # Shared helpers
    # =========================================================================

    def _robot_state_callback(self, msg):
        self.latest_o_tee     = msg.O_T_EE
        self.has_received_data = True

    def _fext_callback(self, msg):
        f = msg.wrench.force
        self.latest_force   = [f.x, f.y, f.z]
        self.has_force_data = True

    def _create_pose_stamped(self, pose_dict):
        msg = PoseStamped()
        msg.header.frame_id = "0"
        msg.header.stamp    = rospy.Time.now()
        msg.pose.position.x    = float(pose_dict["position"]["x"])
        msg.pose.position.y    = float(pose_dict["position"]["y"])
        msg.pose.position.z    = float(pose_dict["position"]["z"])
        msg.pose.orientation.x = float(pose_dict["orientation"]["x"])
        msg.pose.orientation.y = float(pose_dict["orientation"]["y"])
        msg.pose.orientation.z = float(pose_dict["orientation"]["z"])
        msg.pose.orientation.w = float(pose_dict["orientation"]["w"])
        return msg

    def _get_current_pose_dict(self):
        """
        Return current EE pose as a dict from O_T_EE (FrankaState topic).
        Converts the 4×4 column-major transform to position + quaternion
        using Shepperd's method with normalization and canonicalization.
        Returns None if robot state has not been received yet.
        """
        if self.latest_o_tee is None:
            return None

        cx = self.latest_o_tee[12]
        cy = self.latest_o_tee[13]
        cz = self.latest_o_tee[14]

        # Column-major O_T_EE  ->  row-major rotation submatrix
        r = [
            self.latest_o_tee[0], self.latest_o_tee[4], self.latest_o_tee[8],   # R[0,0], R[0,1], R[0,2]
            self.latest_o_tee[1], self.latest_o_tee[5], self.latest_o_tee[9],   # R[1,0], R[1,1], R[1,2]
            self.latest_o_tee[2], self.latest_o_tee[6], self.latest_o_tee[10],  # R[2,0], R[2,1], R[2,2]
        ]

        trace = r[0] + r[4] + r[8]
        if trace > 0:
            s  = math.sqrt(trace + 1.0) * 2.0
            qw = 0.25 * s
            qx = (r[7] - r[5]) / s
            qy = (r[2] - r[6]) / s
            qz = (r[3] - r[1]) / s
        elif (r[0] > r[4]) and (r[0] > r[8]):
            s  = math.sqrt(1.0 + r[0] - r[4] - r[8]) * 2.0
            qw = (r[7] - r[5]) / s
            qx = 0.25 * s
            qy = (r[1] + r[3]) / s
            qz = (r[2] + r[6]) / s
        elif r[4] > r[8]:
            s  = math.sqrt(1.0 + r[4] - r[0] - r[8]) * 2.0
            qw = (r[2] - r[6]) / s
            qx = (r[1] + r[3]) / s
            qy = 0.25 * s
            qz = (r[5] + r[7]) / s
        else:
            s  = math.sqrt(1.0 + r[8] - r[0] - r[4]) * 2.0
            qw = (r[3] - r[1]) / s
            qx = (r[2] + r[6]) / s
            qy = (r[5] + r[7]) / s
            qz = 0.25 * s

        return {
            "position":    {"x": cx,  "y": cy,  "z": cz},
            "orientation": self._normalize_quaternion({"x": qx, "y": qy, "z": qz, "w": qw}),
        }

    def _ensure_quaternion_continuity(self, waypoints_ori_dicts):
        """
        Enforce quaternion sign consistency across the trajectory so the
        controller's SLERP always takes the short arc.
        Each entry is a dict {x, y, z, w}.
        """
        result = []
        for i, q in enumerate(waypoints_ori_dicts):
            if i == 0:
                # Align first waypoint to the robot's current orientation
                current = self._get_current_pose_dict()
                if current is not None:
                    ref = current["orientation"]
                    dot = (ref["w"]*q["w"] + ref["x"]*q["x"] +
                        ref["y"]*q["y"] + ref["z"]*q["z"])
                    if dot < 0:
                        q = {k: -v for k, v in q.items()}
            else:
                # Align to the previous waypoint to avoid mid-trajectory flips
                prev = result[-1]
                dot = (prev["w"]*q["w"] + prev["x"]*q["x"] +
                    prev["y"]*q["y"] + prev["z"]*q["z"])
                if dot < 0:
                    q = {k: -v for k, v in q.items()}
            result.append(q)
        return result

    def _check_at_target(self, target):
        """Return True when position error is within tolerance."""
        if self.latest_o_tee is None:
            return False
        cx = self.latest_o_tee[12]
        cy = self.latest_o_tee[13]
        cz = self.latest_o_tee[14]
        dx = cx - target["position"]["x"]
        dy = cy - target["position"]["y"]
        dz = cz - target["position"]["z"]
        return math.sqrt(dx**2 + dy**2 + dz**2) <= self.position_tolerance

    # =========================================================================
    # Quaternion helpers
    # =========================================================================
    def _apply_gripper_shift(self, target_pose, shift_m):
        """
        Shift target_pose backward along its OWN local -Z axis by `shift_m`.
        Orientation is unchanged. Returns a new pose dict.

        Equivalent to the TF-based "shifted_placement" trick: rotate
        (0, 0, -shift_m) by the target orientation to express the shift in
        the base frame, then add to the target position.
        """
        if shift_m == 0.0:
            return target_pose

        shift_local = {"x": 0.0, "y": 0.0, "z": -float(shift_m)}
        shift_base  = self._rotate_vector_by_quaternion(
            shift_local, target_pose["orientation"]
        )
        return {
            "position": {
                "x": target_pose["position"]["x"] + shift_base["x"],
                "y": target_pose["position"]["y"] + shift_base["y"],
                "z": target_pose["position"]["z"] + shift_base["z"],
            },
            "orientation": target_pose["orientation"],
        }

    def _normalize_quaternion(self, q):
        """
        Normalize to unit length and canonicalize so w >= 0.
        Falls back to identity quaternion if the norm is near zero.
        """
        qx, qy, qz, qw = float(q["x"]), float(q["y"]), float(q["z"]), float(q["w"])
        norm = math.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
        if norm < 1e-10:
            rospy.logwarn(f"Quaternion norm extremely small ({norm}), using identity")
            return {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
        qx /= norm; qy /= norm; qz /= norm; qw /= norm
        if qw < 0:
            qx, qy, qz, qw = -qx, -qy, -qz, -qw
        return {"x": qx, "y": qy, "z": qz, "w": qw}

    def _rotate_vector_by_quaternion(self, vec, quat):
        """
        Rotate a 3D vector by a unit quaternion (active rotation).

        Given the EE orientation quaternion q (base -> EE), this maps a
        vector expressed in the EE frame to its components in the base frame:
            v_base = R(q) * v_ee
        so e.g. (0, 0, 0.1) in the EE frame becomes a 10 cm step along the
        gripper's approach axis expressed in base coordinates.

        vec : dict {x,y,z} or sequence of length 3
        quat: dict {x,y,z,w}
        Returns: dict {x,y,z}
        """
        qn = self._normalize_quaternion(quat)
        qx, qy, qz, qw = qn["x"], qn["y"], qn["z"], qn["w"]

        if isinstance(vec, dict):
            vx, vy, vz = float(vec["x"]), float(vec["y"]), float(vec["z"])
        else:
            vx, vy, vz = float(vec[0]), float(vec[1]), float(vec[2])

        # Standard quaternion-to-rotation-matrix (q = w + xi + yj + zk)
        r00 = 1.0 - 2.0 * (qy*qy + qz*qz)
        r01 = 2.0 * (qx*qy - qw*qz)
        r02 = 2.0 * (qx*qz + qw*qy)
        r10 = 2.0 * (qx*qy + qw*qz)
        r11 = 1.0 - 2.0 * (qx*qx + qz*qz)
        r12 = 2.0 * (qy*qz - qw*qx)
        r20 = 2.0 * (qx*qz - qw*qy)
        r21 = 2.0 * (qy*qz + qw*qx)
        r22 = 1.0 - 2.0 * (qx*qx + qy*qy)

        return {
            "x": r00 * vx + r01 * vy + r02 * vz,
            "y": r10 * vx + r11 * vy + r12 * vz,
            "z": r20 * vx + r21 * vy + r22 * vz,
        }

    def _rotate_vector_by_quaternion_inverse(self, vec, quat):
        """
        Apply the INVERSE rotation: maps a vector expressed in the base frame
        into the EE frame. For unit quaternions, R^-1 = R^T, so we transpose
        the rotation matrix.

            v_ee = R(q)^T * v_base

        Useful for transforming the base-frame F_ext into the EE frame so we
        can check forces along the gripper's local axes.
        """
        qn = self._normalize_quaternion(quat)
        qx, qy, qz, qw = qn["x"], qn["y"], qn["z"], qn["w"]

        if isinstance(vec, dict):
            vx, vy, vz = float(vec["x"]), float(vec["y"]), float(vec["z"])
        else:
            vx, vy, vz = float(vec[0]), float(vec[1]), float(vec[2])

        # Transposed rotation matrix (i.e. inverse rotation for unit quats)
        r00 = 1.0 - 2.0 * (qy*qy + qz*qz)
        r10 = 2.0 * (qx*qy - qw*qz)   # was r01
        r20 = 2.0 * (qx*qz + qw*qy)   # was r02
        r01 = 2.0 * (qx*qy + qw*qz)   # was r10
        r11 = 1.0 - 2.0 * (qx*qx + qz*qz)
        r21 = 2.0 * (qy*qz - qw*qx)   # was r12
        r02 = 2.0 * (qx*qz - qw*qy)   # was r20
        r12 = 2.0 * (qy*qz + qw*qx)   # was r21
        r22 = 1.0 - 2.0 * (qx*qx + qy*qy)

        return {
            "x": r00 * vx + r01 * vy + r02 * vz,
            "y": r10 * vx + r11 * vy + r12 * vz,
            "z": r20 * vx + r21 * vy + r22 * vz,
        }

    # =========================================================================
    # WebSocket & Trajectory Methods
    # =========================================================================

    def _execute_trajectory_to_pose(self, target_pose, guard_callback=None):
        """
        Core blocking trajectory execution shared by the absolute, relative,
        and guarded move services.

        Validates the target, requests a trajectory from the motion server
        via WebSocket, performs safety checks, publishes waypoints with
        scaled timing, and polls for EE convergence.

        Acquires _traj_lock so it cannot run concurrently with itself or
        with `_execute_move_to_pose`.

        Parameters
        ----------
        target_pose : dict
            {"position": {x,y,z}, "orientation": {x,y,z,w}}
        guard_callback : callable or None
            Optional callable invoked once per published waypoint AND once per
            convergence-poll iteration. Signature: () -> (triggered: bool,
            payload: dict). If it returns triggered=True the trajectory is
            aborted, the equilibrium pose is frozen at the current EE pose,
            and the response carries:
              result_code : SUCCESS (guard tripping is a valid outcome)
              data["data"] : whatever the callback put in ``payload`` plus
                             {"stop_reason": "guard_triggered", "stop_position": {...}}

        Returns
        -------
        RobotCommandResponse
        """
        response = RobotCommandResponse()

        if not self._traj_lock.acquire(blocking=False):
            response.result_code.result_code = ResultCode.FAILURE
            response.result_code.message     = "Another motion is already in progress."
            response.data = json.dumps({"success": False, "error": "Motion in progress"})
            return response

        # Reset cancellation state for this run
        self._cancel_event.clear()
        try:
            t_total_start = time.time()

            # Validate target pose values (cheap sanity check on caller)
            try:
                for a in ["x", "y", "z"]:
                    if not isinstance(target_pose["position"][a], (int, float)):
                        raise ValueError(f"Invalid position {a}")
                for a in ["x", "y", "z", "w"]:
                    if not isinstance(target_pose["orientation"][a], (int, float)):
                        raise ValueError(f"Invalid orientation {a}")
            except (ValueError, KeyError) as e:
                response.result_code.result_code = ResultCode.INVALID_INPUT
                response.result_code.message     = f"Bad target pose: {e}"
                response.data = json.dumps({"success": False, "error": str(e)})
                return response
            
            try:
                if target_pose["position"]["z"] > 0.7:
                    rospy.logwarn("Target position z is too high. Capping to 0.7m to avoid unsafe trajectories.")
                    target_pose["position"]["z"] = 0.7
            except Exception as e:
                rospy.logwarn(f"Error checking/capping target z: {e}")
                response.result_code.result_code = ResultCode.INVALID_INPUT
                response.result_code.message     = f"Error checking/capping target z: {e}"
                response.data = json.dumps({"success": False, "error": str(e)})
                return response

            if not self.has_received_data:
                response.result_code.result_code = ResultCode.SERVICE_NOT_RUNNING
                response.result_code.message     = "Robot not connected (no state data received)"
                response.data = json.dumps({"success": False, "error": "Robot not connected"})
                return response

            # ---- Get current state ----
            current_joints = self._get_current_joints()
            if current_joints is None:
                response.result_code.result_code = ResultCode.FAILURE
                response.result_code.message     = "Failed to get current joint state"
                response.data = json.dumps({"success": False, "error": "get_current_joints failed"})
                return response

            current_ee_pose = self._get_current_pose_dict()
            if current_ee_pose is None:
                response.result_code.result_code = ResultCode.FAILURE
                response.result_code.message     = "Failed to get current EE pose"
                response.data = json.dumps({"success": False, "error": "get_current_ee_pose failed"})
                return response

            rospy.loginfo(
                f"Current EE pos: ({current_ee_pose['position']['x']:.3f}, "
                f"{current_ee_pose['position']['y']:.3f}, "
                f"{current_ee_pose['position']['z']:.3f})  "
                f"Target: ({target_pose['position']['x']:.3f}, "
                f"{target_pose['position']['y']:.3f}, "
                f"{target_pose['position']['z']:.3f})"
            )

            # ---- Sanity-check target distance ----
            dx = current_ee_pose["position"]["x"] - target_pose["position"]["x"]
            dy = current_ee_pose["position"]["y"] - target_pose["position"]["y"]
            dz = current_ee_pose["position"]["z"] - target_pose["position"]["z"]
            dist_to_target = math.sqrt(dx**2 + dy**2 + dz**2)

            if dist_to_target > 1.0:
                response.result_code.result_code = ResultCode.INVALID_INPUT
                response.result_code.message     = (
                    f"Target too far ({dist_to_target:.2f}m). Max ~1.0m per move."
                )
                response.data = json.dumps({"success": False, "error": "Target distance too large"})
                return response

            # ---- WebSocket request ----
            ws_msg = self._build_ws_message(current_joints, target_pose)
            rospy.loginfo(f"Sending to WebSocket: {ws_msg[:100]}...")
            ws_raw, ws_err = self._send_ws(ws_msg)
            if ws_err:
                response.result_code.result_code = ResultCode.FAILURE
                response.result_code.message     = f"WebSocket error: {ws_err}"
                response.data = json.dumps({"success": False, "error": f"WebSocket: {ws_err}"})
                return response

            # ---- Parse trajectory ----
            traj_data, parse_err = self._parse_trajectory(ws_raw)
            if parse_err:
                response.result_code.result_code = ResultCode.FAILURE
                response.result_code.message     = f"Trajectory parse error: {parse_err}"
                response.data = json.dumps({"success": False, "error": parse_err})
                return response

            waypoints = traj_data.get("waypoints", [])
            if not waypoints:
                response.result_code.result_code = ResultCode.FAILURE
                response.result_code.message     = "Trajectory contains no waypoints"
                response.data = json.dumps({"success": False, "error": "Empty trajectory"})
                return response

            rospy.loginfo(f"Received trajectory with {len(waypoints)} waypoints")

            # ---- Safety checks ----
            safe, safety_msg = self._check_waypoint_safety(waypoints)
            if not safe:
                response.result_code.result_code = ResultCode.FAILURE
                response.result_code.message     = f"Safety check failed: {safety_msg}"
                response.data = json.dumps({"success": False, "error": safety_msg})
                return response
            rospy.loginfo("Safety check: position jump detection OK")

            # ---- Publish waypoints with scaled timing ----
            raw_dt        = float(traj_data.get("metadata", {}).get("dt", 0.02))
            scaled_dt     = raw_dt * self.time_scale
            traj_duration = (len(waypoints) - 1) * scaled_dt

            rospy.loginfo(
                f"Trajectory: raw_dt={raw_dt:.3f}s, scaled_dt={scaled_dt:.3f}s, "
                f"duration={traj_duration:.2f}s ({self.time_scale}x)"
            )

            ori_dicts = []
            for waypoint in waypoints:
                ori = waypoint.get("orientation", [
                    target_pose["orientation"]["w"],
                    target_pose["orientation"]["x"],
                    target_pose["orientation"]["y"],
                    target_pose["orientation"]["z"],
                ])
                if isinstance(ori, (list, tuple)) and len(ori) == 4:
                    ori_dicts.append({"w": float(ori[0]), "x": float(ori[1]),
                                    "y": float(ori[2]), "z": float(ori[3])})
                else:
                    ori_dicts.append(target_pose["orientation"])

            ori_dicts = self._ensure_quaternion_continuity(ori_dicts)

            guard_payload = {}

            t_start_pub = time.time()
            guard_tripped = False
            last_published_pose = None

            for idx, waypoint in enumerate(waypoints):
                if guard_callback is not None:
                    try:
                        triggered, payload = guard_callback()
                    except Exception as e:
                        rospy.logerr(f"Guard callback raised: {e}")
                        triggered, payload = False, {}
                    if triggered:
                        guard_tripped = True
                        guard_payload = payload or {}
                        rospy.loginfo(
                            f"[Trajectory] Guard tripped at waypoint {idx}/{len(waypoints)}"
                        )
                        break

                desired_time = t_start_pub + idx * scaled_dt
                sleep_time   = desired_time - time.time()
                if sleep_time > 0:
                    # sleep in small slices to allow guard checks at a reasonable frequency
                    slice_dt = 0.01
                    remaining = sleep_time
                    while remaining > 0 and not rospy.is_shutdown():
                        if guard_callback is not None:
                            try:
                                triggered, payload = guard_callback()
                            except Exception as e:
                                rospy.logerr(f"Guard callback raised: {e}")
                                triggered, payload = False, {}
                            if triggered:
                                guard_tripped = True
                                guard_payload = payload or {}
                                break
                        time.sleep(min(slice_dt, remaining))
                        remaining -= slice_dt
                    if guard_tripped:
                        rospy.loginfo(
                            f"[Trajectory] Guard tripped while pacing waypoint {idx}"
                        )
                        break
                try:
                    pos = waypoint.get("position", [0, 0, 0])
                    pose_dict = {
                        "position": {
                            "x": float(pos[0][0]),
                            "y": float(pos[0][1]),
                            "z": float(pos[0][2]),
                        },
                        "orientation": ori_dicts[idx],
                    }
                    self.pose_pub.publish(self._create_pose_stamped(pose_dict))
                    last_published_pose = pose_dict

                    if idx % 50 == 0:
                        rospy.logdebug(
                            f"Waypoint {idx}/{len(waypoints)}: "
                            f"({pose_dict['position']['x']:.3f}, "
                            f"{pose_dict['position']['y']:.3f}, "
                            f"{pose_dict['position']['z']:.3f})"
                        )
                except (IndexError, TypeError, KeyError) as e:
                    rospy.logwarn(f"Error parsing waypoint {idx}: {e}")
                    continue
            
            # Check guard one last time after publishing all waypoints, before we start waiting for convergence.
            if guard_tripped:
                cur_pose_now = self._get_current_pose_dict() or current_ee_pose
                freeze_pose = {
                    "position":    dict(cur_pose_now["position"]),
                    "orientation": last_published_pose["orientation"]
                                   if last_published_pose is not None
                                   else cur_pose_now["orientation"],
                }
                # Publish freeze pose several times to make sure the
                # impedance controller latches onto it.
                for _ in range(5):
                    self.pose_pub.publish(self._create_pose_stamped(freeze_pose))
                    rospy.sleep(0.02)

                total_elapsed = time.time() - t_total_start
                response.result_code.result_code = ResultCode.SUCCESS
                response.result_code.message     = guard_payload.get(
                    "message", "Trajectory cancelled by guard"
                )
                data_block = {
                    "stop_reason":   guard_payload.get("stop_reason", "guard_triggered"),
                    "stop_position": dict(cur_pose_now["position"]),
                    "elapsed":       round(total_elapsed, 3),
                }
                for k, v in guard_payload.items():
                    if k not in ("message", "stop_reason"):
                        data_block[k] = v

                response.data = json.dumps({
                    "result_code": ResultCode.SUCCESS,
                    "message":     response.result_code.message,
                    "data":        data_block,
                })
                return response

            rospy.loginfo(
                f"Finished publishing {len(waypoints)} waypoints in "
                f"{time.time() - t_start_pub:.2f}s"
            )

            # ---- Convergence (guard-aware) ----
            reached, convergence_elapsed, guard_during_settle = \
                self._wait_for_ee_convergence(target_pose, guard_callback=guard_callback)
            total_elapsed = time.time() - t_total_start

            # Guard could still trip while we wait for settling
            if guard_during_settle is not None:
                cur_pose_now = self._get_current_pose_dict() or current_ee_pose
                freeze_pose = {
                    "position":    dict(cur_pose_now["position"]),
                    "orientation": cur_pose_now["orientation"],
                }
                for _ in range(5):
                    self.pose_pub.publish(self._create_pose_stamped(freeze_pose))
                    rospy.sleep(0.02)

                response.result_code.result_code = ResultCode.SUCCESS
                response.result_code.message     = guard_during_settle.get(
                    "message", "Trajectory cancelled by guard during settling"
                )
                data_block = {
                    "stop_reason":   guard_during_settle.get("stop_reason", "guard_triggered"),
                    "stop_position": dict(cur_pose_now["position"]),
                    "elapsed":       round(total_elapsed, 3),
                }
                for k, v in guard_during_settle.items():
                    if k not in ("message", "stop_reason"):
                        data_block[k] = v
                response.data = json.dumps({
                    "result_code": ResultCode.SUCCESS,
                    "message":     response.result_code.message,
                    "data":        data_block,
                })
                return response

            if reached:
                response.result_code.result_code = ResultCode.SUCCESS
                response.result_code.message     = "Trajectory execution completed successfully"
                cur_pose_now = self._get_current_pose_dict() or current_ee_pose
                response.data = json.dumps({
                    "success": True,
                    "message": "Trajectory execution completed successfully",
                    "elapsed_time": round(total_elapsed, 3),
                    "convergence_time": round(convergence_elapsed, 3),
                    "data": {
                        "stop_reason":   "completed",
                        "stop_position": dict(cur_pose_now["position"]),
                        "elapsed":       round(total_elapsed, 3),
                    },
                })
                rospy.loginfo(
                    f"trajectory move SUCCESS — total={total_elapsed:.2f}s "
                    f"(convergence poll={convergence_elapsed:.2f}s)"
                )
            else:
                response.result_code.result_code = ResultCode.TIMEOUT
                response.result_code.message     = (
                    f"EE did not converge within "
                    f"{self.traj_buffer + self.ee_convergence_timeout:.1f}s"
                )
                cur_pose_now = self._get_current_pose_dict() or current_ee_pose
                response.data = json.dumps({
                    "success": False,
                    "error": "Convergence timeout",
                    "elapsed_time": round(total_elapsed, 3),
                    "convergence_time": round(convergence_elapsed, 3),
                    "data": {
                        "stop_reason":   "timeout",
                        "stop_position": dict(cur_pose_now["position"]),
                        "elapsed":       round(total_elapsed, 3),
                    },
                })
                rospy.logwarn(f"trajectory move TIMEOUT — total={total_elapsed:.2f}s")

            return response

        except Exception as e:
            rospy.logerr(f"Unexpected error in _execute_trajectory_to_pose: {traceback.format_exc()}")
            response.result_code.result_code = ResultCode.FAILURE
            response.result_code.message     = f"Unexpected error: {e}"
            response.data = json.dumps({"success": False, "error": str(e)})
            return response

        finally:
            self._traj_lock.release()

    def _get_current_joints(self):
        """
        Get current joint state (9 values: 7 arm + 2 gripper) via service.
        Returns list[float] or None on failure.
        """
        try:
            resp = self._current_joints_proxy()
        except rospy.ServiceException as e:
            rospy.logerr(f"get_current_joints failed: {e}")
            return None

        if resp.result_code.result_code != ResultCode.SUCCESS:
            rospy.logerr(f"get_current_joints non-success: {resp.result_code.message}")
            return None

        try:
            data = json.loads(resp.data)
            joints_dict = data["joints"]
            panda_joints = [
                "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
                "panda_joint5", "panda_joint6", "panda_joint7",
            ]
            joint_positions = [joints_dict[j]["position"] for j in panda_joints]
            gripper_left  = 0.04
            gripper_right = 0.04
            return joint_positions + [gripper_left, gripper_right]
        except (json.JSONDecodeError, KeyError) as e:
            rospy.logerr(f"Failed to parse get_current_joints response: {e}")
            return None

    def _get_ee_pose(self):
        """
        Get current EE pose from proprioception service.
        Returns dict {"position": {x,y,z}, "orientation": {x,y,z,w}} or None.
        """
        try:
            resp = self._ee_pose_proxy()
        except rospy.ServiceException as e:
            rospy.logerr(f"get_current_ee_pose failed: {e}")
            return None

        if resp.result_code.result_code != ResultCode.SUCCESS:
            rospy.logerr(f"get_current_ee_pose non-success: {resp.result_code.message}")
            return None

        try:
            data    = json.loads(resp.data)
            ee_pose = data["ee_pose"]
            _       = ee_pose["position"]["x"]   # validate structure
            _       = ee_pose["orientation"]["w"]
            return ee_pose
        except (json.JSONDecodeError, KeyError) as e:
            rospy.logerr(f"Failed to parse get_current_ee_pose response: {e}")
            return None

    def _build_ws_message(self, current_joints, target_pose):
        """
        Build WebSocket message for the motion server.
        Format: "<j1..j7> <finger1> <finger2>  <tx> <ty> <tz> <tqw> <tqx> <tqy> <tqz>"
        """
        sp = current_joints
        tp = target_pose["position"]
        to = target_pose["orientation"]
        return (
            f"{sp[0]} {sp[1]} {sp[2]} {sp[3]} {sp[4]} {sp[5]} {sp[6]} {sp[7]} {sp[8]} "
            f"{tp['x']} {tp['y']} {tp['z']} "
            f"{to['w']} {to['x']} {to['y']} {to['z']}"
        )

    async def _ws_communicate(self, message):
        """Async WebSocket send/receive."""
        uri = f"ws://{self.ws_host}:{self.ws_port}/ws"
        rospy.loginfo(f"WebSocket connecting to {uri} ...")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(uri, heartbeat=None) as ws:
                    await ws.send_str(message)
                    msg = await asyncio.wait_for(ws.receive(), timeout=self.ws_timeout)
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        return msg.data
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        raise RuntimeError("WebSocket error frame received")
                    else:
                        raise RuntimeError(f"Unexpected WebSocket message type: {msg.type}")
        except asyncio.TimeoutError:
            raise TimeoutError(f"WebSocket did not respond within {self.ws_timeout}s")
        except aiohttp.ClientConnectorError as e:
            raise RuntimeError(f"WebSocket connection refused at {uri}: {e}")

    def _send_ws(self, message):
        """
        Synchronous wrapper around async WebSocket call.
        Returns (response_str, None) on success or (None, error_str) on failure.
        """
        if not _WEBSOCKETS_OK:
            return None, "aiohttp not installed (pip install aiohttp)"
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                response = loop.run_until_complete(self._ws_communicate(message))
            finally:
                loop.close()
            return response, None
        except Exception as e:
            return None, str(e)

    def _parse_trajectory(self, ws_response_str):
        """
        Parse WebSocket JSON response into trajectory dict.
        Accepts a top-level 'trajectory' key or direct trajectory data.
        Returns (traj_dict, None) or (None, error_string).
        """
        try:
            data = json.loads(ws_response_str)
        except json.JSONDecodeError as e:
            return None, f"WebSocket response is not valid JSON: {e}"

        if "trajectory" in data:
            data = data["trajectory"]

        if "waypoints" not in data:
            return None, "Trajectory JSON missing 'waypoints' field"

        return data, None

    def _check_waypoint_safety(self, waypoints):
        """
        Validate that position jumps between consecutive waypoints are within tolerance.
        Returns (True, None) or (False, error_string).
        """
        for i in range(1, len(waypoints)):
            prev_pos = waypoints[i - 1].get("position", [0, 0, 0])
            curr_pos = waypoints[i].get("position", [0, 0, 0])
            if len(curr_pos) >= 3 and len(prev_pos) >= 3:
                dx = float(curr_pos[0]) - float(prev_pos[0])
                dy = float(curr_pos[1]) - float(prev_pos[1])
                dz = float(curr_pos[2]) - float(prev_pos[2])
                jump = math.sqrt(dx**2 + dy**2 + dz**2)
                if jump > self.position_jump_tolerance:
                    return False, (
                        f"Position jump too large between waypoint {i-1} and {i}: "
                        f"{jump:.4f}m (tolerance={self.position_jump_tolerance}m)"
                    )
        return True, None

    # =========================================================================
    # Core blocking movement — used by reset
    # =========================================================================

    def _execute_move_to_pose(self, target_pose):
        """
        Publish *target_pose* continuously until the EE arrives within tolerance
        or the timeout expires. Acquires _traj_lock to prevent concurrent moves.
        Returns a RobotCommandResponse.
        """
        response = RobotCommandResponse()

        if not self._traj_lock.acquire(blocking=False):
            response.result_code.result_code = ResultCode.FAILURE
            response.result_code.message     = "Another motion is already in progress."
            response.data = json.dumps({"success": False, "error": "Motion in progress"})
            return response

        try:
            start_time = rospy.Time.now()
            rate       = rospy.Rate(self.publish_rate)

            tx = target_pose["position"]["x"]
            ty = target_pose["position"]["y"]
            tz = target_pose["position"]["z"]
            rospy.loginfo(f"Moving to: ({tx:.3f}, {ty:.3f}, {tz:.3f})")

            while not rospy.is_shutdown():
                elapsed = (rospy.Time.now() - start_time).to_sec()

                if elapsed > self.execution_timeout:
                    response.result_code.result_code = ResultCode.TIMEOUT
                    response.result_code.message     = f"Timeout after {elapsed:.1f}s"
                    response.data = json.dumps({"success": False, "error": "Timeout", "elapsed": elapsed})
                    return response

                if self._check_at_target(target_pose):
                    rospy.loginfo(f"Reached target in {elapsed:.1f}s")
                    response.result_code.result_code = ResultCode.SUCCESS
                    response.result_code.message     = "Target reached"
                    response.data = json.dumps({"success": True, "elapsed_time": elapsed})
                    return response

                self.pose_pub.publish(self._create_pose_stamped(target_pose))
                rate.sleep()

            response.result_code.result_code = ResultCode.FAILURE
            response.result_code.message     = "Service interrupted"
            response.data = json.dumps({"success": False, "error": "Interrupted"})
            return response

        finally:
            self._traj_lock.release()

    # =========================================================================
    # Convergence polling — used after WebSocket trajectory publishing
    # =========================================================================

    def _wait_for_ee_convergence(self, target_pose, guard_callback=None):
        """
        Poll EE pose until convergence or timeout. Uses the franka_states
        topic data (self.latest_o_tee) directly — no service calls.

        Two acceptance conditions (either succeeds):
          (1) STRICT: position error <= position_tolerance.
          (2) SETTLED: position error <= settle_position_tolerance AND
              EE speed < settle_velocity_threshold for settle_samples
              consecutive polls. This handles compliant impedance
              controllers that have a non-zero steady-state error.

        If guard_callback is provided, it is called every poll. If it returns
        (True, payload), the function aborts and returns
        (False, elapsed, payload). Otherwise the third element is None.

        Returns (reached: bool, elapsed: float, guard_payload: dict|None)
        """
        t_start    = time.time()
        timeout_at = t_start + self.traj_buffer + self.ee_convergence_timeout
        poll_dt    = 0.05  # 20 Hz

        prev_pos      = None
        prev_t        = None
        min_error     = float("inf")
        settled_count = 0
        last_speed    = float("nan")

        # Brief grace period so the controller starts tracking before we judge
        time.sleep(min(self.traj_buffer, 0.2))

        while time.time() < timeout_at and not rospy.is_shutdown():
            # Guard check
            if guard_callback is not None:
                try:
                    triggered, payload = guard_callback()
                except Exception as e:
                    rospy.logerr(f"Guard callback raised: {e}")
                    triggered, payload = False, {}
                if triggered:
                    elapsed = time.time() - t_start
                    rospy.loginfo(f"[Convergence] Guard tripped after {elapsed:.2f}s")
                    return False, elapsed, (payload or {})

            if self.latest_o_tee is None:
                time.sleep(poll_dt)
                continue

            cx = self.latest_o_tee[12]
            cy = self.latest_o_tee[13]
            cz = self.latest_o_tee[14]

            dx = cx - target_pose["position"]["x"]
            dy = cy - target_pose["position"]["y"]
            dz = cz - target_pose["position"]["z"]
            pos_error = math.sqrt(dx*dx + dy*dy + dz*dz)
            if pos_error < min_error:
                min_error = pos_error

            # (1) Strict convergence
            if pos_error <= self.position_tolerance:
                elapsed = time.time() - t_start
                rospy.loginfo(
                    f"EE converged (strict) in {elapsed:.2f}s "
                    f"(error={pos_error*1000:.1f}mm)"
                )
                return True, elapsed, None

            # (2) Settling check — needs a previous sample for velocity
            now = time.time()
            if prev_pos is not None and prev_t is not None:
                dt = now - prev_t
                if dt > 1e-4:
                    vx = (cx - prev_pos[0]) / dt
                    vy = (cy - prev_pos[1]) / dt
                    vz = (cz - prev_pos[2]) / dt
                    last_speed = math.sqrt(vx*vx + vy*vy + vz*vz)

                    if (last_speed <= self.settle_velocity_threshold and
                            pos_error <= self.settle_position_tolerance):
                        settled_count += 1
                        if settled_count >= self.settle_samples:
                            elapsed = now - t_start
                            rospy.loginfo(
                                f"EE settled in {elapsed:.2f}s "
                                f"(error={pos_error*1000:.1f}mm, "
                                f"speed={last_speed*1000:.2f}mm/s)"
                            )
                            return True, elapsed, None
                    else:
                        settled_count = 0

            prev_pos = (cx, cy, cz)
            prev_t   = now
            time.sleep(poll_dt)

        elapsed = time.time() - t_start
        rospy.logwarn(
            f"EE did not converge within {elapsed:.1f}s | "
            f"min_error={min_error*1000:.1f}mm "
            f"(strict_tol={self.position_tolerance*1000:.1f}mm, "
            f"settle_tol={self.settle_position_tolerance*1000:.1f}mm) | "
            f"last_speed={last_speed*1000:.2f}mm/s"
        )
        return False, elapsed, None

    # =========================================================================
    # Service Handlers
    # =========================================================================

    def _handle_move_ee_to_pose(self, req):
        """
        /robot/control/move_ee_to_pose — absolute pose via WebSocket trajectory.

        Automatically applies a backward shift of `gripper_shift_m` along the
        target's local -Z axis (gripper approach axis). The shift can be
        overridden per call by including "gripper_shift" (meters) in the
        request JSON; pass 0.0 to disable for that call.
        """
        response = RobotCommandResponse()
        rospy.loginfo(f"move_ee_to_pose request: {req.req}")

        try:
            data = json.loads(req.req)
            if "target_pose" not in data:
                raise ValueError("Missing 'target_pose'")
            target_pose = data["target_pose"]
            for k in ["x", "y", "z"]:
                if k not in target_pose.get("position", {}):
                    raise ValueError(f"Missing '{k}' in position")
                if k not in target_pose.get("orientation", {}):
                    raise ValueError(f"Missing '{k}' in orientation")
            if "w" not in target_pose.get("orientation", {}):
                raise ValueError("Missing 'w' in orientation")
            shift_m = 0.18
        except (json.JSONDecodeError, ValueError, KeyError, TypeError) as e:
            response.result_code.result_code = ResultCode.INVALID_INPUT
            response.result_code.message     = f"Bad request: {e}"
            response.data = json.dumps({"success": False, "error": str(e)})
            return response

        if shift_m != 0.0:
            shifted = self._apply_gripper_shift(target_pose, shift_m)
            rospy.loginfo(
                f"Gripper shift {shift_m:+.3f}m (local -Z) | "
                f"orig=({target_pose['position']['x']:.3f}, "
                f"{target_pose['position']['y']:.3f}, "
                f"{target_pose['position']['z']:.3f}) -> "
                f"cmd=({shifted['position']['x']:.3f}, "
                f"{shifted['position']['y']:.3f}, "
                f"{shifted['position']['z']:.3f})"
            )
            target_pose = shifted

        return self._execute_trajectory_to_pose(target_pose)

    # -------------------------------------------------------------------------
    # /robot/control/move_ee_to_rel_pose
    # -------------------------------------------------------------------------

    def _parse_delta_position(self, req_json):
        """
        Parse {"delta_position": {"x": float, "y": float, "z": float}}.
        Returns validated delta dict.
        """
        data = json.loads(req_json)
        if "delta_position" not in data:
            raise ValueError("Missing 'delta_position'")
        dp = data["delta_position"]
        for k in ["x", "y", "z"]:
            if k not in dp:
                raise ValueError(f"Missing '{k}' in delta_position")
            if not isinstance(dp[k], (int, float)):
                raise ValueError(f"delta_position '{k}' must be a number")
        return dp

    def _handle_move_ee_to_rel_pose(self, req):
        """
        /robot/control/move_ee_to_rel_pose — applies delta in the END-EFFECTOR
        frame.

        The requested (x, y, z) delta is rotated by the current EE orientation
        so that, e.g., +z always moves along the gripper's approach axis,
        regardless of how the EE is currently posed in the base frame.
        Orientation is preserved, and the resulting absolute pose is executed
        through the same WebSocket trajectory pipeline as
        /robot/control/move_ee_to_pose.
        """
        response = RobotCommandResponse()

        try:
            delta_ee = self._parse_delta_position(req.req)
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            response.result_code.result_code = ResultCode.INVALID_INPUT
            response.result_code.message     = f"Bad request: {e}"
            response.data = json.dumps({"success": False, "error": str(e)})
            return response
        delta_ee["y"] = -delta_ee["y"]
        delta_ee["z"] = -delta_ee["z"]

        if not self.has_received_data:
            response.result_code.result_code = ResultCode.SERVICE_NOT_RUNNING
            response.result_code.message     = "Robot not connected"
            response.data = json.dumps({"success": False, "error": "Robot not connected"})
            return response

        current_pose = self._get_current_pose_dict()
        if current_pose is None:
            response.result_code.result_code = ResultCode.SERVICE_NOT_RUNNING
            response.result_code.message     = "Current EE pose unavailable"
            response.data = json.dumps({"success": False, "error": "Current EE pose unavailable"})
            return response

        # EE-frame delta -> base-frame delta using current EE orientation
        delta_base = self._rotate_vector_by_quaternion(delta_ee, current_pose["orientation"])

        target_pose = {
            "position": {
                "x": current_pose["position"]["x"] + delta_base["x"],
                "y": current_pose["position"]["y"] + delta_base["y"],
                "z": current_pose["position"]["z"] + delta_base["z"],
            },
            # Orientation unchanged
            "orientation": current_pose["orientation"],
        }

        rospy.loginfo(
            f"Relative move (EE frame) — "
            f"delta_ee:   ({delta_ee['x']:.3f}, {delta_ee['y']:.3f}, {delta_ee['z']:.3f}) | "
            f"delta_base: ({delta_base['x']:.3f}, {delta_base['y']:.3f}, {delta_base['z']:.3f}) | "
            f"current:    ({current_pose['position']['x']:.3f}, "
            f"{current_pose['position']['y']:.3f}, "
            f"{current_pose['position']['z']:.3f}) | "
            f"target:     ({target_pose['position']['x']:.3f}, "
            f"{target_pose['position']['y']:.3f}, "
            f"{target_pose['position']['z']:.3f})"
        )

        return self._execute_trajectory_to_pose(target_pose)

    # -------------------------------------------------------------------------
    # /robot/control/move_ee_guarded
    # -------------------------------------------------------------------------

    def _parse_guarded_request(self, req_json):
        """
        Parse and validate the guarded-move JSON request string.

        Returns (axis, distance, force_threshold) or raises ValueError.

        JSON request fields:
          axis             (str)   : "x", "y", or "z"   [required]
          distance         (float) : signed metres to travel [required]
          force_threshold  (float) : N, magnitude on the monitored axis
                                     (default: per-axis ROS param)
        """
        try:
            data = json.loads(req_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON: {exc}") from exc

        # ── axis ─────────────────────────────────────────────────────────────
        if "axis" not in data:
            raise ValueError("Missing required field 'axis'")
        axis = str(data["axis"]).lower().strip()
        if axis not in AXIS_TO_IDX:
            raise ValueError(f"'axis' must be 'x', 'y', or 'z'; got '{axis}'")

        # ── distance ─────────────────────────────────────────────────────────
        if "distance" not in data:
            raise ValueError("Missing required field 'distance'")
        try:
            distance = float(data["distance"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"'distance' must be a number: {exc}") from exc
        if distance == 0.0:
            raise ValueError("'distance' must be non-zero")

        # ── optional force_threshold ─────────────────────────────────────────
        if "force_threshold" in data:
            try:
                force_threshold = float(data["force_threshold"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"'force_threshold' must be a number: {exc}"
                ) from exc
            if force_threshold <= 0:
                raise ValueError("'force_threshold' must be positive")
        else:
            force_threshold = self.default_force_threshold[axis]

        return axis, distance, force_threshold

    def _handle_move_ee_guarded(self, req):
        """
        /robot/control/move_ee_guarded — move along a single axis (EE frame)
        by *distance* metres, aborting the trajectory if the measured
        external force on that axis exceeds *force_threshold*.

        Uses the same WebSocket trajectory planner as move_ee_to_rel_pose;
        the difference is that publishing is cancelled mid-flight if the
        guard trips. On contact the equilibrium pose is frozen at the
        current EE position so the controller stops pulling.

        JSON response data fields:
          stop_reason   : "contact" | "completed" | "timeout" | "error"
          axis          : axis that was monitored
          stop_position : {x, y, z} EE position when motion ended
          elapsed       : seconds elapsed
          contact_force : (only on contact) force vector {x, y, z}
        """
        response = RobotCommandResponse()
        rospy.loginfo(f"move_ee_guarded request: {req.req}")

        # ── readiness guards ─────────────────────────────────────────────────
        if not self.has_received_data:
            response.result_code.result_code = ResultCode.SERVICE_NOT_RUNNING
            response.result_code.message     = "Robot state not available (no /franka_states data)"
            response.data = json.dumps({
                "success": False,
                "error":   "Robot not connected",
            })
            return response

        if not self.has_force_data:
            response.result_code.result_code = ResultCode.SERVICE_NOT_RUNNING
            response.result_code.message     = "F_ext data not available — cannot guard"
            response.data = json.dumps({
                "success": False,
                "error":   "Force data not available",
            })
            return response

        # ── parse request ────────────────────────────────────────────────────
        try:
            axis, distance, force_threshold = self._parse_guarded_request(req.req)
        except ValueError as exc:
            response.result_code.result_code = ResultCode.INVALID_INPUT
            response.result_code.message     = str(exc)
            response.data = json.dumps({"success": False, "error": str(exc)})
            return response

        delta_ee = {"x": 0.0, "y": 0.0, "z": 0.0}
        delta_ee[axis] = distance

        delta_ee_for_motion = dict(delta_ee)
        delta_ee_for_motion["y"] = -delta_ee_for_motion["y"]
        delta_ee_for_motion["z"] = -delta_ee_for_motion["z"]

        current_pose = self._get_current_pose_dict()
        if current_pose is None:
            response.result_code.result_code = ResultCode.SERVICE_NOT_RUNNING
            response.result_code.message     = "Current EE pose unavailable"
            response.data = json.dumps({"success": False, "error": "Current EE pose unavailable"})
            return response

        delta_base = self._rotate_vector_by_quaternion(
            delta_ee_for_motion, current_pose["orientation"]
        )

        target_pose = {
            "position": {
                "x": current_pose["position"]["x"] + delta_base["x"],
                "y": current_pose["position"]["y"] + delta_base["y"],
                "z": current_pose["position"]["z"] + delta_base["z"],
            },
            "orientation": current_pose["orientation"],
        }

        rospy.loginfo(
            f"[GuardedMove] axis={axis}  distance={distance:+.4f} m  "
            f"threshold={force_threshold:.1f} N  frame={self.guard_force_frame}\n"
            f"  delta_ee   : ({delta_ee['x']:.4f}, {delta_ee['y']:.4f}, {delta_ee['z']:.4f})\n"
            f"  delta_base : ({delta_base['x']:.4f}, {delta_base['y']:.4f}, {delta_base['z']:.4f})\n"
            f"  current    : ({current_pose['position']['x']:.4f}, "
            f"{current_pose['position']['y']:.4f}, "
            f"{current_pose['position']['z']:.4f})\n"
            f"  target     : ({target_pose['position']['x']:.4f}, "
            f"{target_pose['position']['y']:.4f}, "
            f"{target_pose['position']['z']:.4f})"
        )

        # ── build the guard callback ─────────────────────────────────────────
        axis_idx        = AXIS_TO_IDX[axis]
        guard_force_frame = self.guard_force_frame

        def _guard_check():
            """
            Returns (triggered: bool, payload: dict).

            payload (when triggered):
              stop_reason      : "contact"
              axis             : monitored axis
              force_threshold  : threshold used
              force_on_axis    : signed force on the monitored axis (frame as configured)
              contact_force    : base-frame force vector {x,y,z}
              contact_force_ee : ee-frame force vector {x,y,z}
              message          : human-readable
            """
            if not self.has_force_data or self.latest_force is None:
                return False, {}

            f_base = {
                "x": self.latest_force[0],
                "y": self.latest_force[1],
                "z": self.latest_force[2],
            }

            ee_ori = (self._get_current_pose_dict()
                      or current_pose)["orientation"]
            f_ee = self._rotate_vector_by_quaternion_inverse(f_base, ee_ori)

            if guard_force_frame == "ee":
                axis_val = (f_ee["x"], f_ee["y"], f_ee["z"])[axis_idx]
            else:
                axis_val = (f_base["x"], f_base["y"], f_base["z"])[axis_idx]

            if abs(axis_val) >= force_threshold:
                msg = (
                    f"Contact detected on axis '{axis}' "
                    f"(|F|={abs(axis_val):.2f} N >= threshold {force_threshold:.1f} N, "
                    f"frame={guard_force_frame})"
                )
                rospy.loginfo(f"[GuardedMove] {msg}")
                return True, {
                    "stop_reason":     "contact",
                    "axis":            axis,
                    "force_threshold": force_threshold,
                    "force_frame":     guard_force_frame,
                    "force_on_axis":   axis_val,
                    "contact_force":   f_base,
                    "contact_force_ee": f_ee,
                    "message":         msg,
                }

            return False, {}

        # ── execute via the shared trajectory pipeline ──────────────────────
        traj_response = self._execute_trajectory_to_pose(
            target_pose, guard_callback=_guard_check
        )

        try:
            payload = json.loads(traj_response.data)
        except (TypeError, ValueError):
            payload = {"raw": traj_response.data}

        data_block = payload.get("data", {})
        if not isinstance(data_block, dict):
            data_block = {}
        data_block.setdefault("axis", axis)
        data_block.setdefault("distance", distance)
        data_block.setdefault("force_threshold", force_threshold)
        data_block.setdefault("force_frame", self.guard_force_frame)

        if (traj_response.result_code.result_code == ResultCode.SUCCESS
                and data_block.get("stop_reason") is None):
            data_block["stop_reason"] = "completed"

        payload["data"] = data_block
        traj_response.data = json.dumps(payload)
        return traj_response

    # -------------------------------------------------------------------------
    # /robot/control/reset_robot
    # -------------------------------------------------------------------------

    def _set_gripper_open(self):
        """
        Helper method to set the gripper to open position after reset.
        Returns a RobotCommandResponse.
        """
        response = RobotCommandResponse()
        try:
            # Assuming the gripper open position corresponds to 0.04m for both fingers
            gripper_open_position = 0.085
            # rosservice call /robot/control/set_gripper_width '{"req": "{\"width\": 0.085}"}'
            req_json = json.dumps({"width": gripper_open_position})
            set_gripper_resp = self._set_gripper_width_proxy(req_json)
            response.result_code = set_gripper_resp.result_code
            response.data = set_gripper_resp.data
            if set_gripper_resp.result_code.result_code != ResultCode.SUCCESS:
                rospy.logerr(f"Failed to open gripper: {set_gripper_resp.result_code.message}")
            else:
                rospy.loginfo("Gripper opened successfully.")
        except Exception as e:
            rospy.logerr(f"Error setting gripper state: {e}")
            response.result_code.result_code = ResultCode.FAILURE
            response.result_code.message     = f"Failed to open gripper: {e}"
            response.data = json.dumps({"success": False, "error": str(e)})
        return response

    def _handle_reset_robot(self, req):
        """
        /robot/control/reset_robot — move to hard-coded home pose using the
        same WebSocket / curobo trajectory pipeline as move_ee_to_pose.

        Internally this forwards a RobotCommand request to
        _handle_move_ee_to_pose with gripper_shift=0.0 (the home pose
        is already the desired TCP pose, not a grasp pose, so no
        approach-axis offset is applied).
        """
        response = RobotQueryResponse()
        try:
            if not self.has_received_data:
                response.result_code.result_code = ResultCode.SERVICE_NOT_RUNNING
                response.result_code.message     = "Robot not connected"
                response.data = json.dumps({"success": False, "error": "Robot not connected"})
                return response

            reset_req_payload = {
                "target_pose":   self.reset_robot_pose_config
            }

            class _ReqShim:
                """Minimal stand-in for a RobotCommand service request."""
                __slots__ = ("req",)
                def __init__(self, req_str):
                    self.req = req_str

            shim_req = _ReqShim(json.dumps(reset_req_payload))
            rospy.loginfo(
                "reset_robot: delegating to move_ee_to_pose via curobo "
                f"trajectory (target={self.reset_robot_pose_config})"
            )
            cmd_res = self._handle_move_ee_to_pose(shim_req)

            if cmd_res.result_code.result_code != ResultCode.SUCCESS:
                rospy.logwarn("Failed to move to reset pose: " + cmd_res.result_code.message)
                response.result_code.result_code = ResultCode.FAILURE
                response.result_code.message     = "Failed to move to reset pose: " + cmd_res.result_code.message
                response.data = json.dumps({
                    "success": False,
                    "error":   "Failed to move to reset pose: " + cmd_res.result_code.message,
                    "move_data": cmd_res.data,
                })
                return response

            # set gripper to open after reset
            gripper_cmd_res = self._set_gripper_open()
            if gripper_cmd_res.result_code.result_code != ResultCode.SUCCESS:
                rospy.logwarn("Failed to open gripper after reset: " + gripper_cmd_res.result_code.message)
                response.result_code.result_code = ResultCode.FAILURE
                response.result_code.message     = "Reset succeeded but failed to open gripper: " + gripper_cmd_res.result_code.message
                response.data = json.dumps({
                    "success": False,
                    "error":   "Failed to open gripper after reset: " + gripper_cmd_res.result_code.message,
                    "move_data": cmd_res.data,
                })
            else:
                rospy.loginfo("Gripper opened successfully after reset.")
                response.result_code.result_code = ResultCode.SUCCESS
                response.result_code.message     = "Reset and gripper open succeeded"
                response.data = json.dumps({
                    "success":   True,
                    "message":   "Reset and gripper open succeeded",
                    "move_data": cmd_res.data,
                })

            return response

        except Exception as e:
            rospy.logerr(f"Unexpected error in _handle_reset_robot: {traceback.format_exc()}")
            response.result_code.result_code = ResultCode.FAILURE
            response.result_code.message     = str(e)
            response.data = json.dumps({"success": False, "error": str(e)})

        return response


def main():
    try:
        node = MoveEEControllerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Shutting down...")
    except Exception as e:
        rospy.logerr(f"Failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()