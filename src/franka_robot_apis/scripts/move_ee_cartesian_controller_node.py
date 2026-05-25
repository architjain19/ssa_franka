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

import copy
import json
import math
import time
import asyncio
import threading
import traceback
import rospy
import actionlib
from geometry_msgs.msg import PoseStamped, WrenchStamped
from robot_api_interfaces.srv import RobotCommand, RobotCommandResponse, RobotQuery, RobotQueryResponse
from robot_api_interfaces.msg import ResultCode
from franka_msgs.msg import FrankaState

# ErrorRecoveryAction has moved between franka_msgs.msg and franka_control.msg
# in various franka_ros versions; try both.
try:
    from franka_msgs.msg import ErrorRecoveryAction, ErrorRecoveryGoal
    _ERROR_RECOVERY_OK = True
except ImportError:
    try:
        from franka_control.msg import ErrorRecoveryAction, ErrorRecoveryGoal
        _ERROR_RECOVERY_OK = True
    except ImportError:
        _ERROR_RECOVERY_OK = False
        ErrorRecoveryAction = None
        ErrorRecoveryGoal   = None
        rospy.logwarn_once(
            "ErrorRecoveryAction not importable from franka_msgs or "
            "franka_control. Reflex auto-recovery will be disabled."
        )

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
            "x": float(rospy.get_param("~default_force_threshold_x", 6.0)),
            "y": float(rospy.get_param("~default_force_threshold_y", 6.0)),
            "z": float(rospy.get_param("~default_force_threshold_z", 6.0)),
        }
        self.guard_force_frame = str(
            rospy.get_param("~guard_force_frame", "ee")
        ).lower().strip()
        if self.guard_force_frame not in ("ee", "base"):
            rospy.logwarn(
                f"Invalid guard_force_frame '{self.guard_force_frame}', defaulting to 'ee'"
            )
            self.guard_force_frame = "ee"

        # ===== Error Recovery Parameters =====
        # Automatic reflex recovery: when the robot enters REFLEX mode
        # (e.g. cartesian_reflex from contact), we publish the current EE
        # pose as the new equilibrium, then call /franka_control/error_recovery
        # so the impedance controller can resume without a script restart.
        self.error_recovery_action_name = rospy.get_param(
            "~error_recovery_action", "/franka_control/error_recovery"
        )
        self.auto_recover_reflex   = bool(rospy.get_param("~auto_recover_reflex", True))
        self.reflex_cooldown_s     = float(rospy.get_param("~reflex_cooldown_s", 0.5))
        self.error_recovery_timeout = float(rospy.get_param("~error_recovery_timeout", 5.0))
        # franka_msgs RobotMode constants:
        #   0 OTHER, 1 IDLE, 2 MOVE, 3 GUIDING, 4 REFLEX, 5 USER_STOPPED, 6 AUTOMATIC_ERROR_RECOVERY
        self.ROBOT_MODE_REFLEX                 = 4
        self.ROBOT_MODE_AUTOMATIC_ERROR_RECOVERY = 6

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

        # ===== Reflex Recovery State =====
        # Track latest robot_mode reported by franka_state_controller and
        # whether a reflex occurred in this process's lifetime (so the active
        # trajectory loop can notice it and respond).
        self.latest_robot_mode      = None      # int, see ROBOT_MODE_* above
        self.reflex_count           = 0         # monotonic count of REFLEX entries seen
        self._last_reflex_event_t   = None      # wall-clock time of last REFLEX entry
        self._last_recovery_attempt = 0.0       # wall-clock time of last recovery action send
        self._reflex_lock           = threading.Lock()

        # ErrorRecovery action client. Don't block init if the action server
        # isn't up yet — we retry on each call.
        self._error_recovery_client = None
        if _ERROR_RECOVERY_OK:
            self._error_recovery_client = actionlib.SimpleActionClient(
                self.error_recovery_action_name, ErrorRecoveryAction
            )
            rospy.loginfo(
                f"Waiting briefly for error_recovery action server at "
                f"'{self.error_recovery_action_name}' ..."
            )
            if not self._error_recovery_client.wait_for_server(rospy.Duration(2.0)):
                rospy.logwarn(
                    f"Error-recovery action server '{self.error_recovery_action_name}' "
                    f"not available yet — will retry when needed."
                )
            else:
                rospy.loginfo("Error-recovery action server connected.")
        else:
            rospy.logwarn(
                "Reflex auto-recovery disabled (ErrorRecoveryAction import failed)."
            )
            self.auto_recover_reflex = False

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
        rospy.Service("/robot/control/recover_from_reflex", RobotQuery,   self._handle_recover_from_reflex)

        rospy.loginfo(
            f"MoveEEControllerNode ready.\n"
            f"  /robot/control/move_ee_to_pose\n"
            f"  /robot/control/move_ee_to_rel_pose\n"
            f"  /robot/control/move_ee_guarded\n"
            f"  /robot/control/reset_robot\n"
            f"  /robot/control/recover_from_reflex\n"
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
            f"  guard_force_frame      : {self.guard_force_frame}\n"
            f"  auto_recover_reflex    : {self.auto_recover_reflex}\n"
            f"  reflex_cooldown_s      : {self.reflex_cooldown_s}"
        )

    # =========================================================================
    # Shared helpers
    # =========================================================================

    def _robot_state_callback(self, msg):
        self.latest_o_tee      = msg.O_T_EE
        self.has_received_data = True

        # ---- robot_mode tracking & reflex watchdog ----
        try:
            mode = int(msg.robot_mode)
        except (TypeError, ValueError, AttributeError):
            return
        prev_mode = self.latest_robot_mode
        self.latest_robot_mode = mode

        # Detect entering REFLEX. Use a lock to avoid duplicate increments
        # from rapid callbacks while we're already handling one.
        if mode == self.ROBOT_MODE_REFLEX and prev_mode != self.ROBOT_MODE_REFLEX:
            with self._reflex_lock:
                self.reflex_count        += 1
                self._last_reflex_event_t = time.time()
                rospy.logwarn(
                    f"[ReflexWatchdog] REFLEX entered "
                    f"(event #{self.reflex_count}). prev_mode={prev_mode}"
                )

            if self.auto_recover_reflex:
                # Don't block the franka_states callback — kick off recovery
                # in a separate thread.
                t = threading.Thread(
                    target=self._auto_recover_thread,
                    name="reflex-auto-recover",
                    daemon=True,
                )
                t.start()

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
        """
        if self.latest_o_tee is None:
            return None

        cx = self.latest_o_tee[12]
        cy = self.latest_o_tee[13]
        cz = self.latest_o_tee[14]

        r = [
            self.latest_o_tee[0], self.latest_o_tee[4], self.latest_o_tee[8],
            self.latest_o_tee[1], self.latest_o_tee[5], self.latest_o_tee[9],
            self.latest_o_tee[2], self.latest_o_tee[6], self.latest_o_tee[10],
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
        result = []
        for i, q in enumerate(waypoints_ori_dicts):
            if i == 0:
                current = self._get_current_pose_dict()
                if current is not None:
                    ref = current["orientation"]
                    dot = (ref["w"]*q["w"] + ref["x"]*q["x"] +
                        ref["y"]*q["y"] + ref["z"]*q["z"])
                    if dot < 0:
                        q = {k: -v for k, v in q.items()}
            else:
                prev = result[-1]
                dot = (prev["w"]*q["w"] + prev["x"]*q["x"] +
                    prev["y"]*q["y"] + prev["z"]*q["z"])
                if dot < 0:
                    q = {k: -v for k, v in q.items()}
            result.append(q)
        return result

    def _check_at_target(self, target):
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
    # Reflex / Error Recovery
    # =========================================================================

    def _freeze_equilibrium_pose_here(self, n_publishes=8, sleep_s=0.02):
        """
        Publish the *current* EE pose as the equilibrium pose several times so
        the impedance controller stops pulling toward the old equilibrium
        (which is what caused the reflex). Returns the pose dict that was
        published, or None if pose data isn't available.
        """
        cur = self._get_current_pose_dict()
        if cur is None:
            rospy.logwarn(
                "[ReflexRecovery] Cannot freeze equilibrium — no EE pose data."
            )
            return None
        freeze_pose = {
            "position":    dict(cur["position"]),
            "orientation": dict(cur["orientation"]),
        }
        # Same z-cap safety used elsewhere
        if freeze_pose["position"]["z"] <= 0.19:
            if (abs(freeze_pose["orientation"]["x"]) > 0.9 and
                    abs(freeze_pose["orientation"]["w"]) < 0.1):
                freeze_pose["position"]["z"] = 0.19

        for _ in range(n_publishes):
            try:
                self.pose_pub.publish(self._create_pose_stamped(freeze_pose))
            except Exception as e:
                rospy.logwarn(f"[ReflexRecovery] freeze publish failed: {e}")
                break
            rospy.sleep(sleep_s)
        return freeze_pose

    def _send_error_recovery(self, wait=True):
        """
        Send a goal to /franka_control/error_recovery (or whatever
        ~error_recovery_action is set to). Returns (success: bool, msg: str).

        wait=True blocks for up to self.error_recovery_timeout seconds for
        the action server to finish. wait=False returns immediately after
        sending the goal.
        """
        client = self._error_recovery_client
        if client is None:
            return False, "Error-recovery client not available (import failed)"
        if not client.wait_for_server(rospy.Duration(1.0)):
            return False, (
                f"Error-recovery action server "
                f"'{self.error_recovery_action_name}' not available"
            )

        try:
            client.send_goal(ErrorRecoveryGoal())
        except Exception as e:
            return False, f"send_goal failed: {e}"

        if not wait:
            return True, "Recovery goal sent (no wait)"

        finished = client.wait_for_result(rospy.Duration(self.error_recovery_timeout))
        if not finished:
            return False, (
                f"Error-recovery did not complete within "
                f"{self.error_recovery_timeout}s"
            )

        state = client.get_state()
        # actionlib.GoalStatus.SUCCEEDED = 3
        if state == actionlib.GoalStatus.SUCCEEDED:
            return True, "Error recovery succeeded"
        else:
            return False, (
                f"Error recovery finished with non-success state {state}"
            )

    def _auto_recover_thread(self):
        """
        Background-thread worker that the franka_states callback spawns when
        REFLEX is detected. Freezes the equilibrium pose at the current EE
        position, then calls the error-recovery action. Honors a cooldown
        so we don't spam recovery requests.
        """
        try:
            with self._reflex_lock:
                now = time.time()
                if (now - self._last_recovery_attempt) < self.reflex_cooldown_s:
                    rospy.logwarn(
                        f"[ReflexWatchdog] Skipping auto-recover "
                        f"(cooldown {self.reflex_cooldown_s}s active)."
                    )
                    return
                self._last_recovery_attempt = now

            rospy.logwarn(
                "[ReflexWatchdog] Auto-recovering from REFLEX: freezing "
                "equilibrium pose, then calling error_recovery action."
            )
            self._freeze_equilibrium_pose_here()

            ok, msg = self._send_error_recovery(wait=True)
            if ok:
                rospy.loginfo(f"[ReflexWatchdog] Recovery OK: {msg}")
                # Re-publish freeze pose once more so the restarted
                # controller picks it up immediately.
                self._freeze_equilibrium_pose_here(n_publishes=3, sleep_s=0.02)
            else:
                rospy.logerr(f"[ReflexWatchdog] Recovery FAILED: {msg}")
        except Exception:
            rospy.logerr(
                f"[ReflexWatchdog] Unhandled exception:\n{traceback.format_exc()}"
            )

    def _has_unhandled_reflex_since(self, t_start):
        """
        Return (occurred: bool, event_time: float|None). True if any REFLEX
        event was observed at-or-after wall-clock time t_start.
        """
        evt = self._last_reflex_event_t
        if evt is None:
            return False, None
        return (evt >= t_start), evt

    def _maybe_override_with_reflex(self, response, t_total_start, target_pose_for_logs=None):
        """
        If a REFLEX event occurred during this trajectory (i.e. at or after
        t_total_start), wait briefly for the auto-recovery thread to clear
        it, then rewrite the service response as:

          result_code  : SUCCESS
          data.stop_reason   : "reflex_recovered" (if recovery cleared mode)
                               or "reflex"        (if still in REFLEX)
          data.contact_stopped : True
          data.reflex_event_time : wall-clock timestamp of the reflex
          data.robot_mode_after  : robot_mode after our wait window

        If no reflex was observed since t_total_start, the response is
        returned unchanged.
        """
        occurred, evt_t = self._has_unhandled_reflex_since(t_total_start)
        if not occurred:
            return response

        rospy.logwarn(
            f"[ReflexOverride] Reflex event detected during trajectory "
            f"(event_t={evt_t:.3f}, t_start={t_total_start:.3f}). "
            f"Waiting for recovery to clear..."
        )

        # Wait for auto-recovery to bring us out of REFLEX, bounded by
        # error_recovery_timeout plus a small slack. The watchdog thread is
        # already running (or has finished); we just observe robot_mode.
        deadline = time.time() + self.error_recovery_timeout + 1.0
        while time.time() < deadline and not rospy.is_shutdown():
            if (self.latest_robot_mode is not None and
                    self.latest_robot_mode != self.ROBOT_MODE_REFLEX and
                    self.latest_robot_mode != self.ROBOT_MODE_AUTOMATIC_ERROR_RECOVERY):
                break
            time.sleep(0.05)

        cur_pose_now = self._get_current_pose_dict()
        if cur_pose_now is None:
            cur_pose_now = {"position": {"x": 0.0, "y": 0.0, "z": 0.0},
                            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}

        recovered = (self.latest_robot_mode is not None and
                     self.latest_robot_mode != self.ROBOT_MODE_REFLEX)
        stop_reason = "reflex_recovered" if recovered else "reflex"

        total_elapsed = time.time() - t_total_start
        msg = (
            f"Cartesian reflex during trajectory; "
            f"{'auto-recovered' if recovered else 'still in REFLEX'}."
        )
        rospy.logwarn(f"[ReflexOverride] {msg}")

        # Try to extract any existing data block from the response.data JSON,
        # so we preserve diagnostic fields the caller might rely on.
        existing = {}
        try:
            existing = json.loads(response.data) if response.data else {}
        except (TypeError, ValueError):
            existing = {}
        existing_data = existing.get("data", {})
        if not isinstance(existing_data, dict):
            existing_data = {}

        # Build the new payload
        data_block = dict(existing_data)
        data_block["stop_reason"]      = stop_reason
        data_block["stop_position"]    = dict(cur_pose_now["position"])
        data_block["elapsed"]          = round(total_elapsed, 3)
        data_block["contact_stopped"]  = True
        data_block["reflex_event_time"] = round(evt_t, 3)
        data_block["robot_mode_after"]  = self.latest_robot_mode
        data_block["reflex_count"]      = self.reflex_count

        response.result_code.result_code = ResultCode.SUCCESS
        response.result_code.message     = msg
        response.data = json.dumps({
            "success":      False,
            "message":      msg,
            "elapsed_time": round(total_elapsed, 3),
            "data":         data_block,
        })
        return response

    # =========================================================================
    # Quaternion helpers
    # =========================================================================
    def _apply_gripper_shift(self, target_pose, shift_m):
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
        qn = self._normalize_quaternion(quat)
        qx, qy, qz, qw = qn["x"], qn["y"], qn["z"], qn["w"]

        if isinstance(vec, dict):
            vx, vy, vz = float(vec["x"]), float(vec["y"]), float(vec["z"])
        else:
            vx, vy, vz = float(vec[0]), float(vec[1]), float(vec[2])

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
        qn = self._normalize_quaternion(quat)
        qx, qy, qz, qw = qn["x"], qn["y"], qn["z"], qn["w"]

        if isinstance(vec, dict):
            vx, vy, vz = float(vec["x"]), float(vec["y"]), float(vec["z"])
        else:
            vx, vy, vz = float(vec[0]), float(vec[1]), float(vec[2])

        r00 = 1.0 - 2.0 * (qy*qy + qz*qz)
        r10 = 2.0 * (qx*qy - qw*qz)
        r20 = 2.0 * (qx*qz + qw*qy)
        r01 = 2.0 * (qx*qy + qw*qz)
        r11 = 1.0 - 2.0 * (qx*qx + qz*qz)
        r21 = 2.0 * (qy*qz - qw*qx)
        r02 = 2.0 * (qx*qz - qw*qy)
        r12 = 2.0 * (qy*qz + qw*qx)
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
        """
        response = RobotCommandResponse()

        if not self._traj_lock.acquire(blocking=False):
            response.result_code.result_code = ResultCode.FAILURE
            response.result_code.message     = "Another motion is already in progress."
            response.data = json.dumps({"success": False, "error": "Motion in progress"})
            return response

        self._cancel_event.clear()
        try:
            t_total_start = time.time()

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

            ws_msg = self._build_ws_message(current_joints, target_pose)
            rospy.loginfo(f"Sending to WebSocket: {ws_msg[:100]}...")
            ws_raw, ws_err = self._send_ws(ws_msg)
            if ws_err:
                response.result_code.result_code = ResultCode.FAILURE
                response.result_code.message     = f"WebSocket error: {ws_err}"
                response.data = json.dumps({"success": False, "error": f"WebSocket: {ws_err}"})
                return response

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

            safe, safety_msg = self._check_waypoint_safety(waypoints)
            if not safe:
                response.result_code.result_code = ResultCode.FAILURE
                response.result_code.message     = f"Safety check failed: {safety_msg}"
                response.data = json.dumps({"success": False, "error": safety_msg})
                return response
            rospy.loginfo("Safety check: position jump detection OK")

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
                    if pose_dict["position"]["z"] <= 0.19:
                        if abs(pose_dict["orientation"]["x"]) > 0.9 and abs(pose_dict["orientation"]["w"]) < 0.1:
                            rospy.logwarn(f"CAPPING Z to 0.19m to avoid unsafe trajectory at waypoint {idx}")
                            pose_dict["position"]["z"] = 0.19
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

            if guard_tripped:
                cur_pose_now = self._get_current_pose_dict() or current_ee_pose
                freeze_pose = {
                    "position":    dict(cur_pose_now["position"]),
                    "orientation": last_published_pose["orientation"]
                                   if last_published_pose is not None
                                   else cur_pose_now["orientation"],
                }
                for _ in range(5):
                    if freeze_pose["position"]["z"] <= 0.19:
                        if abs(freeze_pose["orientation"]["x"]) > 0.9 and abs(freeze_pose["orientation"]["w"]) < 0.1:
                            rospy.logwarn(f"CAPPING Z to 0.19m to avoid unsafe trajectory at waypoint {idx}")
                            freeze_pose["position"]["z"] = 0.19
                    self.pose_pub.publish(self._create_pose_stamped(freeze_pose))
                    rospy.sleep(0.02)

                total_elapsed = time.time() - t_total_start
                response.result_code.result_code = ResultCode.SUCCESS
                response.result_code.message     = guard_payload.get(
                    "message", "Trajectory cancelled by guard"
                )
                data_block = {
                    "stop_reason":     guard_payload.get("stop_reason", "guard_triggered"),
                    "stop_position":   dict(cur_pose_now["position"]),
                    "elapsed":         round(total_elapsed, 3),
                    "contact_stopped": True,
                }
                for k, v in guard_payload.items():
                    if k not in ("message", "stop_reason"):
                        data_block[k] = v

                response.data = json.dumps({
                    "result_code": ResultCode.SUCCESS,
                    "message":     response.result_code.message,
                    "data":        data_block,
                })
                return self._maybe_override_with_reflex(response, t_total_start)

            rospy.loginfo(
                f"Finished publishing {len(waypoints)} waypoints in "
                f"{time.time() - t_start_pub:.2f}s"
            )

            reached, convergence_elapsed, guard_during_settle = \
                self._wait_for_ee_convergence(target_pose, guard_callback=guard_callback)
            total_elapsed = time.time() - t_total_start

            if guard_during_settle is not None:
                cur_pose_now = self._get_current_pose_dict() or current_ee_pose
                freeze_pose = {
                    "position":    dict(cur_pose_now["position"]),
                    "orientation": cur_pose_now["orientation"],
                }
                for _ in range(5):
                    if freeze_pose["position"]["z"] <= 0.19:
                        if abs(freeze_pose["orientation"]["x"]) > 0.9 and abs(freeze_pose["orientation"]["w"]) < 0.1:
                            rospy.logwarn(f"CAPPING Z to 0.19m to avoid unsafe trajectory at waypoint {idx}")
                            freeze_pose["position"]["z"] = 0.19
                    self.pose_pub.publish(self._create_pose_stamped(freeze_pose))
                    rospy.sleep(0.02)

                response.result_code.result_code = ResultCode.SUCCESS
                response.result_code.message     = guard_during_settle.get(
                    "message", "Trajectory cancelled by guard during settling"
                )
                data_block = {
                    "stop_reason":     guard_during_settle.get("stop_reason", "guard_triggered"),
                    "stop_position":   dict(cur_pose_now["position"]),
                    "elapsed":         round(total_elapsed, 3),
                    "contact_stopped": True,
                }
                for k, v in guard_during_settle.items():
                    if k not in ("message", "stop_reason"):
                        data_block[k] = v
                response.data = json.dumps({
                    "result_code": ResultCode.SUCCESS,
                    "message":     response.result_code.message,
                    "data":        data_block,
                })
                return self._maybe_override_with_reflex(response, t_total_start)

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
                        "stop_reason":     "completed",
                        "stop_position":   dict(cur_pose_now["position"]),
                        "elapsed":         round(total_elapsed, 3),
                        "contact_stopped": False,
                    },
                })
                rospy.loginfo(
                    f"trajectory move SUCCESS — total={total_elapsed:.2f}s "
                    f"(convergence poll={convergence_elapsed:.2f}s)"
                )
            else:
                # Convergence timeout. If contact-guarding is enabled, this
                # almost always means the robot is being physically blocked
                # (force ramped just below the guard threshold, or contact
                # spread across multiple axes such that no single axis
                # crossed its threshold). Treat as a successful contact
                # stop rather than a failure, and freeze the equilibrium
                # pose so the controller stops pulling against the obstacle.
                cur_pose_now = self._get_current_pose_dict() or current_ee_pose

                if guard_callback is not None:
                    freeze_pose = {
                        "position":    dict(cur_pose_now["position"]),
                        "orientation": cur_pose_now["orientation"],
                    }
                    for _ in range(5):
                        if freeze_pose["position"]["z"] <= 0.19:
                            if (abs(freeze_pose["orientation"]["x"]) > 0.9 and
                                    abs(freeze_pose["orientation"]["w"]) < 0.1):
                                freeze_pose["position"]["z"] = 0.19
                        self.pose_pub.publish(self._create_pose_stamped(freeze_pose))
                        rospy.sleep(0.02)

                    # Snapshot the current force reading for diagnostics
                    force_snapshot = None
                    if self.latest_force is not None:
                        f_base = {
                            "x": self.latest_force[0],
                            "y": self.latest_force[1],
                            "z": self.latest_force[2],
                        }
                        try:
                            f_ee = self._rotate_vector_by_quaternion_inverse(
                                f_base, cur_pose_now["orientation"]
                            )
                        except Exception:
                            f_ee = None
                        force_snapshot = {
                            "contact_force":    f_base,
                            "contact_force_ee": f_ee,
                        }

                    msg = (
                        f"Trajectory stalled before reaching target "
                        f"({convergence_elapsed:.2f}s); inferring contact "
                        f"since guard is enabled."
                    )
                    rospy.logwarn(f"[Trajectory] {msg}")

                    response.result_code.result_code = ResultCode.SUCCESS
                    response.result_code.message     = msg
                    data_block = {
                        "stop_reason":     "contact_inferred",
                        "stop_position":   dict(cur_pose_now["position"]),
                        "elapsed":         round(total_elapsed, 3),
                        "contact_stopped": True,
                    }
                    if force_snapshot is not None:
                        data_block.update(force_snapshot)

                    response.data = json.dumps({
                        "success":          True,
                        "message":          msg,
                        "elapsed_time":     round(total_elapsed, 3),
                        "convergence_time": round(convergence_elapsed, 3),
                        "data":             data_block,
                    })
                else:
                    response.result_code.result_code = ResultCode.TIMEOUT
                    response.result_code.message     = (
                        f"IK convergence timeout after {convergence_elapsed:.2f}s while waiting for EE to reach target pose"
                    )
                    response.data = json.dumps({
                        "success": False,
                        "error": "Convergence timeout",
                        "elapsed_time": round(total_elapsed, 3),
                        "convergence_time": round(convergence_elapsed, 3),
                        "data": {
                            "stop_reason":     "timeout",
                            "stop_position":   dict(cur_pose_now["position"]),
                            "elapsed":         round(total_elapsed, 3),
                            "contact_stopped": False,
                        },
                    })
                    rospy.logwarn(f"trajectory move TIMEOUT — total={total_elapsed:.2f}s")

            return self._maybe_override_with_reflex(response, t_total_start)

        except Exception as e:
            rospy.logerr(f"Unexpected error in _execute_trajectory_to_pose: {traceback.format_exc()}")
            response.result_code.result_code = ResultCode.FAILURE
            response.result_code.message     = f"Unexpected error: {e}"
            response.data = json.dumps({"success": False, "error": str(e)})
            return response

        finally:
            self._traj_lock.release()

    def _get_current_joints(self):
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
            _       = ee_pose["position"]["x"]
            _       = ee_pose["orientation"]["w"]
            return ee_pose
        except (json.JSONDecodeError, KeyError) as e:
            rospy.logerr(f"Failed to parse get_current_ee_pose response: {e}")
            return None

    def _build_ws_message(self, current_joints, target_pose):
        sp = current_joints
        tp = target_pose["position"]
        to = target_pose["orientation"]
        return (
            f"{sp[0]} {sp[1]} {sp[2]} {sp[3]} {sp[4]} {sp[5]} {sp[6]} {sp[7]} {sp[8]} "
            f"{tp['x']} {tp['y']} {tp['z']} "
            f"{to['w']} {to['x']} {to['y']} {to['z']}"
        )

    async def _ws_communicate(self, message):
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
        try:
            data = json.loads(ws_response_str)
        except json.JSONDecodeError as e:
            return None, f"WebSocket response is not valid JSON: {e}"

        if "trajectory" in data:
            data = data["trajectory"]

        if "waypoints" not in data:
            return None, "Failed to solve IK"

        return data, None

    def _check_waypoint_safety(self, waypoints):
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
    # Convergence polling — used after WebSocket trajectory publishing
    # =========================================================================

    def _wait_for_ee_convergence(self, target_pose, guard_callback=None):
        t_start    = time.time()
        timeout_at = t_start + self.traj_buffer + self.ee_convergence_timeout
        poll_dt    = 0.05

        if target_pose["position"]["z"] < 0.19:
            if abs(target_pose["orientation"]["x"]) > 0.9 and abs(target_pose["orientation"]["w"]) < 0.1:
                rospy.logwarn(f"Convergence check: CAPPING target Z to 0.19m to avoid unsafe trajectory")
                target_pose = copy.deepcopy(target_pose)
                target_pose["position"]["z"] = 0.19

        prev_pos      = None
        prev_t        = None
        min_error     = float("inf")
        settled_count = 0
        last_speed    = float("nan")

        time.sleep(min(self.traj_buffer, 0.2))

        while time.time() < timeout_at and not rospy.is_shutdown():
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

            if pos_error <= self.position_tolerance:
                elapsed = time.time() - t_start
                rospy.loginfo(
                    f"EE converged (strict) in {elapsed:.2f}s "
                    f"(error={pos_error*1000:.1f}mm)"
                )
                return True, elapsed, None

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
    # Contact-guard helpers (shared by pose/rel_pose/guarded services)
    # =========================================================================

    def _parse_guard_options(self, data, default_axes=("x", "y", "z")):
        """
        Parse optional contact-guard fields from a request JSON dict.

        Recognised fields:
          guard            (bool)            : enable contact guarding (default False)
          force_threshold  (number or dict)  : N. A scalar applies to all
                                               monitored axes; a dict
                                               {x:.., y:.., z:..} sets per-axis
                                               thresholds. Missing axes fall
                                               back to self.default_force_threshold.
          guard_axes       (list[str])       : subset of ["x","y","z"] to monitor.
                                               Defaults to default_axes.

        Returns (guard_enabled, per_axis_thresholds_dict, monitored_axes_tuple).
        Raises ValueError on malformed input.
        """
        guard_enabled = bool(data.get("guard", False))

        # Axes
        axes_in = data.get("guard_axes", list(default_axes))
        if not isinstance(axes_in, (list, tuple)):
            raise ValueError("'guard_axes' must be a list of axis names")
        monitored_axes = []
        for a in axes_in:
            a_norm = str(a).lower().strip()
            if a_norm not in AXIS_TO_IDX:
                raise ValueError(f"Invalid axis '{a}' in guard_axes; must be x/y/z")
            if a_norm not in monitored_axes:
                monitored_axes.append(a_norm)
        if not monitored_axes:
            raise ValueError("'guard_axes' must contain at least one axis")

        # Thresholds — start from defaults, override from request
        thresholds = dict(self.default_force_threshold)
        ft_in = data.get("force_threshold", None)
        if ft_in is not None:
            if isinstance(ft_in, dict):
                for a, v in ft_in.items():
                    a_norm = str(a).lower().strip()
                    if a_norm not in AXIS_TO_IDX:
                        raise ValueError(
                            f"Invalid axis '{a}' in force_threshold dict"
                        )
                    try:
                        fv = float(v)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            f"force_threshold['{a}'] must be a number: {exc}"
                        ) from exc
                    if fv <= 0:
                        raise ValueError(
                            f"force_threshold['{a}'] must be positive"
                        )
                    thresholds[a_norm] = fv
            else:
                try:
                    fv = float(ft_in)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"'force_threshold' must be a number or dict: {exc}"
                    ) from exc
                if fv <= 0:
                    raise ValueError("'force_threshold' must be positive")
                for a in monitored_axes:
                    thresholds[a] = fv

        return guard_enabled, thresholds, tuple(monitored_axes)

    def _build_contact_guard(self, monitored_axes, thresholds, fallback_orientation=None):
        """
        Build a guard callback that monitors EE-frame (or base-frame) external
        forces on the given axes and trips when any one exceeds its threshold.

        monitored_axes      : iterable of "x"/"y"/"z"
        thresholds          : dict {axis: N}
        fallback_orientation: orientation dict to use if the current EE pose
                              is momentarily unavailable. Optional.

        Returns a zero-arg callable returning (triggered, payload). The payload
        on contact mirrors the shape historically used by move_ee_guarded:
          stop_reason     : "contact"
          axis            : axis that triggered
          force_threshold : threshold that was exceeded (N)
          force_frame     : "ee" or "base"
          force_on_axis   : signed force on the triggering axis
          contact_force   : base-frame force {x,y,z}
          contact_force_ee: ee-frame force   {x,y,z}
          message         : human-readable description
          monitored_axes  : list of axes that were being watched
        """
        if not monitored_axes:
            # Disabled guard: never trips
            def _noop():
                return False, {}
            return _noop

        guard_frame = self.guard_force_frame
        axis_indices = [(a, AXIS_TO_IDX[a]) for a in monitored_axes]

        def _guard_check():
            if not self.has_force_data or self.latest_force is None:
                return False, {}

            f_base = {
                "x": self.latest_force[0],
                "y": self.latest_force[1],
                "z": self.latest_force[2],
            }

            cur_pose = self._get_current_pose_dict()
            if cur_pose is not None:
                ee_ori = cur_pose["orientation"]
            elif fallback_orientation is not None:
                ee_ori = fallback_orientation
            else:
                # Without orientation we can't transform to EE frame; bail safely.
                return False, {}

            f_ee = self._rotate_vector_by_quaternion_inverse(f_base, ee_ori)

            for axis, idx in axis_indices:
                if guard_frame == "ee":
                    axis_val = (f_ee["x"], f_ee["y"], f_ee["z"])[idx]
                else:
                    axis_val = (f_base["x"], f_base["y"], f_base["z"])[idx]

                thr = thresholds.get(axis, self.default_force_threshold[axis])
                if abs(axis_val) >= thr:
                    msg = (
                        f"Contact detected on axis '{axis}' "
                        f"(|F|={abs(axis_val):.2f} N >= threshold {thr:.1f} N, "
                        f"frame={guard_frame})"
                    )
                    rospy.loginfo(f"[ContactGuard] {msg}")
                    return True, {
                        "stop_reason":      "contact",
                        "axis":             axis,
                        "force_threshold":  thr,
                        "force_frame":      guard_frame,
                        "force_on_axis":    axis_val,
                        "contact_force":    f_base,
                        "contact_force_ee": f_ee,
                        "monitored_axes":   list(monitored_axes),
                        "message":          msg,
                    }

            return False, {}

        return _guard_check

    # =========================================================================
    # Service Handlers
    # =========================================================================

    def _handle_move_ee_to_pose(self, req):
        """
        /robot/control/move_ee_to_pose — absolute pose via WebSocket trajectory.

        JSON request fields:
          target_pose      (dict, required)    : {"position": {...}, "orientation": {...}}
          guard            (bool, optional)    : enable contact-guarded motion
          force_threshold  (number|dict, opt.) : N. Scalar or per-axis dict.
          guard_axes       (list, optional)    : subset of ["x","y","z"]; default all.

        On contact: result_code = SUCCESS, response.data.data.stop_reason = "contact",
        data.contact_stopped = True, plus the usual contact payload.
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

            guard_enabled, thresholds, monitored_axes = self._parse_guard_options(data)
        except (json.JSONDecodeError, ValueError, KeyError, TypeError) as e:
            response.result_code.result_code = ResultCode.INVALID_INPUT
            response.result_code.message     = f"Bad request: {e}"
            response.data = json.dumps({"success": False, "error": str(e)})
            return response

        # Refuse to guard if we never received F_ext data — quietly disable
        # would mask a hardware problem.
        if guard_enabled and not self.has_force_data:
            response.result_code.result_code = ResultCode.SERVICE_NOT_RUNNING
            response.result_code.message     = "F_ext data not available — cannot guard"
            response.data = json.dumps({
                "success": False,
                "error":   "Force data not available",
            })
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

        guard_callback = None
        if guard_enabled:
            rospy.loginfo(
                f"[move_ee_to_pose] Contact guarding ENABLED  "
                f"axes={list(monitored_axes)} thresholds={ {a: thresholds[a] for a in monitored_axes} } "
                f"frame={self.guard_force_frame}"
            )
            guard_callback = self._build_contact_guard(
                monitored_axes, thresholds,
                fallback_orientation=target_pose["orientation"],
            )

        return self._execute_trajectory_to_pose(target_pose, guard_callback=guard_callback)

    # -------------------------------------------------------------------------
    # /robot/control/move_ee_to_rel_pose
    # -------------------------------------------------------------------------

    def _parse_delta_position(self, req_json):
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
        /robot/control/move_ee_to_rel_pose — applies delta in the EE frame.

        JSON request fields:
          delta_position   (dict, required)    : {"x":..., "y":..., "z":...} in EE frame
          guard            (bool, optional)    : enable contact-guarded motion
          force_threshold  (number|dict, opt.) : N. Scalar or per-axis dict.
          guard_axes       (list, optional)    : subset of ["x","y","z"]; default all.

        On contact: result_code = SUCCESS, response.data.data.stop_reason = "contact",
        data.contact_stopped = True, plus the usual contact payload.
        """
        response = RobotCommandResponse()

        try:
            data_in   = json.loads(req.req)
            delta_ee  = self._parse_delta_position(req.req)
            guard_enabled, thresholds, monitored_axes = self._parse_guard_options(data_in)
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

        if guard_enabled and not self.has_force_data:
            response.result_code.result_code = ResultCode.SERVICE_NOT_RUNNING
            response.result_code.message     = "F_ext data not available — cannot guard"
            response.data = json.dumps({
                "success": False,
                "error":   "Force data not available",
            })
            return response

        current_pose = self._get_current_pose_dict()
        if current_pose is None:
            response.result_code.result_code = ResultCode.SERVICE_NOT_RUNNING
            response.result_code.message     = "Current EE pose unavailable"
            response.data = json.dumps({"success": False, "error": "Current EE pose unavailable"})
            return response

        delta_base = self._rotate_vector_by_quaternion(delta_ee, current_pose["orientation"])

        target_pose = {
            "position": {
                "x": current_pose["position"]["x"] + delta_base["x"],
                "y": current_pose["position"]["y"] + delta_base["y"],
                "z": current_pose["position"]["z"] + delta_base["z"],
            },
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

        guard_callback = None
        if guard_enabled:
            rospy.loginfo(
                f"[move_ee_to_rel_pose] Contact guarding ENABLED  "
                f"axes={list(monitored_axes)} thresholds={ {a: thresholds[a] for a in monitored_axes} } "
                f"frame={self.guard_force_frame}"
            )
            guard_callback = self._build_contact_guard(
                monitored_axes, thresholds,
                fallback_orientation=current_pose["orientation"],
            )

        return self._execute_trajectory_to_pose(target_pose, guard_callback=guard_callback)

    # -------------------------------------------------------------------------
    # /robot/control/move_ee_guarded
    # -------------------------------------------------------------------------

    def _parse_guarded_request(self, req_json):
        try:
            data = json.loads(req_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON: {exc}") from exc

        if "axis" not in data:
            raise ValueError("Missing required field 'axis'")
        axis = str(data["axis"]).lower().strip()
        if axis not in AXIS_TO_IDX:
            raise ValueError(f"'axis' must be 'x', 'y', or 'z'; got '{axis}'")

        if "distance" not in data:
            raise ValueError("Missing required field 'distance'")
        try:
            distance = float(data["distance"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"'distance' must be a number: {exc}") from exc
        if distance == 0.0:
            raise ValueError("'distance' must be non-zero")

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
        /robot/control/move_ee_guarded — single-axis guarded move.
        Now implemented in terms of the shared _build_contact_guard helper.
        """
        response = RobotCommandResponse()
        rospy.loginfo(f"move_ee_guarded request: {req.req}")

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

        # Build single-axis guard using the shared helper
        thresholds = dict(self.default_force_threshold)
        thresholds[axis] = force_threshold
        guard_callback = self._build_contact_guard(
            (axis,), thresholds,
            fallback_orientation=current_pose["orientation"],
        )

        traj_response = self._execute_trajectory_to_pose(
            target_pose, guard_callback=guard_callback
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
        response = RobotCommandResponse()
        try:
            gripper_open_position = 0.085
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

    def _handle_recover_from_reflex(self, req):
        """
        /robot/control/recover_from_reflex — manually clear a Franka reflex /
        error state.

        Always: freeze the equilibrium pose at the current EE position so the
        impedance controller doesn't snap back to the previous goal, then call
        the franka_control/error_recovery action. Returns success unless the
        action call itself failed.

        Useful for:
          - calling explicitly after a hard contact, instead of relying on
            the background watchdog
          - clearing USER_STOPPED state after the user-stop button is released
          - operator scripts that want a known recovery point
        """
        response = RobotQueryResponse()
        try:
            cur_pose = self._get_current_pose_dict()
            mode_before = self.latest_robot_mode

            self._freeze_equilibrium_pose_here()
            ok, msg = self._send_error_recovery(wait=True)
            self._freeze_equilibrium_pose_here(n_publishes=3, sleep_s=0.02)

            data = {
                "robot_mode_before": mode_before,
                "robot_mode_after":  self.latest_robot_mode,
                "reflex_count":      self.reflex_count,
                "current_position":  dict(cur_pose["position"]) if cur_pose else None,
                "action_message":    msg,
            }

            if ok:
                response.result_code.result_code = ResultCode.SUCCESS
                response.result_code.message     = msg
                response.data = json.dumps({"success": True, "message": msg, "data": data})
            else:
                response.result_code.result_code = ResultCode.FAILURE
                response.result_code.message     = msg
                response.data = json.dumps({"success": False, "error": msg, "data": data})
            return response

        except Exception as e:
            rospy.logerr(
                f"Unexpected error in _handle_recover_from_reflex: "
                f"{traceback.format_exc()}"
            )
            response.result_code.result_code = ResultCode.FAILURE
            response.result_code.message     = str(e)
            response.data = json.dumps({"success": False, "error": str(e)})
            return response

    def _handle_reset_robot(self, req):
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