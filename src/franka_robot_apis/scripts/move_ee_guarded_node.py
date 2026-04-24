#!/usr/bin/env python3
"""
ROS1 Noetic service: /robot/control/move_ee_guarded

Moves the Franka EE along a single axis by a given distance while monitoring
external forces. If the measured force on the requested axis exceeds a
configurable threshold, the robot stops immediately and the service returns
a CONTACT result. If the target is reached without contact the service
returns SUCCESS. If neither happens within the timeout it returns TIMEOUT.

Service type : RobotCommand.srv  (robot_api_interfaces)
Request  field: req  (string, JSON)
Response fields: result_code (ResultCode), data (string, JSON)

Example call:
  rosservice call /robot/control/move_ee_guarded \
    "req: '{\"axis\": \"z\", \"distance\": -0.05}'"

  rosservice call /robot/control/move_ee_guarded \
    "req: '{\"axis\": \"x\", \"distance\": 0.1, \"force_threshold\": 8.0}'"

JSON request fields:
  axis             (str)   : "x", "y", or "z"   [required]
  distance         (float) : signed metres to travel [required]
  force_threshold  (float) : N, per-axis magnitude that counts as contact
                             (default: ROS param ~force_threshold, fallback 10 N)

JSON response data fields:
  stop_reason  : "contact" | "completed" | "timeout" | "error"
  axis         : axis that was monitored
  stop_position: {x, y, z} EE position when motion ended
  elapsed      : seconds elapsed
  contact_force: (only on contact) force vector {x, y, z} at stop
"""

import json
import math
import rospy

from geometry_msgs.msg import PoseStamped, WrenchStamped
from franka_msgs.msg import FrankaState
from robot_api_interfaces.srv import RobotCommand, RobotCommandResponse
from robot_api_interfaces.msg import ResultCode


# ---------------------------------------------------------------------------
# Result code constants (mirrors ROS 2 sample)
# ---------------------------------------------------------------------------
RC_SUCCESS   = ResultCode.SUCCESS
RC_TIMEOUT   = ResultCode.TIMEOUT
RC_FAILURE   = ResultCode.FAILURE
RC_INVALID   = ResultCode.INVALID_INPUT
RC_NOT_READY = ResultCode.SERVICE_NOT_RUNNING

# Map axis label -> index in the force vector
AXIS_TO_FORCE_IDX = {"x": 0, "y": 1, "z": 2}


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class MoveEEGuardedNode:
    def __init__(self):
        rospy.init_node("move_ee_guarded_node")

        # ── ROS parameters ──────────────────────────────────────────────────
        self.equilibrium_pose_topic = rospy.get_param(
            "~equilibrium_pose_topic",
            "/cartesian_impedance_controller/equilibrium_pose",
        )
        self.publish_rate      = rospy.get_param("~publish_rate",      20)
        self.execution_timeout = rospy.get_param("~execution_timeout", 20.0)
        self.position_tolerance = rospy.get_param("~position_tolerance", 0.01)
        # Default force threshold (N).
        self.default_force_threshold = {
            "x": rospy.get_param("~default_force_threshold_x", 4.0),
            "y": rospy.get_param("~default_force_threshold_y", 4.0),
            "z": rospy.get_param("~default_force_threshold_z", 5.0),
        }

        # ── Publishers ───────────────────────────────────────────────────────
        self.pose_pub = rospy.Publisher(
            self.equilibrium_pose_topic, PoseStamped, queue_size=1
        )

        # ── Subscribers ──────────────────────────────────────────────────────
        self.state_sub = rospy.Subscriber(
            "/franka_state_controller/franka_states", FrankaState,
            self._state_callback, queue_size=1,
        )
        self.fext_sub = rospy.Subscriber(
            "/franka_state_controller/F_ext", WrenchStamped,
            self._fext_callback, queue_size=1,
        )

        # ── Internal state ───────────────────────────────────────────────────
        self.latest_o_tee       = None   # 16-element column-major transform
        self.has_state_data     = False

        self.latest_force       = None   # [fx, fy, fz]
        self.has_force_data     = False

        # ── Service ──────────────────────────────────────────────────────────
        self.service = rospy.Service(
            "/robot/control/move_ee_guarded",
            RobotCommand,
            self._handle_guarded_move,
        )

        rospy.loginfo(
            "MoveEEGuardedNode ready.\n"
            "  Service : /robot/control/move_ee_guarded\n"
            f"  Default force threshold : {self.default_force_threshold} N\n"
            f"  Publishing to           : {self.equilibrium_pose_topic}"
        )

    # ────────────────────────────────────────────────────────────────────────
    # Subscriber callbacks
    # ────────────────────────────────────────────────────────────────────────

    def _state_callback(self, msg: FrankaState):
        self.latest_o_tee   = msg.O_T_EE
        self.has_state_data = True

    def _fext_callback(self, msg: WrenchStamped):
        f = msg.wrench.force
        self.latest_force   = [f.x, f.y, f.z]
        self.has_force_data = True

    # ────────────────────────────────────────────────────────────────────────
    # Pose helpers and error metrics
    # ────────────────────────────────────────────────────────────────────────

    def _current_position(self):
        """Return (x, y, z) from latest O_T_EE or None."""
        if self.latest_o_tee is None:
            return None
        return (
            self.latest_o_tee[12],
            self.latest_o_tee[13],
            self.latest_o_tee[14],
        )

    def _rotation_to_quat(self, m):
        """
        Convert a 3x3 rotation matrix (row-major list of 9 floats) to
        quaternion (qx, qy, qz, qw) using Shepperd's method.
        """
        trace = m[0] + m[4] + m[8]
        if trace > 0:
            s  = math.sqrt(trace + 1.0) * 2.0
            qw = 0.25 * s
            qx = (m[7] - m[5]) / s
            qy = (m[2] - m[6]) / s
            qz = (m[3] - m[1]) / s
        elif (m[0] > m[4]) and (m[0] > m[8]):
            s  = math.sqrt(1.0 + m[0] - m[4] - m[8]) * 2.0
            qw = (m[7] - m[5]) / s
            qx = 0.25 * s
            qy = (m[1] + m[3]) / s
            qz = (m[2] + m[6]) / s
        elif m[4] > m[8]:
            s  = math.sqrt(1.0 + m[4] - m[0] - m[8]) * 2.0
            qw = (m[2] - m[6]) / s
            qx = (m[1] + m[3]) / s
            qy = 0.25 * s
            qz = (m[5] + m[7]) / s
        else:
            s  = math.sqrt(1.0 + m[8] - m[0] - m[4]) * 2.0
            qw = (m[3] - m[1]) / s
            qx = (m[2] + m[6]) / s
            qy = (m[5] + m[7]) / s
            qz = 0.25 * s
        return qx, qy, qz, qw

    def _current_pose_dict(self):
        """
        Return current EE pose as a dict or None.
        """
        if self.latest_o_tee is None:
            return None

        cx, cy, cz = (
            self.latest_o_tee[12],
            self.latest_o_tee[13],
            self.latest_o_tee[14],
        )

        rot = [
            self.latest_o_tee[0], self.latest_o_tee[1], self.latest_o_tee[2],
            self.latest_o_tee[4], self.latest_o_tee[5], self.latest_o_tee[6],
            self.latest_o_tee[8], self.latest_o_tee[9], self.latest_o_tee[10],
        ]
        qx, qy, qz, qw = self._rotation_to_quat(rot)

        return {
            "position":    {"x": cx,  "y": cy,  "z": cz},
            "orientation": {"x": qx,  "y": qy,  "z": qz,  "w": qw},
        }

    def _make_pose_stamped(self, pose_dict):
        msg = PoseStamped()
        msg.header.frame_id = "0"
        msg.header.stamp    = rospy.Time.now()
        p = pose_dict["position"]
        o = pose_dict["orientation"]
        msg.pose.position.x    = float(p["x"])
        msg.pose.position.y    = float(p["y"])
        msg.pose.position.z    = float(p["z"])
        msg.pose.orientation.x = float(o["x"])
        msg.pose.orientation.y = float(o["y"])
        msg.pose.orientation.z = float(o["z"])
        msg.pose.orientation.w = float(o["w"])
        return msg

    def _axis_error(self, target_dict, axis: str):
        """
        Signed error on the requested axis only.
        Using single-axis completion avoids false "not reached" results
        caused by small Y/Z drift that would inflate Euclidean distance.
        """
        pos = self._current_position()
        if pos is None:
            return float("inf")
        idx = AXIS_TO_FORCE_IDX[axis]
        target_val = target_dict["position"][axis]
        current_val = pos[idx]
        return abs(current_val - target_val)

    # ────────────────────────────────────────────────────────────────────────
    # Request parsing
    # ────────────────────────────────────────────────────────────────────────

    def _parse_request(self, req_json: str):
        """
        Parse and validate the JSON request string.

        Returns (axis, distance, force_threshold) or raises ValueError.
        """
        try:
            data = json.loads(req_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON: {exc}") from exc

        # ── axis ─────────────────────────────────────────────────────────────
        if "axis" not in data:
            raise ValueError("Missing required field 'axis'")
        axis = str(data["axis"]).lower().strip()
        if axis not in AXIS_TO_FORCE_IDX:
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

    # ────────────────────────────────────────────────────────────────────────
    # Guarded-move execution loop
    # ────────────────────────────────────────────────────────────────────────

    def _execute_guarded_move(self, axis: str, distance: float, force_threshold: float):
        """
        Blocking guarded-move loop.

        1.  Snapshot current pose and compute absolute target (single-axis delta,
            orientation unchanged).
        2.  Publish the target pose at self.publish_rate Hz, exactly like the
            reference _execute_move_to_pose.
        3.  On every iteration also check the F_ext force on *axis*:
              - |force_on_axis| >= force_threshold  ->  CONTACT
        4.  Position within self.position_tolerance  ->  SUCCESS (completed)
        5.  Elapsed > self.execution_timeout         ->  TIMEOUT

        Returns a fully populated RobotCommandResponse.
        """
        response = RobotCommandResponse()

        # ── snapshot current pose ────────────────────────────────────────────
        current_pose = self._current_pose_dict()
        if current_pose is None:
            response.result_code.result_code = RC_NOT_READY
            response.result_code.message = "Current EE pose unavailable"
            response.data = json.dumps({"success": False,
                                        "error": "Current EE pose unavailable"})
            return response

        # ── build target pose (delta on requested axis only) ─────────────────
        target_pose = {
            "position": dict(current_pose["position"]),   # shallow copy
            "orientation": current_pose["orientation"],   # orientation unchanged
        }
        target_pose["position"][axis] += distance

        start_pos = self._current_position()   # for logging

        rospy.loginfo(
            f"[GuardedMove] axis={axis}  distance={distance:+.4f} m  "
            f"threshold={force_threshold:.1f} N\n"
            f"  start  : ({start_pos[0]:.4f}, {start_pos[1]:.4f}, {start_pos[2]:.4f})\n"
            f"  target : ({target_pose['position']['x']:.4f}, "
            f"{target_pose['position']['y']:.4f}, "
            f"{target_pose['position']['z']:.4f})"
        )

        # ── force-axis index ─────────────────────────────────────────────────
        force_idx = AXIS_TO_FORCE_IDX[axis]

        # ── stall detection ───────────────────────────────────────────────────
        # The impedance controller may settle slightly short of the geometric
        # target (especially after a previous contact freeze repositioned the
        # equilibrium point).  If the EE hasn't moved more than stall_min_move
        # on the target axis for stall_window seconds AND the remaining
        # axis error is within stall_tolerance, treat it as "completed".
        stall_window    = rospy.get_param("~stall_window",    1.5)   # seconds
        stall_min_move  = rospy.get_param("~stall_min_move",  0.001) # metres
        stall_tolerance = rospy.get_param("~stall_tolerance", 0.03)  # metres
        stall_ref_axis  = (self._current_position() or (0.0, 0.0, 0.0))[force_idx]
        stall_ref_time  = rospy.Time.now()

        # ── main blocking loop ───────────────────────────────────────────────
        start_time = rospy.Time.now()
        rate       = rospy.Rate(self.publish_rate)

        while not rospy.is_shutdown():
            elapsed = (rospy.Time.now() - start_time).to_sec()

            # Current EE position (refreshed every iteration)
            cur_pos = self._current_position() or (0.0, 0.0, 0.0)
            stop_pos = {"x": cur_pos[0], "y": cur_pos[1], "z": cur_pos[2]}
            cur_axis_pos = cur_pos[force_idx]

            # ── timeout ──────────────────────────────────────────────────────
            if elapsed > self.execution_timeout:
                rospy.logwarn(f"[GuardedMove] TIMEOUT after {elapsed:.1f} s")
                msg = (
                    f"Guarded move timed out after {elapsed:.1f} s "
                    f"(axis='{axis}', distance={distance:+.4f} m)"
                )
                response.result_code.result_code = RC_TIMEOUT
                response.result_code.message     = msg
                response.data = json.dumps({
                    "result_code":   RC_TIMEOUT,
                    "message":       msg,
                    "data": {
                        "stop_reason":   "timeout",
                        "axis":          axis,
                        "stop_position": stop_pos,
                        "elapsed":       elapsed,
                    },
                })
                return response

            # ── contact / force guard ─────────────────────────────────────────
            if self.has_force_data and self.latest_force is not None:
                force_on_axis = abs(self.latest_force[force_idx])
                if force_on_axis >= force_threshold:
                    contact_force = {
                        "x": self.latest_force[0],
                        "y": self.latest_force[1],
                        "z": self.latest_force[2],
                    }
                    rospy.loginfo(
                        f"[GuardedMove] CONTACT on axis '{axis}': "
                        f"|F|={force_on_axis:.2f} N >= threshold {force_threshold:.1f} N "
                        f"at position ({cur_pos[0]:.4f}, {cur_pos[1]:.4f}, {cur_pos[2]:.4f})"
                    )
                    # Freeze the equilibrium point at the current EE position
                    # so the controller stops pulling.
                    stop_goal = {
                        "position":    stop_pos,
                        "orientation": current_pose["orientation"],
                    }
                    for _ in range(5):   # publish a few times to ensure receipt
                        self.pose_pub.publish(self._make_pose_stamped(stop_goal))
                        rospy.sleep(0.02)

                    stop_val = cur_axis_pos
                    msg = (
                        f"Contact detected on axis '{axis}' "
                        f"at position {stop_val:.4f} m. "
                    )
                    response.result_code.result_code = RC_SUCCESS   # contact IS a valid outcome
                    response.result_code.message     = msg
                    response.data = json.dumps({
                        "result_code":   RC_SUCCESS,
                        "message":       msg,
                        "data": {
                            "stop_reason":   "contact",
                            "axis":          axis,
                            "stop_position": stop_pos,
                            "elapsed":       elapsed,
                            "contact_force": contact_force,
                        },
                    })
                    return response

            # ── target reached (axis-only check) ─────────────────────────────
            # Only the commanded axis is checked.  Cross-axis drift (Y/Z wobble)
            # is intentionally ignored — it is not part of the requested motion.
            axis_err = self._axis_error(target_pose, axis)
            if axis_err <= self.position_tolerance:
                rospy.loginfo(
                    f"[GuardedMove] TARGET REACHED in {elapsed:.2f} s "
                    f"(axis_err={axis_err:.4f} m, no contact on axis '{axis}')"
                )
                msg = (
                    f"Guarded move completed and no contact detected "
                    f"(axis='{axis}', distance={distance:+.4f} m, {elapsed:.2f} s)"
                )
                response.result_code.result_code = RC_SUCCESS
                response.result_code.message     = msg
                response.data = json.dumps({
                    "result_code":   RC_SUCCESS,
                    "message":       msg,
                    "data": {
                        "stop_reason":   "completed",
                        "axis":          axis,
                        "stop_position": stop_pos,
                        "elapsed":       elapsed,
                    },
                })
                return response

            # ── stall detection ───────────────────────────────────────────────
            # Update stall reference if EE is still making meaningful progress.
            if abs(cur_axis_pos - stall_ref_axis) >= stall_min_move:
                stall_ref_axis = cur_axis_pos
                stall_ref_time = rospy.Time.now()

            stall_elapsed = (rospy.Time.now() - stall_ref_time).to_sec()
            if stall_elapsed >= stall_window:
                # EE has been stationary long enough — check how close we are.
                if axis_err <= stall_tolerance:
                    rospy.loginfo(
                        f"[GuardedMove] STALL->COMPLETED: EE settled within "
                        f"{axis_err:.4f} m of target (stall_tolerance={stall_tolerance} m) "
                        f"after {elapsed:.2f} s"
                    )
                    msg = (
                        f"Guarded move completed (settled) — no contact detected "
                        f"(axis='{axis}', axis_err={axis_err:.4f} m, {elapsed:.2f} s)"
                    )
                    response.result_code.result_code = RC_SUCCESS
                    response.result_code.message     = msg
                    response.data = json.dumps({
                        "result_code":   RC_SUCCESS,
                        "message":       msg,
                        "data": {
                            "stop_reason":   "completed",
                            "axis":          axis,
                            "stop_position": stop_pos,
                            "elapsed":       elapsed,
                        },
                    })
                    return response
                else:
                    # Stalled but far from target — treat as unexpected obstruction.
                    rospy.logwarn(
                        f"[GuardedMove] STALL: EE stopped {axis_err:.4f} m from target "
                        f"(stall_tolerance={stall_tolerance} m) possible obstruction"
                    )
                    stop_goal = {
                        "position":    stop_pos,
                        "orientation": current_pose["orientation"],
                    }
                    for _ in range(5):
                        self.pose_pub.publish(self._make_pose_stamped(stop_goal))
                        rospy.sleep(0.02)

                    msg = (
                        f"Guarded move stalled {axis_err:.4f} m from target on axis '{axis}' "
                        f"— possible obstruction (no force threshold exceeded)"
                    )
                    response.result_code.result_code = RC_FAILURE
                    response.result_code.message     = msg
                    response.data = json.dumps({
                        "result_code":   RC_FAILURE,
                        "message":       msg,
                        "data": {
                            "stop_reason":   "stall",
                            "axis":          axis,
                            "stop_position": stop_pos,
                            "elapsed":       elapsed,
                        },
                    })
                    return response

            # ── keep publishing the goal (mirrors reference implementation) ───
            self.pose_pub.publish(self._make_pose_stamped(target_pose))
            rate.sleep()

        # rospy shutdown
        cur_pos  = self._current_position() or (0.0, 0.0, 0.0)
        stop_pos = {"x": cur_pos[0], "y": cur_pos[1], "z": cur_pos[2]}
        response.result_code.result_code = RC_FAILURE
        response.result_code.message     = "Service interrupted (ROS shutdown)"
        response.data = json.dumps({
            "result_code": RC_FAILURE,
            "message":     "Service interrupted (ROS shutdown)",
            "data": {
                "stop_reason":   "interrupted",
                "axis":          axis,
                "stop_position": stop_pos,
            },
        })
        return response

    # ────────────────────────────────────────────────────────────────────────
    # Service handler
    # ────────────────────────────────────────────────────────────────────────

    def _handle_guarded_move(self, req):
        response = RobotCommandResponse()

        # ── readiness guards ─────────────────────────────────────────────────
        if not self.has_state_data:
            response.result_code.result_code = RC_NOT_READY
            response.result_code.message     = "Robot state not available (no /franka_states data)"
            response.data = json.dumps({
                "success": False,
                "error":   "Robot not connected",
            })
            return response

        if not self.has_force_data:
            # We can still move but cannot guard — warn and proceed only if
            # the caller accepts it; for safety, reject.
            response.result_code.result_code = RC_NOT_READY
            response.result_code.message     = "F_ext data not available — cannot guard"
            response.data = json.dumps({
                "success": False,
                "error":   "Force data not available",
            })
            return response

        # ── parse request ────────────────────────────────────────────────────
        try:
            axis, distance, force_threshold = self._parse_request(req.req)
        except ValueError as exc:
            response.result_code.result_code = RC_INVALID
            response.result_code.message     = str(exc)
            response.data = json.dumps({"success": False, "error": str(exc)})
            return response

        # ── execute ───────────────────────────────────────────────────────────
        try:
            return self._execute_guarded_move(axis, distance, force_threshold)
        except Exception as exc:
            rospy.logerr(f"[GuardedMove] Unhandled exception: {exc}")
            response.result_code.result_code = RC_FAILURE
            response.result_code.message     = str(exc)
            response.data = json.dumps({"success": False, "error": str(exc)})
            return response


# ────────────────────────────────────────────────────────────────────────────
# Entry-point
# ────────────────────────────────────────────────────────────────────────────

def main():
    try:
        node = MoveEEGuardedNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("[GuardedMove] Shutting down.")
    except Exception as exc:
        rospy.logerr(f"[GuardedMove] Fatal: {exc}")
        raise


if __name__ == "__main__":
    main()