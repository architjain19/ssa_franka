#!/usr/bin/env python3
"""
ROS1 Noetic service node: Cartesian Impedance Trajectory Executor
------------------------------------------------------------------
Service: /robot/control/execute_impedance_trajectory  (robot_api_interfaces/RobotCommand)

Loops over a list of pose waypoints (base frame). For each waypoint:
  1. Publish the target as geometry_msgs/PoseStamped on the Franka cartesian
     impedance controller's equilibrium_pose topic (republished at a fixed
     rate while waiting — most impedance controllers expect this).
  2. Monitor the current EE pose via TF (base_frame -> ee_frame) and exit
     when both position and orientation errors fall below tolerance.
  3. After reaching the pose, if the waypoint has gripper_action in
     {"open", "close"} call the /robot/control/set_gripper_width service.
     "keep" is a no-op.
  4. If a waypoint isn't reached within `waypoint_timeout_s`, fail fast
     and report which waypoint and what the residual errors were.

Request JSON (in .req field):
{
    "waypoints": [
        {
            "position_base":          [x, y, z],
            "quaternion_base_xyzw":   [x, y, z, w],
            "gripper_action":         "keep" | "open" | "close",
            "label":                  "pregrasp"   # optional, only used for logs
        },
        ...
    ],

    # All optional — defaults come from ~params
    "position_tolerance_m":      0.01,
    "orientation_tolerance_rad": 0.10,
    "waypoint_timeout_s":        15.0,
    "settle_time_s":             0.3,
    "publish_rate_hz":           50.0,
    "gripper_open_width_m":      0.085,
    "gripper_close_width_m":     0.0
}

(`trajectory_base_frame` is also accepted as a synonym for `waypoints` so
you can splice the DIFT response in directly during integration.)

Response JSON (in .data field):
{
    "status":          "success" | "error",
    "message":         "...",
    "num_waypoints":   N,
    "duration_s":      12.34,
    "waypoints": [
        {
            "index":                  0,
            "label":                  "pregrasp",
            "gripper_action":         "keep",
            "reached":                true,
            "position_error_m":       0.0042,
            "orientation_error_rad":  0.018,
            "duration_s":             2.31,
            "status":                 "reached",
            "gripper_status":         "keep"
        },
        ...
    ]
}

Usage:
    rosrun <your_pkg> execute_impedance_trajectory_service.py

    rosservice call /robot/control/execute_impedance_trajectory \\
        '{"req": "{\\"waypoints\\": [ ... ]}"}'
"""

import json
import math
import time
import traceback

import numpy as np

import rospy
import tf2_ros

from geometry_msgs.msg import PoseStamped
from robot_api_interfaces.srv import (
    RobotCommand, RobotCommandRequest, RobotCommandResponse,
)
from robot_api_interfaces.msg import ResultCode


# ---------------------------------------------------------------------------
# Main service class
# ---------------------------------------------------------------------------
class TrajectoryExecutorNode:
    """
    Streams equilibrium poses to a Cartesian impedance controller and
    coordinates gripper actions across a sequence of waypoints.
    """

    # Valid values for the "gripper_action" field on each waypoint.
    _GRIPPER_ACTIONS = ("keep", "open", "close")

    def __init__(self):
        # ------------------------------------------------------------------ #
        #  Parameters                                                          #
        # ------------------------------------------------------------------ #
        self.equilibrium_topic = rospy.get_param(
            "~equilibrium_pose_topic",
            "/cartesian_impedance_controller/equilibrium_pose",
        )
        self.gripper_service = rospy.get_param(
            "~gripper_service", "/robot/control/set_gripper_width"
        )

        # Frames used for TF feedback
        self.base_frame = rospy.get_param("~base_frame", "panda_link0")
        self.ee_frame   = rospy.get_param("~ee_frame",   "panda_EE")

        # Default tolerances / timing — each can be overridden per request
        self.default_position_tol_m    = float(rospy.get_param("~position_tolerance_m",     0.015))
        self.default_orientation_tol_r = float(rospy.get_param("~orientation_tolerance_rad", 0.15))
        self.default_waypoint_timeout  = float(rospy.get_param("~waypoint_timeout_s",       20.0))
        self.default_settle_time       = float(rospy.get_param("~settle_time_s",            0.5))
        self.default_publish_rate_hz   = float(rospy.get_param("~publish_rate_hz",          50.0))

        # Default gripper widths (meters). Franka FE3 gripper: 0.0 closed, ~0.085 open.
        self.default_gripper_open_m  = float(rospy.get_param("~gripper_open_width_m",  0.085))
        self.default_gripper_close_m = float(rospy.get_param("~gripper_close_width_m", 0.0))

        # Gripper service call timeout
        self.gripper_call_timeout = float(rospy.get_param("~gripper_call_timeout_s", 15.0))

        # Safety: if the first waypoint is more than this far from the current
        # EE pose, abort. Prevents the impedance controller from yanking the
        # arm if someone sends a stale trajectory or wrong frame. Set to <=0
        # to disable.
        self.max_initial_jump_m = float(rospy.get_param("~max_initial_jump_m", 0.7))

        # ------------------------------------------------------------------ #
        #  TF                                                                  #
        # ------------------------------------------------------------------ #
        self._tf_buffer   = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer)

        # ------------------------------------------------------------------ #
        #  Publisher (latched=False — we republish at a fixed rate)            #
        # ------------------------------------------------------------------ #
        self._pose_pub = rospy.Publisher(
            self.equilibrium_topic, PoseStamped, queue_size=1,
        )

        # ------------------------------------------------------------------ #
        #  Gripper service client (lazy-connected on first use)                #
        # ------------------------------------------------------------------ #
        self._gripper_client = None

        # ------------------------------------------------------------------ #
        #  Service                                                             #
        # ------------------------------------------------------------------ #
        self._service = rospy.Service(
            "/robot/control/execute_impedance_trajectory",
            RobotCommand,
            self._handle_request,
        )

        rospy.loginfo(
            "\nTrajectoryExecutorNode (ROS1) ready.\n"
            f"  Service        : /robot/control/execute_impedance_trajectory\n"
            f"  Equilibrium    : {self.equilibrium_topic}\n"
            f"  Gripper svc    : {self.gripper_service}\n"
            f"  Base frame     : {self.base_frame}\n"
            f"  EE frame       : {self.ee_frame}\n"
            f"  Pos tol (m)    : {self.default_position_tol_m}\n"
            f"  Ori tol (rad)  : {self.default_orientation_tol_r}\n"
            f"  Timeout (s)    : {self.default_waypoint_timeout}\n"
            f"  Publish rate Hz: {self.default_publish_rate_hz}"
        )

    # ------------------------------------------------------------------ #
    #  Gripper service helper                                              #
    # ------------------------------------------------------------------ #

    def _get_gripper_client(self):
        """Lazy-connect the gripper service proxy, with a bounded wait."""
        if self._gripper_client is None:
            rospy.loginfo(f"Waiting for gripper service '{self.gripper_service}' ...")
            rospy.wait_for_service(
                self.gripper_service, timeout=self.gripper_call_timeout,
            )
            self._gripper_client = rospy.ServiceProxy(
                self.gripper_service, RobotCommand, persistent=False,
            )
        return self._gripper_client

    def _call_gripper(self, width_m):
        """
        Call /robot/control/set_gripper_width with the given width (meters).

        Returns:
            (ok: bool, message: str)
        """
        try:
            client = self._get_gripper_client()
            req = RobotCommandRequest()
            req.req = json.dumps({"width": float(width_m)})
            resp = client(req)
            if resp.result_code.result_code != ResultCode.SUCCESS:
                return False, f"Gripper service failed: {resp.result_code.message}"
            return True, resp.result_code.message or "ok"
        except rospy.ServiceException as e:
            return False, f"Gripper ServiceException: {e}"
        except rospy.ROSException as e:
            # raised by wait_for_service on timeout
            return False, f"Gripper service unavailable: {e}"
        except Exception as e:
            return False, f"Unexpected gripper error: {e}"

    # ------------------------------------------------------------------ #
    #  Pose publishing / TF feedback                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _normalize_quat(q_xyzw):
        """Return a unit quaternion (numpy array)."""
        q = np.asarray(q_xyzw, dtype=np.float64)
        n = float(np.linalg.norm(q))
        if n < 1e-12:
            # Degenerate input — return identity to avoid NaN propagation
            return np.array([0.0, 0.0, 0.0, 1.0])
        return q / n

    def _publish_pose(self, position, quaternion_xyzw):
        """Publish a single PoseStamped on the equilibrium topic."""
        q = self._normalize_quat(quaternion_xyzw)
        msg = PoseStamped()
        msg.header.stamp    = rospy.Time.now()
        msg.header.frame_id = self.base_frame
        msg.pose.position.x = float(position[0])
        msg.pose.position.y = float(position[1])
        msg.pose.position.z = float(position[2])
        msg.pose.orientation.x = float(q[0])
        msg.pose.orientation.y = float(q[1])
        msg.pose.orientation.z = float(q[2])
        msg.pose.orientation.w = float(q[3])
        self._pose_pub.publish(msg)

    def _get_current_ee_pose(self):
        """
        Look up the current EE pose via TF.

        Returns:
            (position: np.ndarray(3,) | None,
             quaternion_xyzw: np.ndarray(4,) | None,
             error: str | None)
            Exactly one of (position+quat) and error is non-None.
        """
        try:
            tr = self._tf_buffer.lookup_transform(
                self.base_frame, self.ee_frame,
                rospy.Time(0), rospy.Duration(0.5),
            )
            pos = np.array([
                tr.transform.translation.x,
                tr.transform.translation.y,
                tr.transform.translation.z,
            ])
            quat = np.array([
                tr.transform.rotation.x,
                tr.transform.rotation.y,
                tr.transform.rotation.z,
                tr.transform.rotation.w,
            ])
            return pos, quat, None
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            return None, None, f"{type(e).__name__}: {e}"
        except Exception as e:
            return None, None, f"Unexpected TF error: {e}"

    @staticmethod
    def _quat_geodesic_angle(q1_xyzw, q2_xyzw):
        """
        Geodesic angle (radians) between two quaternions, accounting for
        the q ~ -q double cover.
        """
        d = abs(float(np.dot(q1_xyzw, q2_xyzw)))
        d = min(1.0, max(-1.0, d))
        return 2.0 * math.acos(d)

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

        def _normalize_quaternion(q):
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

        qn = _normalize_quaternion(quat)
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

    def _apply_z_shift(self, target_pos_arr, target_quart_arr, z_shift_m=0.18):
        """Return a new (pos, quat) with the given vertical shift applied."""
        _pos = np.array(target_pos_arr)
        _quat = np.array(target_quart_arr)
        shift_local = {"x": 0.0, "y": 0.0, "z": -float(z_shift_m)}
        shift_base  = self._rotate_vector_by_quaternion(
            shift_local, {"x": _quat[0], "y": _quat[1], "z": _quat[2], "w": _quat[3]}
        )
        shifted_pos = _pos + np.array([shift_base["x"], shift_base["y"], shift_base["z"]])
        return shifted_pos, _quat

    def _wait_for_waypoint(self, target_pos, target_quat,
                            position_tol, orientation_tol,
                            timeout_s, publish_rate_hz):
        """
        Republish the target pose at the given rate and watch TF until both
        position and orientation errors fall below tolerance.

        Returns:
            (reached: bool, status_msg: str,
             last_pos_err: float | None, last_ori_err: float | None)
        """
        rate = rospy.Rate(publish_rate_hz)
        deadline = rospy.Time.now() + rospy.Duration(timeout_s)

        target_pos_arr  = np.asarray(target_pos,  dtype=np.float64)
        target_quat_arr = self._normalize_quat(target_quat)
        target_pos_arr, target_quat_arr = self._apply_z_shift(
            target_pos_arr, target_quat_arr, z_shift_m=0.18,
        )

        last_pos_err = None
        last_ori_err = None
        tf_failures  = 0
        max_tf_failures = 20   # ~0.4s at 50Hz before we give up

        while not rospy.is_shutdown():
            # Republish the equilibrium pose every loop — impedance controllers
            # generally accept this safely, and one-shot publishing can be
            # lost on transient subscriber drops.
            self._publish_pose(target_pos_arr, target_quat_arr)

            cur_pos, cur_quat, tf_err = self._get_current_ee_pose()
            if tf_err is not None:
                tf_failures += 1
                if tf_failures >= max_tf_failures:
                    return (False,
                            f"TF lookup failed {tf_failures} times: {tf_err}",
                            last_pos_err, last_ori_err)
                rate.sleep()
                continue
            tf_failures = 0

            pos_err = float(np.linalg.norm(cur_pos - target_pos_arr))
            ori_err = self._quat_geodesic_angle(cur_quat, target_quat_arr)
            last_pos_err = pos_err
            last_ori_err = ori_err

            if pos_err <= position_tol and ori_err <= orientation_tol:
                return True, "reached", pos_err, ori_err

            if rospy.Time.now() >= deadline:
                return (False,
                        f"Timeout after {timeout_s:.2f}s "
                        f"(pos_err={pos_err*1000:.1f}mm, "
                        f"ori_err={math.degrees(ori_err):.2f}deg)",
                        pos_err, ori_err)

            rate.sleep()

        return False, "ROS shutdown requested", last_pos_err, last_ori_err

    # ------------------------------------------------------------------ #
    #  Validation                                                          #
    # ------------------------------------------------------------------ #

    @classmethod
    def _validate_waypoint(cls, wp, idx):
        """Return an error string if invalid, else None."""
        if not isinstance(wp, dict):
            return f"Waypoint {idx}: expected object, got {type(wp).__name__}"

        for k in ("position_base", "quaternion_base_xyzw"):
            if k not in wp:
                return f"Waypoint {idx}: missing required field '{k}'"

        pos = wp["position_base"]
        if not (isinstance(pos, (list, tuple)) and len(pos) == 3
                and all(isinstance(v, (int, float)) for v in pos)):
            return f"Waypoint {idx}: 'position_base' must be a list of 3 numbers"

        quat = wp["quaternion_base_xyzw"]
        if not (isinstance(quat, (list, tuple)) and len(quat) == 4
                and all(isinstance(v, (int, float)) for v in quat)):
            return (f"Waypoint {idx}: 'quaternion_base_xyzw' must be a list "
                    f"of 4 numbers [x, y, z, w]")
        if float(np.linalg.norm(quat)) < 1e-6:
            return f"Waypoint {idx}: 'quaternion_base_xyzw' is degenerate (zero norm)"

        gripper = wp.get("gripper_action", "keep")
        if gripper not in cls._GRIPPER_ACTIONS:
            return (f"Waypoint {idx}: invalid 'gripper_action'={gripper!r} "
                    f"(must be one of {cls._GRIPPER_ACTIONS})")

        return None

    # ------------------------------------------------------------------ #
    #  Service handler                                                     #
    # ------------------------------------------------------------------ #

    def _handle_request(self, request):
        rospy.loginfo(f"execute_impedance_trajectory request received ({len(request.req)} bytes)")
        response = RobotCommandResponse()

        # --- 1. Parse request -------------------------------------------
        try:
            req_data = json.loads(request.req)
        except (json.JSONDecodeError, ValueError) as e:
            return self._fail(response, f"Bad request (not valid JSON): {e}")

        # Accept either 'waypoints' or 'trajectory_base_frame' (DIFT-shaped).
        waypoints = req_data.get("waypoints")
        if waypoints is None:
            waypoints = req_data.get("trajectory_base_frame")
        if not isinstance(waypoints, list) or len(waypoints) == 0:
            return self._fail(
                response,
                "Missing or empty 'waypoints' (or 'trajectory_base_frame') in request.",
            )

        # --- 2. Validate every waypoint upfront -------------------------
        for i, wp in enumerate(waypoints):
            err = self._validate_waypoint(wp, i)
            if err:
                return self._fail(response, err)

        # --- 3. Per-request options -------------------------------------
        position_tol     = float(req_data.get("position_tolerance_m",     self.default_position_tol_m))
        orientation_tol  = float(req_data.get("orientation_tolerance_rad", self.default_orientation_tol_r))
        waypoint_timeout = float(req_data.get("waypoint_timeout_s",       self.default_waypoint_timeout))
        settle_time      = float(req_data.get("settle_time_s",            self.default_settle_time))
        publish_rate_hz  = float(req_data.get("publish_rate_hz",          self.default_publish_rate_hz))
        gripper_open_w   = float(req_data.get("gripper_open_width_m",     self.default_gripper_open_m))
        gripper_close_w  = float(req_data.get("gripper_close_width_m",    self.default_gripper_close_m))

        if publish_rate_hz <= 0.0:
            return self._fail(response, "'publish_rate_hz' must be > 0.")
        if waypoint_timeout <= 0.0:
            return self._fail(response, "'waypoint_timeout_s' must be > 0.")

        rospy.loginfo(
            f"Executing {len(waypoints)} waypoints | "
            f"pos_tol={position_tol*1000:.1f}mm "
            f"ori_tol={math.degrees(orientation_tol):.1f}deg "
            f"timeout={waypoint_timeout:.1f}s "
            f"settle={settle_time:.2f}s "
            f"rate={publish_rate_hz:.0f}Hz"
        )

        # --- 4. Give TF + publisher a moment to wire up -----------------
        rospy.sleep(0.2)

        # --- 5. Safety: check the initial jump from current EE pose -----
        first_pos = np.asarray(waypoints[0]["position_base"], dtype=np.float64)
        cur_pos, _, tf_err = self._get_current_ee_pose()
        if tf_err is not None:
            return self._fail(
                response,
                f"Cannot read current EE pose from TF "
                f"({self.base_frame} -> {self.ee_frame}): {tf_err}",
            )
        initial_jump = float(np.linalg.norm(cur_pos - first_pos))
        rospy.loginfo(
            f"Current EE at ({cur_pos[0]:.3f}, {cur_pos[1]:.3f}, {cur_pos[2]:.3f}); "
            f"first waypoint is {initial_jump*1000:.1f}mm away."
        )
        if self.max_initial_jump_m > 0.0 and initial_jump > self.max_initial_jump_m:
            return self._fail(
                response,
                f"Initial jump from current EE pose to first waypoint is "
                f"{initial_jump:.3f}m, which exceeds the safety limit "
                f"max_initial_jump_m={self.max_initial_jump_m:.3f}m. "
                f"Aborting before publishing equilibrium pose to avoid a "
                f"large impedance reaction.",
            )

        # --- 6. Execute waypoints one by one ----------------------------
        executed = []
        t_start  = time.time()

        for i, wp in enumerate(waypoints):
            pos     = wp["position_base"]
            quat    = wp["quaternion_base_xyzw"]
            gripper = wp.get("gripper_action", "keep")
            label   = wp.get("label", "")

            rospy.loginfo(
                f"[{i+1}/{len(waypoints)}] label='{label}' "
                f"pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) "
                f"gripper={gripper}"
            )

            t_wp = time.time()
            ok, status_msg, pos_err, ori_err = self._wait_for_waypoint(
                target_pos=pos,
                target_quat=quat,
                position_tol=position_tol,
                orientation_tol=orientation_tol,
                timeout_s=waypoint_timeout,
                publish_rate_hz=publish_rate_hz,
            )
            dt_wp = time.time() - t_wp

            wp_record = {
                "index":                 i,
                "label":                 label,
                "gripper_action":        gripper,
                "reached":               bool(ok),
                "position_error_m":      pos_err,
                "orientation_error_rad": ori_err,
                "duration_s":            round(dt_wp, 3),
                "status":                status_msg,
                "gripper_status":        None,
            }

            if not ok:
                rospy.logerr(
                    f"[{i+1}/{len(waypoints)}] '{label}' failed: {status_msg}"
                )
                wp_record["gripper_status"] = "skipped (waypoint failed)"
                executed.append(wp_record)
                return self._fail_with_partial(
                    response,
                    f"Waypoint {i} '{label}' failed: {status_msg}",
                    executed, t_start,
                )

            rospy.loginfo(
                f"[{i+1}/{len(waypoints)}] reached in {dt_wp:.2f}s "
                f"(pos_err={pos_err*1000:.1f}mm, "
                f"ori_err={math.degrees(ori_err):.2f}deg)"
            )

            # Brief settle (impedance can overshoot slightly on arrival)
            if settle_time > 0.0:
                rospy.sleep(settle_time)

            # Gripper action runs AFTER the waypoint pose is reached.
            if gripper == "close":
                rospy.loginfo(f"  -> gripper close ({gripper_close_w*1000:.0f}mm)")
                g_ok, g_msg = self._call_gripper(gripper_close_w)
            elif gripper == "open":
                rospy.loginfo(f"  -> gripper open ({gripper_open_w*1000:.0f}mm)")
                g_ok, g_msg = self._call_gripper(gripper_open_w)
            else:
                g_ok, g_msg = True, "keep"

            wp_record["gripper_status"] = g_msg
            executed.append(wp_record)

            if not g_ok:
                rospy.logerr(
                    f"[{i+1}/{len(waypoints)}] '{label}' gripper {gripper} failed: {g_msg}"
                )
                return self._fail_with_partial(
                    response,
                    f"Waypoint {i} '{label}' gripper {gripper} failed: {g_msg}",
                    executed, t_start,
                )

        # --- 7. Build success response ----------------------------------
        total_dt = time.time() - t_start
        payload = {
            "status":        "success",
            "message":       f"Executed {len(executed)} waypoints in {total_dt:.2f}s.",
            "num_waypoints": len(executed),
            "duration_s":    round(total_dt, 3),
            "waypoints":     executed,
        }

        response.result_code.result_code = ResultCode.SUCCESS
        response.result_code.message     = f"Executed {len(executed)} waypoints."
        response.data                    = json.dumps(payload)

        rospy.loginfo(f"execute_impedance_trajectory done: {payload['message']}")
        return response

    # ------------------------------------------------------------------ #
    #  Response helpers                                                    #
    # ------------------------------------------------------------------ #

    def _fail(self, response, msg):
        """Populate *response* as a hard failure with no partial progress."""
        rospy.logerr(f"execute_impedance_trajectory error: {msg}")
        response.result_code.result_code = ResultCode.FAILURE
        response.result_code.message     = msg
        response.data = json.dumps({"status": "error", "message": msg})
        return response

    def _fail_with_partial(self, response, msg, executed, t_start):
        """Populate *response* as a failure but include the per-waypoint log."""
        rospy.logerr(f"execute_impedance_trajectory error: {msg}")
        response.result_code.result_code = ResultCode.FAILURE
        response.result_code.message     = msg
        response.data = json.dumps({
            "status":                  "error",
            "message":                 msg,
            "num_waypoints_attempted": len(executed),
            "duration_s":              round(time.time() - t_start, 3),
            "waypoints":               executed,
        })
        return response

    # ------------------------------------------------------------------ #
    #  Spin                                                                #
    # ------------------------------------------------------------------ #

    def spin(self):
        rospy.spin()


def main():
    rospy.init_node("trajectory_executor_service", anonymous=False)
    try:
        rospy.loginfo("Creating TrajectoryExecutorNode ...")
        node = TrajectoryExecutorNode()
        rospy.loginfo("TrajectoryExecutorNode spinning ...")
        node.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS interrupt — shutting down TrajectoryExecutorNode.")
    except Exception as e:
        rospy.logerr(f"Fatal error: {e}\n{traceback.format_exc()}")
    finally:
        rospy.loginfo("TrajectoryExecutorNode shutdown complete.")


if __name__ == "__main__":
    main()