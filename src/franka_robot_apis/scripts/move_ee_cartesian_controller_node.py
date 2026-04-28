#!/usr/bin/env python3
"""
ROS1 services that move robot EE to target pose and BLOCK until completion.

Services:
  /robot/control/move_ee_to_pose     - Move to absolute pose (WebSocket trajectory)
  /robot/control/move_ee_to_rel_pose - Move by delta position (orientation unchanged)
  /robot/control/reset_robot         - Return to home pose

Type: RobotCommand.srv / RobotQuery.srv

Example service requests:
  rosservice call /robot/control/move_ee_to_pose \
    "req: '{\"target_pose\": {\"position\": {\"x\": 0.5, \"y\": 0.0, \"z\": 0.5}, \
    \"orientation\": {\"x\": 0.8722, \"y\": -0.4867, \"z\": -0.0424, \"w\": 0.0264}}}'"

  rosservice call /robot/control/move_ee_to_rel_pose \
    "req: '{\"delta_position\": {\"x\": 0.0, \"y\": 0.1, \"z\": 0.3}}'"

  rosservice call /robot/control/reset_robot "{}"
"""

import json
import math
import time
import asyncio
import threading
import traceback
import rospy
from geometry_msgs.msg import PoseStamped
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

        # Service timeouts
        self.ee_pose_svc_timeout      = float(rospy.get_param("~ee_pose_svc_timeout", 5.0))
        self.current_joints_svc_timeout = float(rospy.get_param("~current_joints_svc_timeout", 5.0))

        # Publisher
        self.pose_pub = rospy.Publisher(
            self.equilibrium_pose_topic, PoseStamped, queue_size=1)

        # Subscriber for monitoring
        self.robot_state_sub = rospy.Subscriber(
            "/franka_state_controller/franka_states", FrankaState,
            self._robot_state_callback, queue_size=1)

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

        # ===== Shared State =====
        self.latest_o_tee    = None
        self.has_received_data = False

        # One motion at a time — used by both absolute and relative services
        self._traj_lock = threading.Lock()

        self.reset_robot_pose_config = {
            "position":    {"x": 0.5, "y": 0.0, "z": 0.5},
            "orientation": {"x": 0.8722, "y": -0.4867, "z": -0.0424, "w": 0.0264},
        }

        # ===== ROS Services =====
        rospy.Service("/robot/control/move_ee_to_pose",     RobotCommand, self._handle_move_ee_to_pose)
        rospy.Service("/robot/control/move_ee_to_rel_pose", RobotCommand, self._handle_move_ee_to_rel_pose)
        rospy.Service("/robot/control/reset_robot",         RobotQuery,   self._handle_reset_robot)

        rospy.loginfo(
            f"MoveEEControllerNode ready.\n"
            f"  /robot/control/move_ee_to_pose\n"
            f"  /robot/control/move_ee_to_rel_pose\n"
            f"  /robot/control/reset_robot\n"
            f"  equilibrium_pose_topic : {self.equilibrium_pose_topic}\n"
            f"  ws                     : ws://{self.ws_host}:{self.ws_port}/ws\n"
            f"  time_scale             : {self.time_scale}x\n"
            f"  position_tolerance     : {self.position_tolerance} m\n"
            f"  position_jump_tol      : {self.position_jump_tolerance} m\n"
            f"  websockets             : {'OK' if _WEBSOCKETS_OK else 'MISSING - pip install aiohttp'}"
        )

    # =========================================================================
    # Shared helpers
    # =========================================================================

    def _robot_state_callback(self, msg):
        self.latest_o_tee     = msg.O_T_EE
        self.has_received_data = True

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

        # Column-major → row-major rotation sub-matrix
        r = [
            self.latest_o_tee[0], self.latest_o_tee[1], self.latest_o_tee[2],
            self.latest_o_tee[4], self.latest_o_tee[5], self.latest_o_tee[6],
            self.latest_o_tee[8], self.latest_o_tee[9], self.latest_o_tee[10],
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

    # =========================================================================
    # WebSocket & Trajectory Methods
    # =========================================================================

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
    # Core blocking movement — used by rel_pose and reset
    # =========================================================================

    def _execute_move_to_pose(self, target_pose):
        """
        Publish *target_pose* continuously until the EE arrives within tolerance
        or the timeout expires. Acquires _traj_lock to prevent concurrent moves.
        Returns a RobotCommandResponse.
        """
        response = RobotCommandResponse()

        # FIX: acquire lock so this cannot run concurrently with WebSocket trajectory
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

    def _wait_for_ee_convergence(self, target_pose):
        """
        Poll the EE pose service until position error is within tolerance.

        Called immediately after all waypoints have been published (publishing
        itself already consumed the trajectory duration), so this method does
        NOT add an extra sleep for that duration. Only a short traj_buffer is
        waited before starting to poll, to let the controller begin tracking.

        FIX: removed the redundant `time.sleep(traj_duration)` that previously
        caused an idle wait equal to the entire trajectory duration after all
        waypoints had already been sent.

        Returns (reached: bool, elapsed: float)
        """
        t_start    = time.time()
        timeout_at = t_start + self.traj_buffer + self.ee_convergence_timeout
        poll_rate  = rospy.Rate(10)

        while time.time() < timeout_at and not rospy.is_shutdown():
            current_pose = self._get_ee_pose()

            if current_pose is not None:
                dx = current_pose["position"]["x"] - target_pose["position"]["x"]
                dy = current_pose["position"]["y"] - target_pose["position"]["y"]
                dz = current_pose["position"]["z"] - target_pose["position"]["z"]
                pos_error = math.sqrt(dx**2 + dy**2 + dz**2)

                if pos_error <= self.position_tolerance:
                    elapsed = time.time() - t_start
                    rospy.loginfo(f"EE converged in {elapsed:.2f}s (error: {pos_error:.4f}m)")
                    return True, elapsed
            else:
                rospy.logwarn("Failed to get current EE pose - retrying")

            poll_rate.sleep()

        return False, time.time() - t_start

    # =========================================================================
    # Service Handlers
    # =========================================================================

    def _parse_target_pose(self, req_json):
        data = json.loads(req_json)
        if "target_pose" not in data:
            raise ValueError("Missing 'target_pose'")
        tp = data["target_pose"]
        for k in ["x", "y", "z"]:
            if k not in tp.get("position", {}):
                raise ValueError(f"Missing '{k}' in position")
            if k not in tp.get("orientation", {}):
                raise ValueError(f"Missing '{k}' in orientation")
        if "w" not in tp.get("orientation", {}):
            raise ValueError("Missing 'w' in orientation")
        return tp

    def _handle_move_ee_to_pose(self, req):
        """
        /robot/control/move_ee_to_pose — absolute pose via WebSocket trajectory.

        Flow:
          1. Parse & validate target pose
          2. Get current joint state & EE pose
          3. Send to motion server via WebSocket
          4. Parse + safety-check trajectory
          5. Publish waypoints respecting scaled timing
          6. Poll for EE convergence (no redundant idle wait)
        """
        response = RobotCommandResponse()
        rospy.loginfo(f"move_ee_to_pose request: {req.req}")

        if not self._traj_lock.acquire(blocking=False):
            response.result_code.result_code = ResultCode.FAILURE
            response.result_code.message     = "Another motion is already in progress."
            response.data = json.dumps({"success": False, "error": "Motion in progress"})
            return response

        try:
            t_total_start = time.time()  # wall-clock start for the entire service call

            # ----------------------------------------------------------------
            # Step 1: Parse and validate target pose
            # ----------------------------------------------------------------
            try:
                target_pose = self._parse_target_pose(req.req)
                for a in ["x", "y", "z"]:
                    if not isinstance(target_pose["position"][a], (int, float)):
                        raise ValueError(f"Invalid position {a}")
                for a in ["x", "y", "z", "w"]:
                    if not isinstance(target_pose["orientation"][a], (int, float)):
                        raise ValueError(f"Invalid orientation {a}")
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                response.result_code.result_code = ResultCode.INVALID_INPUT
                response.result_code.message     = f"Bad request: {e}"
                response.data = json.dumps({"success": False, "error": str(e)})
                return response

            if not self.has_received_data:
                response.result_code.result_code = ResultCode.SERVICE_NOT_RUNNING
                response.result_code.message     = "Robot not connected (no state data received)"
                response.data = json.dumps({"success": False, "error": "Robot not connected"})
                return response

            # ----------------------------------------------------------------
            # Step 2: Get current state
            # ----------------------------------------------------------------
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

            # ----------------------------------------------------------------
            # Step 3: Sanity-check target distance
            # ----------------------------------------------------------------
            dx = current_ee_pose["position"]["x"] - target_pose["position"]["x"]
            dy = current_ee_pose["position"]["y"] - target_pose["position"]["y"]
            dz = current_ee_pose["position"]["z"] - target_pose["position"]["z"]
            dist_to_target = math.sqrt(dx**2 + dy**2 + dz**2)

            if dist_to_target > 2.0:
                response.result_code.result_code = ResultCode.INVALID_INPUT
                response.result_code.message     = (
                    f"Target too far ({dist_to_target:.2f}m). Max ~2.0m per move."
                )
                response.data = json.dumps({"success": False, "error": "Target distance too large"})
                return response

            # ----------------------------------------------------------------
            # Step 4: WebSocket request
            # ----------------------------------------------------------------
            ws_msg = self._build_ws_message(current_joints, target_pose)
            rospy.loginfo(f"Sending to WebSocket: {ws_msg[:100]}...")
            ws_raw, ws_err = self._send_ws(ws_msg)

            if ws_err:
                response.result_code.result_code = ResultCode.FAILURE
                response.result_code.message     = f"WebSocket error: {ws_err}"
                response.data = json.dumps({"success": False, "error": f"WebSocket: {ws_err}"})
                return response

            # ----------------------------------------------------------------
            # Step 5: Parse trajectory
            # ----------------------------------------------------------------
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

            # ----------------------------------------------------------------
            # Step 6: Safety checks
            # ----------------------------------------------------------------
            safe, safety_msg = self._check_waypoint_safety(waypoints)
            if not safe:
                response.result_code.result_code = ResultCode.FAILURE
                response.result_code.message     = f"Safety check failed: {safety_msg}"
                response.data = json.dumps({"success": False, "error": safety_msg})
                return response

            rospy.loginfo("Safety check: position jump detection OK")

            # ----------------------------------------------------------------
            # Step 7: Publish waypoints with scaled timing
            # ----------------------------------------------------------------
            raw_dt     = float(traj_data.get("metadata", {}).get("dt", 0.02))
            scaled_dt  = raw_dt * self.time_scale
            traj_duration = (len(waypoints) - 1) * scaled_dt

            rospy.loginfo(
                f"Trajectory: raw_dt={raw_dt:.3f}s, scaled_dt={scaled_dt:.3f}s, "
                f"duration={traj_duration:.2f}s ({self.time_scale}x)"
            )

            t_start_pub = time.time()

            for idx, waypoint in enumerate(waypoints):
                desired_time = t_start_pub + idx * scaled_dt
                sleep_time   = desired_time - time.time()
                if sleep_time > 0:
                    time.sleep(sleep_time)

                try:
                    pos = waypoint.get("position", [0, 0, 0])
                    ori = waypoint.get("orientation", [
                        target_pose["orientation"]["x"],
                        target_pose["orientation"]["y"],
                        target_pose["orientation"]["z"],
                        target_pose["orientation"]["w"],
                    ])

                    # orientation from server is [w, x, y, z] as a list
                    if isinstance(ori, (list, tuple)) and len(ori) == 4:
                        ori_dict = {
                            "x": float(ori[1]),
                            "y": float(ori[2]),
                            "z": float(ori[3]),
                            "w": float(ori[0]),
                        }
                    else:
                        ori_dict = target_pose["orientation"]

                    pose_dict = {
                        "position": {
                            "x": float(pos[0][0]),
                            "y": float(pos[0][1]),
                            "z": float(pos[0][2]),
                        },
                        "orientation": ori_dict,
                    }

                    self.pose_pub.publish(self._create_pose_stamped(pose_dict))

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

            rospy.loginfo(
                f"Finished publishing {len(waypoints)} waypoints in "
                f"{time.time() - t_start_pub:.2f}s"
            )

            # ----------------------------------------------------------------
            # Step 8: Poll for convergence — no extra idle sleep here.
            # Publishing has already consumed traj_duration; we check immediately.
            # ----------------------------------------------------------------
            reached, convergence_elapsed = self._wait_for_ee_convergence(target_pose)
            total_elapsed = time.time() - t_total_start

            if reached:
                response.result_code.result_code = ResultCode.SUCCESS
                response.result_code.message     = "Trajectory execution completed successfully"
                response.data = json.dumps({
                    "success": True,
                    "message": "Trajectory execution completed successfully",
                    "elapsed_time": round(total_elapsed, 3),
                    "convergence_time": round(convergence_elapsed, 3),
                })
                rospy.loginfo(
                    f"move_ee_to_pose SUCCESS — total={total_elapsed:.2f}s "
                    f"(convergence poll={convergence_elapsed:.2f}s)"
                )
            else:
                response.result_code.result_code = ResultCode.TIMEOUT
                response.result_code.message     = (
                    f"EE did not converge within "
                    f"{self.traj_buffer + self.ee_convergence_timeout:.1f}s"
                )
                response.data = json.dumps({
                    "success": False,
                    "error": "Convergence timeout",
                    "elapsed_time": round(total_elapsed, 3),
                    "convergence_time": round(convergence_elapsed, 3),
                })
                rospy.logwarn(f"move_ee_to_pose TIMEOUT — total={total_elapsed:.2f}s")

            return response

        except Exception as e:
            rospy.logerr(f"Unexpected error in move_ee_to_pose: {traceback.format_exc()}")
            response.result_code.result_code = ResultCode.FAILURE
            response.result_code.message     = f"Unexpected error: {e}"
            response.data = json.dumps({"success": False, "error": str(e)})
            return response

        finally:
            self._traj_lock.release()

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
        /robot/control/move_ee_to_rel_pose — applies delta to current position,
        keeps orientation unchanged, then delegates to _execute_move_to_pose.
        """
        response = RobotCommandResponse()
        try:
            delta = self._parse_delta_position(req.req)

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

            target_pose = {
                "position": {
                    "x": current_pose["position"]["x"] + float(delta["x"]),
                    "y": current_pose["position"]["y"] + float(delta["y"]),
                    "z": current_pose["position"]["z"] + float(delta["z"]),
                },
                "orientation": current_pose["orientation"],
            }

            rospy.loginfo(
                f"Relative move — delta: ({delta['x']:.3f}, {delta['y']:.3f}, {delta['z']:.3f}) | "
                f"current: ({current_pose['position']['x']:.3f}, "
                f"{current_pose['position']['y']:.3f}, "
                f"{current_pose['position']['z']:.3f}) | "
                f"target: ({target_pose['position']['x']:.3f}, "
                f"{target_pose['position']['y']:.3f}, "
                f"{target_pose['position']['z']:.3f})"
            )

            return self._execute_move_to_pose(target_pose)

        except ValueError as e:
            response.result_code.result_code = ResultCode.INVALID_INPUT
            response.result_code.message     = str(e)
            response.data = json.dumps({"success": False, "error": str(e)})
        except Exception as e:
            response.result_code.result_code = ResultCode.FAILURE
            response.result_code.message     = str(e)
            response.data = json.dumps({"success": False, "error": str(e)})

        return response

    # -------------------------------------------------------------------------
    # /robot/control/reset_robot
    # -------------------------------------------------------------------------

    def _handle_reset_robot(self, req):
        """
        /robot/control/reset_robot — move to hard-coded home pose.
        """
        response = RobotQueryResponse()
        try:
            reset_pose = self.reset_robot_pose_config

            if not self.has_received_data:
                response.result_code.result_code = ResultCode.SERVICE_NOT_RUNNING
                response.result_code.message     = "Robot not connected"
                response.data = json.dumps({"success": False, "error": "Robot not connected"})
                return response

            cmd_res = self._execute_move_to_pose(reset_pose)
            # Convert RobotCommandResponse → RobotQueryResponse
            response.result_code = cmd_res.result_code
            response.data        = cmd_res.data
            return response

        except Exception as e:
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