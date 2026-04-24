#!/usr/bin/env python3
"""
ROS1 services that move robot EE to target pose and BLOCK until completion.

Services:
  /robot/control/move_ee_to_pose     - Move to absolute pose
  /robot/control/move_ee_to_rel_pose - Move by delta position (orientation unchanged)

Type: RobotCommand.srv

Example service requests:
  rosservice call /robot/control/move_ee_to_pose \
    "req: '{\"target_pose\": {\"position\": {\"x\": 0.5, \"y\": 0.0, \"z\": 0.5}, \
    \"orientation\": {\"x\": 0.8722, \"y\": -0.4867, \"z\": -0.0424, \"w\": 0.0264}}}'"

  rosservice call /robot/control/move_ee_to_rel_pose \
    "req: '{\"delta_position\": {\"x\": 0.0, \"y\": 0.1, \"z\": 0.3}}'"

  rosservice call /robot/control/reset_robot "{}" 

Features:
- Blocks until robot reaches target (within tolerance)
- Monitors progress via /franka_state_controller/franka_states
- Returns success/failure with timing info
"""

import json
import math
import rospy
from geometry_msgs.msg import PoseStamped
from robot_api_interfaces.srv import RobotCommand, RobotCommandResponse, RobotQuery, RobotQueryResponse
from robot_api_interfaces.msg import ResultCode
from franka_msgs.msg import FrankaState


class MoveEEControllerNode:
    def __init__(self):
        rospy.init_node("move_ee_controller_node")

        # Parameters
        self.equilibrium_pose_topic = rospy.get_param(
            "~equilibrium_pose_topic",
            "/cartesian_impedance_controller/equilibrium_pose"
        )
        self.publish_rate = rospy.get_param("~publish_rate", 20)
        self.execution_timeout = rospy.get_param("~execution_timeout", 20.0)
        self.position_tolerance = rospy.get_param("~position_tolerance", 0.01)
        self.orientation_tolerance = rospy.get_param("~orientation_tolerance", 0.05)

        # Publisher
        self.pose_pub = rospy.Publisher(
            self.equilibrium_pose_topic, PoseStamped, queue_size=1)

        # Subscriber for monitoring
        self.robot_state_sub = rospy.Subscriber(
            "/franka_state_controller/franka_states", FrankaState,
            self._robot_state_callback, queue_size=1)

        # State
        self.latest_o_tee = None
        self.has_received_data = False
        self.is_moving = False
        self.target_pose = None
        self.movement_start_time = None
        self.reset_robot_pose_config = {
            "position": {"x": 0.5, "y": 0.0, "z": 0.5},
            "orientation": {"x": 0.8722, "y": -0.4867, "z": -0.0424, "w": 0.0264},
        }

        # Services
        self.move_to_pose_service = rospy.Service(
            "/robot/control/move_ee_to_pose", RobotCommand,
            self._handle_move_ee_to_pose)

        self.move_to_rel_pose_service = rospy.Service(
            "/robot/control/move_ee_to_rel_pose", RobotCommand,
            self._handle_move_ee_to_rel_pose)
        
        self.reset_robot_service = rospy.Service(
            "/robot/control/reset_robot", RobotQuery,
            self._handle_reset_robot)

        rospy.loginfo(
            f"Services initialized. Publishing to: {self.equilibrium_pose_topic}\n"
            f"  /robot/control/move_ee_to_pose\n"
            f"  /robot/control/move_ee_to_rel_pose\n"
            f"  /robot/control/reset_robot"
        )

    # -------------------------------------------------------------------------
    # Shared helpers
    # -------------------------------------------------------------------------

    def _robot_state_callback(self, msg):
        self.latest_o_tee = msg.O_T_EE
        self.has_received_data = True

    def _create_pose_stamped(self, pose_dict):
        msg = PoseStamped()
        msg.header.frame_id = "0"
        msg.header.stamp = rospy.Time.now()
        msg.pose.position.x = float(pose_dict["position"]["x"])
        msg.pose.position.y = float(pose_dict["position"]["y"])
        msg.pose.position.z = float(pose_dict["position"]["z"])
        msg.pose.orientation.x = float(pose_dict["orientation"]["x"])
        msg.pose.orientation.y = float(pose_dict["orientation"]["y"])
        msg.pose.orientation.z = float(pose_dict["orientation"]["z"])
        msg.pose.orientation.w = float(pose_dict["orientation"]["w"])
        return msg

    def _get_current_pose_dict(self):
        """
        Return current EE pose as a dict compatible with pose_dict convention,
        or None if robot state has not been received yet.

        Reads position from O_T_EE columns 12-14 and converts the 3x3 rotation
        sub-matrix to a quaternion.
        """
        if self.latest_o_tee is None:
            return None

        # Position: column-major 4x4 → elements [12], [13], [14]
        cx = self.latest_o_tee[12]
        cy = self.latest_o_tee[13]
        cz = self.latest_o_tee[14]

        # Rotation sub-matrix (column-major → row-major indexing)
        r = [
            self.latest_o_tee[0], self.latest_o_tee[1], self.latest_o_tee[2],
            self.latest_o_tee[4], self.latest_o_tee[5], self.latest_o_tee[6],
            self.latest_o_tee[8], self.latest_o_tee[9], self.latest_o_tee[10],
        ]

        # Shepperd's method for rotation matrix → quaternion
        trace = r[0] + r[4] + r[8]
        if trace > 0:
            s = math.sqrt(trace + 1.0) * 2.0          # s = 4*qw
            qw = 0.25 * s
            qx = (r[7] - r[5]) / s
            qy = (r[2] - r[6]) / s
            qz = (r[3] - r[1]) / s
        elif (r[0] > r[4]) and (r[0] > r[8]):
            s = math.sqrt(1.0 + r[0] - r[4] - r[8]) * 2.0  # s = 4*qx
            qw = (r[7] - r[5]) / s
            qx = 0.25 * s
            qy = (r[1] + r[3]) / s
            qz = (r[2] + r[6]) / s
        elif r[4] > r[8]:
            s = math.sqrt(1.0 + r[4] - r[0] - r[8]) * 2.0  # s = 4*qy
            qw = (r[2] - r[6]) / s
            qx = (r[1] + r[3]) / s
            qy = 0.25 * s
            qz = (r[5] + r[7]) / s
        else:
            s = math.sqrt(1.0 + r[8] - r[0] - r[4]) * 2.0  # s = 4*qz
            qw = (r[3] - r[1]) / s
            qx = (r[2] + r[6]) / s
            qy = (r[5] + r[7]) / s
            qz = 0.25 * s

        return {
            "position":    {"x": cx,  "y": cy,  "z": cz},
            "orientation": {"x": qx,  "y": qy,  "z": qz,  "w": qw},
        }

    def _check_at_target(self, target):
        """Return True when position error is within tolerance."""
        if self.latest_o_tee is None:
            return False
        cx = self.latest_o_tee[12]
        cy = self.latest_o_tee[13]
        cz = self.latest_o_tee[14]
        tx = target["position"]["x"]
        ty = target["position"]["y"]
        tz = target["position"]["z"]
        pos_dist = math.sqrt((cx - tx) ** 2 + (cy - ty) ** 2 + (cz - tz) ** 2)
        return pos_dist <= self.position_tolerance

    def _execute_move_to_pose(self, target_pose):
        """
        Core blocking movement loop shared by both services.

        Publishes *target_pose* at self.publish_rate until the EE arrives
        within tolerance or the timeout expires.

        Returns a fully populated RobotCommandResponse.
        """
        response = RobotCommandResponse()

        self.target_pose = target_pose
        self.is_moving = True
        self.movement_start_time = rospy.Time.now()

        tx = target_pose["position"]["x"]
        ty = target_pose["position"]["y"]
        tz = target_pose["position"]["z"]
        rospy.loginfo(f"Moving to: ({tx:.3f}, {ty:.3f}, {tz:.3f})")

        rate = rospy.Rate(self.publish_rate)

        while self.is_moving and not rospy.is_shutdown():
            elapsed = (rospy.Time.now() - self.movement_start_time).to_sec()

            # Timeout guard
            if elapsed > self.execution_timeout:
                self.is_moving = False
                self.target_pose = None
                response.result_code.result_code = ResultCode.TIMEOUT
                response.result_code.message = f"Timeout after {elapsed:.1f}s"
                response.data = json.dumps({
                    "success": False,
                    "error": "Timeout",
                    "elapsed": elapsed,
                })
                return response

            # Success check
            if self._check_at_target(target_pose):
                self.is_moving = False
                self.target_pose = None
                rospy.loginfo(f"Reached target in {elapsed:.1f}s")
                response.result_code.result_code = ResultCode.SUCCESS
                response.result_code.message = "Trajectory execution completed successfully"
                response.data = json.dumps({
                    "success": True,
                    "message": "Trajectory execution completed successfully",
                    "elapsed_time": elapsed,
                })
                return response

            # Keep publishing the goal
            self.pose_pub.publish(self._create_pose_stamped(target_pose))
            rate.sleep()

        # Loop exited due to rospy shutdown or external flag
        self.target_pose = None
        self.is_moving = False
        response.result_code.result_code = ResultCode.FAILURE
        response.result_code.message = "Service interrupted"
        response.data = json.dumps({"success": False, "error": "Interrupted"})
        return response

    # -------------------------------------------------------------------------
    # /robot/control/move_ee_to_pose  (absolute pose)
    # -------------------------------------------------------------------------

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
        response = RobotCommandResponse()

        try:
            target_pose = self._parse_target_pose(req.req)

            # Type validation
            for a in ["x", "y", "z"]:
                if not isinstance(target_pose["position"][a], (int, float)):
                    raise ValueError(f"Invalid position {a}")
            for a in ["x", "y", "z", "w"]:
                if not isinstance(target_pose["orientation"][a], (int, float)):
                    raise ValueError(f"Invalid orientation {a}")

            if not self.has_received_data:
                response.result_code.result_code = ResultCode.SERVICE_NOT_RUNNING
                response.result_code.message = "Robot not connected"
                response.data = json.dumps({"success": False, "error": "Robot not connected"})
                return response

            return self._execute_move_to_pose(target_pose)

        except ValueError as e:
            response.result_code.result_code = ResultCode.INVALID_INPUT
            response.result_code.message = str(e)
            response.data = json.dumps({"success": False, "error": str(e)})
        except Exception as e:
            self.is_moving = False
            self.target_pose = None
            response.result_code.result_code = ResultCode.FAILURE
            response.result_code.message = str(e)
            response.data = json.dumps({"success": False, "error": str(e)})

        return response

    # -------------------------------------------------------------------------
    # /robot/control/move_ee_to_rel_pose  (relative / delta position)
    # -------------------------------------------------------------------------

    def _parse_delta_position(self, req_json):
        """
        Parse and validate a delta_position request.

        Expected JSON: {"delta_position": {"x": 0.0, "y": 0.1, "z": 0.3}}
        Returns the validated delta dict.
        """
        data = json.loads(req_json)
        if "delta_position" not in data:
            raise ValueError("Missing 'delta_position'")
        dp = data["delta_position"]
        for k in ["x", "y", "z"]:
            if k not in dp:
                raise ValueError(f"Missing '{k}' in delta_position")
            if not isinstance(dp[k], (int, float)):
                raise ValueError(f"Invalid delta_position '{k}': must be a number")
        return dp

    def _handle_move_ee_to_rel_pose(self, req):
        """
        Service handler for /robot/control/move_ee_to_rel_pose.

        Reads current EE pose, adds the requested delta to the position,
        keeps the current orientation unchanged, then delegates to the
        shared blocking movement loop.
        """
        response = RobotCommandResponse()

        try:
            delta = self._parse_delta_position(req.req)

            if not self.has_received_data:
                response.result_code.result_code = ResultCode.SERVICE_NOT_RUNNING
                response.result_code.message = "Robot not connected"
                response.data = json.dumps({"success": False, "error": "Robot not connected"})
                return response

            # Snapshot current pose
            current_pose = self._get_current_pose_dict()
            if current_pose is None:
                response.result_code.result_code = ResultCode.SERVICE_NOT_RUNNING
                response.result_code.message = "Current EE pose unavailable"
                response.data = json.dumps({"success": False, "error": "Current EE pose unavailable"})
                return response

            # Compute absolute target by applying delta to current position;
            # orientation is preserved from the current EE pose.
            target_pose = {
                "position": {
                    "x": current_pose["position"]["x"] + float(delta["x"]),
                    "y": current_pose["position"]["y"] + float(delta["y"]),
                    "z": current_pose["position"]["z"] + float(delta["z"]),
                },
                "orientation": current_pose["orientation"],  # unchanged
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
            response.result_code.message = str(e)
            response.data = json.dumps({"success": False, "error": str(e)})
        except Exception as e:
            self.is_moving = False
            self.target_pose = None
            response.result_code.result_code = ResultCode.FAILURE
            response.result_code.message = str(e)
            response.data = json.dumps({"success": False, "error": str(e)})

        return response

    # -------------------------------------------------------------------------
    # /robot/control/reset_robot  (absolute pose)
    # -------------------------------------------------------------------------

    def _handle_reset_robot(self, req):
        response = RobotQueryResponse()

        try:
            reset_robot_pose = self.reset_robot_pose_config

            # Type validation
            for a in ["x", "y", "z"]:
                if not isinstance(reset_robot_pose["position"][a], (int, float)):
                    raise ValueError(f"Invalid position {a}")
            for a in ["x", "y", "z", "w"]:
                if not isinstance(reset_robot_pose["orientation"][a], (int, float)):
                    raise ValueError(f"Invalid orientation {a}")

            if not self.has_received_data:
                response.result_code.result_code = ResultCode.SERVICE_NOT_RUNNING
                response.result_code.message = "Robot not connected"
                response.data = json.dumps({"success": False, "error": "Robot not connected"})
                return response

            command_res = self._execute_move_to_pose(reset_robot_pose)
            response = RobotQueryResponse()     # Converting RobotCommandResponse to RobotQueryResponse
            response.result_code = command_res.result_code
            response.data = command_res.data
            return response

        except ValueError as e:
            response.result_code.result_code = ResultCode.INVALID_INPUT
            response.result_code.message = str(e)
            response.data = json.dumps({"success": False, "error": str(e)})
        except Exception as e:
            self.is_moving = False
            self.target_pose = None
            response.result_code.result_code = ResultCode.FAILURE
            response.result_code.message = str(e)
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