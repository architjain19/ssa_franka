#!/usr/bin/env python3
"""
ROS1 service that moves robot EE to target pose and BLOCKS until completion.

Service: /robot/control/move_ee_to_pose
Type: RobotCommand.srv

Example service request:
rosservice call /robot/control/move_ee_to_pose "req: '{\"target_pose\": {\"position\": {\"x\": 0.5, \"y\": 0.0, \"z\": 0.5}, \"orientation\": {\"x\": 0.8722, \"y\": -0.4867, \"z\": -0.0424, \"w\": 0.0264}}}'"

Features:
- Blocks until robot reaches target (within tolerance)
- Monitors progress via /franka_state_controller/franka_states
- Returns success/failure with timing info
"""

import json
import math
import rospy
from geometry_msgs.msg import PoseStamped
from robot_api_interfaces.srv import RobotCommand, RobotCommandResponse
from robot_api_interfaces.msg import ResultCode
from franka_msgs.msg import FrankaState


class MoveEEToPoseNode:
    def __init__(self):
        rospy.init_node("move_ee_to_pose_service")
        
        # Parameters
        self.equilibrium_pose_topic = rospy.get_param(
            "~equilibrium_pose_topic", 
            "/cartesian_impedance_controller/equilibrium_pose"
        )
        self.publish_rate = rospy.get_param("~publish_rate", 20)
        self.execution_timeout = rospy.get_param("~execution_timeout", 30.0)
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
        
        self.service = rospy.Service(
            "/robot/control/move_ee_to_pose", RobotCommand,
            self._handle_move_ee_to_pose)
        
        rospy.loginfo(f"Service initialized. Publishing to: {self.equilibrium_pose_topic}")

    def _robot_state_callback(self, msg):
        self.latest_o_tee = msg.O_T_EE
        self.has_received_data = True

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
        return tp

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

    def _get_current_pose(self):
        """Get current EE pose from FrankaState."""
        if self.latest_o_tee is None:
            return None, None
        # Position: elements 12,13,14
        cx, cy, cz = self.latest_o_tee[12], self.latest_o_tee[13], self.latest_o_tee[14]
        # Orientation: convert rotation matrix to quaternion
        r = [self.latest_o_tee[0], self.latest_o_tee[1], self.latest_o_tee[2],
             self.latest_o_tee[4], self.latest_o_tee[5], self.latest_o_tee[6],
             self.latest_o_tee[8], self.latest_o_tee[9], self.latest_o_tee[10]]
        # Simplified quaternion conversion
        trace = r[0] + r[4] + r[8]
        if trace > 0:
            s = math.sqrt(trace + 1.0) * 2.0
            qw, qx, qy, qz = 0.25*s, (r[7]-r[5])/s, (r[2]-r[6])/s, (r[3]-r[1])/s
        else:
            qx, qy, qz, qw = 0, 0, 0, 1  # Fallback
        return (cx, cy, cz), (qx, qy, qz, qw)

    def _check_at_target(self, target):
        """Check if within tolerance."""
        if self.latest_o_tee is None:
            return False
        cx, cy, cz = self.latest_o_tee[12], self.latest_o_tee[13], self.latest_o_tee[14]
        tx, ty, tz = target["position"]["x"], target["position"]["y"], target["position"]["z"]
        pos_dist = math.sqrt((cx-tx)**2 + (cy-ty)**2 + (cz-tz)**2)
        return pos_dist <= self.position_tolerance

    def _handle_move_ee_to_pose(self, req):
        response = RobotCommandResponse()
        
        try:
            target_pose = self._parse_target_pose(req.req)
            
            # Validate ranges
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
            
            # Start movement
            self.target_pose = target_pose
            self.is_moving = True
            self.movement_start_time = rospy.Time.now()
            tx, ty, tz = target_pose["position"]["x"], target_pose["position"]["y"], target_pose["position"]["z"]
            rospy.loginfo(f"Moving to: ({tx:.3f}, {ty:.3f}, {tz:.3f})")
            
            rate = rospy.Rate(self.publish_rate)
            
            while self.is_moving and not rospy.is_shutdown():
                # Check timeout
                elapsed = (rospy.Time.now() - self.movement_start_time).to_sec()
                if elapsed > self.execution_timeout:
                    self.is_moving = False
                    self.target_pose = None
                    response.result_code.result_code = ResultCode.TIMEOUT
                    response.result_code.message = f"Timeout after {elapsed:.1f}s"
                    response.data = json.dumps({"success": False, "error": "Timeout", "elapsed": elapsed})
                    return response
                
                # Check if at target
                if self._check_at_target(target_pose):
                    self.is_moving = False
                    rospy.loginfo(f"Completed in {elapsed:.1f}s")
                    response.result_code.result_code = ResultCode.SUCCESS
                    response.result_code.message = "Trajectory execution completed successfully"
                    response.data = json.dumps({
                        "success": True,
                        "message": "Trajectory execution completed successfully",
                        "elapsed_time": elapsed,
                    })
                    self.target_pose = None
                    return response
                
                # Publish target pose
                self.pose_pub.publish(self._create_pose_stamped(target_pose))
                rate.sleep()
            
            # Interrupted
            response.result_code.result_code = ResultCode.FAILURE
            response.result_code.message = "Service interrupted"
            response.data = json.dumps({"success": False, "error": "Interrupted"})
            self.target_pose = None
            self.is_moving = False
            
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
        node = MoveEEToPoseNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Shutting down...")
    except Exception as e:
        rospy.logerr(f"Failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()