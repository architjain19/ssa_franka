#!/usr/bin/env python3
"""
ROS1 python node as a ros service which listens to /franka_state_controller/franka_states
topic and returns the current end-effector pose of the robot as a JSON string in the 
service response.

Service message type: RobotQuery.srv
Request: empty
Response: 
    - result_code: robot_api_interfaces/ResultCode
    - data: JSON-encoded payload

Return data as json string with the following format:
{
    "ee_pose": {
        "position": {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0
        },
        "orientation": {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "w": 1.0
        }
    }
}
"""

import json
import math
import rospy
from robot_api_interfaces.srv import RobotQuery, RobotQueryResponse
from robot_api_interfaces.msg import ResultCode
from franka_msgs.msg import FrankaState


class GetCurrentRobotStateNode:
    """
    ROS1 service node that provides current end-effector pose and joints info via FrankaState subscription.
    
    Subscribes to /franka_state_controller/franka_states topic and extracts the
    O_T_EE (4x4 homogeneous transform) to compute position and orientation.
    """

    def __init__(self):
        """Initialize the service node."""
        # Node initialization
        rospy.init_node("get_current_robot_state_service")
        
        # Parameters
        self.tf_timeout = rospy.get_param("~tf_timeout", 1.0)
        self.state_timeout = rospy.get_param("~state_timeout", 1.0)
        
        # Store latest EE transform (16-element array: 4x4 homogeneous matrix)
        # Format: [r11, r12, r13, tx, r21, r22, r23, ty, r31, r32, r33, tz, 0, 0, 0, 1]
        self.latest_o_tee = None
        self.latest_q = None          # Joint positions
        self.latest_dq = None         # Joint velocities
        self.latest_tau_J = None      # Joint torques
        self.latest_stamp = None
        self.has_received_data = False
        
        # Subscribe to FrankaState topic
        self.sub = rospy.Subscriber(
            "/franka_state_controller/franka_states",
            FrankaState,
            self._franka_state_callback,
            queue_size=1,
        )
        
        # Create services (both from same FrankaState topic)
        self.ee_pose_service = rospy.Service(
            "/robot/proprioception/get_current_ee_pose",
            RobotQuery,
            self._handle_get_current_ee_pose,
        )
        
        self.joints_service = rospy.Service(
            "/robot/proprioception/get_current_joints",
            RobotQuery,
            self._handle_get_current_joints,
        )
        
        rospy.loginfo(
            "Services initialized. "
            "Subscribing to: /franka_state_controller/franka_states"
        )

    def _franka_state_callback(self, msg):
        """
        Callback for FrankaState messages.
        
        Args:
            msg: FrankaState message containing O_T_EE transform and joint data
        """
        self.latest_o_tee = msg.O_T_EE
        self.latest_q = msg.q           # Joint positions
        self.latest_dq = msg.dq         # Joint velocities
        self.latest_tau_J = msg.tau_J  # Joint torques
        self.latest_stamp = msg.header.stamp
        self.has_received_data = True
        rospy.logdebug_throttle(1.0, "Received FrankaState update")

    def _rotation_matrix_to_quaternion(self, r):
        """
        Convert 3x3 rotation matrix to quaternion.
        
        Args:
            r: 9-element rotation matrix [r11, r12, r13, r21, r22, r23, r31, r32, r33]
            
        Returns:
            tuple: (x, y, z, w) quaternion components
        """
        # Extract rotation matrix elements
        r11, r12, r13 = r[0], r[1], r[2]
        r21, r22, r23 = r[3], r[4], r[5]
        r31, r32, r33 = r[6], r[7], r[8]
        
        # Trace of the rotation matrix
        trace = r11 + r22 + r33
        
        if trace > 0:
            # Case: trace > 0 (spherical quaternion)
            s = math.sqrt(trace + 1.0) * 2.0
            w = 0.25 * s
            x = (r32 - r23) / s
            y = (r13 - r31) / s
            z = (r21 - r12) / s
            
        elif (r11 > r22) and (r11 > r33):
            # Case: r11 is largest diagonal element
            s = math.sqrt(1.0 + r11 - r22 - r33) * 2.0
            w = (r32 - r23) / s
            x = 0.25 * s
            y = (r12 + r21) / s
            z = (r13 + r31) / s
            
        elif r22 > r33:
            # Case: r22 is largest diagonal element
            s = math.sqrt(1.0 + r22 - r11 - r33) * 2.0
            w = (r13 - r31) / s
            x = (r12 + r21) / s
            y = 0.25 * s
            z = (r23 + r32) / s
            
        else:
            # Case: r33 is largest diagonal element
            s = math.sqrt(1.0 + r33 - r11 - r22) * 2.0
            w = (r21 - r12) / s
            x = (r13 + r31) / s
            y = (r23 + r32) / s
            z = 0.25 * s
        
        # Normalize quaternion
        norm = math.sqrt(x*x + y*y + z*z + w*w)
        if norm > 1e-10:
            x, y, z, w = x/norm, y/norm, z/norm, w/norm
            
        return x, y, z, w

    def _extract_pose_from_o_tee(self, o_tee):
        """
        Extract position and orientation from O_T_EE homogeneous transform.
        
        Args:
            o_tee: 16-element homogeneous transform array
            
        Returns:
            dict: { "position": {x,y,z}, "orientation": {x,y,z,w} }
        """
        if o_tee is None or len(o_tee) != 16:
            raise ValueError("Invalid O_T_EE data")
        
        # Extract position (translation) from elements 12, 13, 14
        tx, ty, tz = o_tee[12], o_tee[13], o_tee[14]
        
        # Extract rotation matrix from elements 0-11
        # Format: [r11, r12, r13, tx, r21, r22, r23, ty, r31, r32, r33, tz, ...]
        rotation_matrix = [
            o_tee[0], o_tee[1], o_tee[2],   # row 1
            o_tee[4], o_tee[5], o_tee[6],   # row 2
            o_tee[8], o_tee[9], o_tee[10],  # row 3
        ]
        
        # Convert rotation matrix to quaternion
        qx, qy, qz, qw = self._rotation_matrix_to_quaternion(rotation_matrix)
        
        return {
            "position": {
                "x": float(tx),
                "y": float(ty),
                "z": float(tz),
            },
            "orientation": {
                "x": float(qx),
                "y": float(qy),
                "z": float(qz),
                "w": float(qw),
            },
        }

    def _handle_get_current_ee_pose(self, req):
        """
        Handle the get_current_ee_pose service request.
        
        Args:
            req: Empty RobotQuery request
            
        Returns:
            RobotQuery.Response with pose data or error details
        """
        response = RobotQueryResponse()
        
        try:
            # Check if we have received data
            if not self.has_received_data:
                error_msg = "No FrankaState data received yet. Robot may be disconnected."
                rospy.logwarn(error_msg)
                
                response.result_code.result_code = ResultCode.SERVICE_NOT_RUNNING
                response.result_code.message = error_msg
                response.data = json.dumps({"error": error_msg})
                return response
            
            # Check if data is fresh (within timeout)
            if self.latest_stamp is not None:
                time_since_update = (rospy.Time.now() - self.latest_stamp).to_sec()
                if time_since_update > self.state_timeout:
                    error_msg = f"Stale data: last update {time_since_update:.2f}s ago (timeout: {self.state_timeout}s)"
                    rospy.logwarn(error_msg)
                    
                    response.result_code.result_code = ResultCode.TIMEOUT
                    response.result_code.message = error_msg
                    response.data = json.dumps({"error": error_msg})
                    return response
            
            # Extract pose from O_T_EE
            pose = self._extract_pose_from_o_tee(self.latest_o_tee)
            
            # Create response payload
            payload = {
                "ee_pose": pose,
            }
            
            # Populate success response
            response.result_code.result_code = ResultCode.SUCCESS
            response.result_code.message = "Successfully retrieved end-effector pose"
            response.data = json.dumps(payload)
            
            rospy.loginfo("Successfully retrieved end-effector pose")
            
        except ValueError as e:
            # Handle invalid data errors
            error_msg = f"Invalid O_T_EE data: {str(e)}"
            rospy.logwarn(error_msg)
            
            response.result_code.result_code = ResultCode.FAILURE
            response.result_code.message = error_msg
            response.data = json.dumps({"error": error_msg})
            
        except Exception as e:
            # Handle unexpected errors
            error_msg = f"Unexpected error while retrieving end-effector pose: {str(e)}"
            rospy.logerr(error_msg)
            
            response.result_code.result_code = ResultCode.FAILURE
            response.result_code.message = error_msg
            response.data = json.dumps({"error": error_msg})
        
        return response

    def _handle_get_current_joints(self, req):
        """
        Handle the get_current_joints service request.
        
        Args:
            req: Empty RobotQuery request
            
        Returns:
            RobotQuery.Response with joint states or error details
        """
        response = RobotQueryResponse()
        
        try:
            # Check if we have received data
            if not self.has_received_data:
                error_msg = "No FrankaState data received yet. Robot may be disconnected."
                rospy.logwarn(error_msg)
                
                response.result_code.result_code = ResultCode.SERVICE_NOT_RUNNING
                response.result_code.message = error_msg
                response.data = json.dumps({"error": error_msg})
                return response
            
            # Check if data is fresh (within timeout)
            if self.latest_stamp is not None:
                time_since_update = (rospy.Time.now() - self.latest_stamp).to_sec()
                if time_since_update > self.state_timeout:
                    error_msg = f"Stale data: last update {time_since_update:.2f}s ago (timeout: {self.state_timeout}s)"
                    rospy.logwarn(error_msg)
                    
                    response.result_code.result_code = ResultCode.TIMEOUT
                    response.result_code.message = error_msg
                    response.data = json.dumps({"error": error_msg})
                    return response
            
            # Extract joint data from FrankaState (q, dq, tau_J)
            # Franka Panda has 7 joints
            joint_names = ["panda_joint1", "panda_joint2", "panda_joint3", 
                           "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7"]
            
            joints_data = {}
            error_messages = []
            
            for i, name in enumerate(joint_names):
                try:
                    # Extract position, velocity, effort with bounds checking
                    position = None
                    velocity = None
                    effort = None
                    
                    if self.latest_q is not None and i < len(self.latest_q):
                        position = float(self.latest_q[i])
                    if self.latest_dq is not None and i < len(self.latest_dq):
                        velocity = float(self.latest_dq[i])
                    if self.latest_tau_J is not None and i < len(self.latest_tau_J):
                        effort = float(self.latest_tau_J[i])
                    
                    joints_data[name] = {
                        "position": position,
                        "velocity": velocity,
                        "effort": effort,
                    }
                    
                except Exception as e:
                    error_msg = f"Error processing joint '{name}': {str(e)}"
                    rospy.logwarn(error_msg)
                    error_messages.append(error_msg)
            
            # Create response payload
            payload = {
                "joints": joints_data,
            }
            
            # Populate success response
            response.result_code.result_code = ResultCode.SUCCESS
            if error_messages:
                response.result_code.message = (
                    f"Retrieved {len(joints_data)} joint state(s). "
                    f"Errors: {'; '.join(error_messages[:2])}"
                )
            else:
                response.result_code.message = (
                    f"Successfully retrieved {len(joints_data)} joint state(s)"
                )
            response.data = json.dumps(payload)
            
            rospy.loginfo(f"Successfully retrieved {len(joints_data)} joint states")
            
        except Exception as e:
            # Handle unexpected errors
            error_msg = f"Unexpected error while retrieving joint states: {str(e)}"
            rospy.logerr(error_msg)
            
            response.result_code.result_code = ResultCode.FAILURE
            response.result_code.message = error_msg
            response.data = json.dumps({"error": error_msg})
        
        return response


def main():
    """Initialize and run the node."""
    try:
        node = GetCurrentRobotStateNode()
        rospy.loginfo("Starting get current ee pose and joints service...")
        rospy.spin()
        
    except rospy.ROSInterruptException:
        rospy.loginfo("Shutting down get current ee pose node...")
    except Exception as e:
        rospy.logerr(f"Failed to start get current ee pose node: {str(e)}")
        raise


if __name__ == "__main__":
    main()