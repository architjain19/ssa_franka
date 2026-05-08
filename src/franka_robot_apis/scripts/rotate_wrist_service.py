#!/usr/bin/env python3
"""
ROS1 service that rotates the EE wrist by a specified angle in the EE frame
about the tool Z-axis (gripper roll / wrist spin).

Service:
  /robot/control/rotate_wrist - Rotate wrist by angle and direction

Type: RobotCommand.srv

Example service requests:
  # Rotate 45 degrees clockwise
  rosservice call /robot/control/rotate_wrist \
    "req: '{\"angle_deg\": 45.0, \"direction\": \"clockwise\"}'"

  # Rotate 30 degrees counter-clockwise
  rosservice call /robot/control/rotate_wrist \
    "req: '{\"angle_deg\": 30.0, \"direction\": \"counter-clockwise\"}'"
"""

import json
import math
import rospy
from robot_api_interfaces.srv import RobotCommand, RobotCommandResponse
from robot_api_interfaces.msg import ResultCode


class RotateWristNode:
    def __init__(self):
        rospy.init_node("rotate_wrist_service_node")

        # ===== Parameters =====
        self.max_angle_deg          = float(rospy.get_param("~max_angle_deg", 180.0))
        self.move_ee_svc             = rospy.get_param("~move_ee_svc",
                                                       "/robot/control/move_ee_to_pose")
        self.get_ee_pose_svc        = rospy.get_param("~get_ee_pose_svc",
                                                       "/robot/proprioception/get_current_ee_pose")
        self.move_ee_svc_timeout    = float(rospy.get_param("~move_ee_svc_timeout", 5.0))
        self.get_ee_pose_svc_timeout = float(rospy.get_param("~get_ee_pose_svc_timeout", 5.0))

        # ===== Service Proxies =====
        from robot_api_interfaces.srv import RobotQuery

        rospy.loginfo(f"Waiting for service {self.get_ee_pose_svc} ...")
        try:
            rospy.wait_for_service(self.get_ee_pose_svc, timeout=self.get_ee_pose_svc_timeout)
        except rospy.ROSException:
            rospy.logwarn(f"Service {self.get_ee_pose_svc} not yet available - will retry on each call.")
        self._ee_pose_proxy = rospy.ServiceProxy(self.get_ee_pose_svc, RobotQuery)

        rospy.loginfo(f"Waiting for service {self.move_ee_svc} ...")
        try:
            rospy.wait_for_service(self.move_ee_svc, timeout=self.move_ee_svc_timeout)
        except rospy.ROSException:
            rospy.logwarn(f"Service {self.move_ee_svc} not yet available - will retry on each call.")
        self._move_ee_proxy = rospy.ServiceProxy(self.move_ee_svc, RobotCommand)

        # ===== ROS Service =====
        rospy.Service("/robot/control/rotate_wrist", RobotCommand, self._handle_rotate_wrist)

        rospy.loginfo(
            f"RotateWristNode ready.\n"
            f"  /robot/control/rotate_wrist\n"
            f"  upstream move_ee_svc   : {self.move_ee_svc}\n"
            f"  upstream ee_pose_svc   : {self.get_ee_pose_svc}\n"
            f"  max_angle_deg          : {self.max_angle_deg}"
        )

    # =========================================================================
    # Quaternion helpers
    # =========================================================================

    def _normalize_quaternion(self, q):
        """Normalize to unit length and canonicalize so w >= 0."""
        qx, qy, qz, qw = float(q["x"]), float(q["y"]), float(q["z"]), float(q["w"])
        norm = math.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
        if norm < 1e-10:
            rospy.logwarn(f"Quaternion norm extremely small ({norm}), using identity")
            return {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
        qx /= norm; qy /= norm; qz /= norm; qw /= norm
        if qw < 0:
            qx, qy, qz, qw = -qx, -qy, -qz, -qw
        return {"x": qx, "y": qy, "z": qz, "w": qw}

    def _quaternion_multiply(self, q1, q2):
        """
        Hamilton product q1 * q2 (each as dict {x, y, z, w}).
        Post-multiplication (q_current * q_delta) applies q_delta in the
        EE/tool frame, which is what we want for wrist spin.
        """
        x1, y1, z1, w1 = q1["x"], q1["y"], q1["z"], q1["w"]
        x2, y2, z2, w2 = q2["x"], q2["y"], q2["z"], q2["w"]
        return {
            "x": w1*x2 + x1*w2 + y1*z2 - z1*y2,
            "y": w1*y2 - x1*z2 + y1*w2 + z1*x2,
            "z": w1*z2 + x1*y2 - y1*x2 + z1*w2,
            "w": w1*w2 - x1*x2 - y1*y2 - z1*z2,
        }

    def _quaternion_from_axis_angle(self, axis, angle_rad):
        """Build a unit quaternion from an axis (x, y, z) and angle in radians."""
        ax, ay, az = axis
        norm = math.sqrt(ax*ax + ay*ay + az*az)
        if norm < 1e-10:
            return {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
        ax /= norm; ay /= norm; az /= norm
        half = angle_rad / 2.0
        s = math.sin(half)
        return {"x": ax*s, "y": ay*s, "z": az*s, "w": math.cos(half)}

    # =========================================================================
    # Helpers to talk to upstream services
    # =========================================================================

    def _get_current_ee_pose(self):
        """Returns dict {position, orientation} or None on failure."""
        try:
            resp = self._ee_pose_proxy()
        except rospy.ServiceException as e:
            rospy.logerr(f"get_current_ee_pose failed: {e}")
            return None

        if resp.result_code.result_code != ResultCode.SUCCESS:
            rospy.logerr(f"get_current_ee_pose non-success: {resp.result_code.message}")
            return None

        try:
            data = json.loads(resp.data)
            ee_pose = data["ee_pose"]
            _ = ee_pose["position"]["x"]
            _ = ee_pose["orientation"]["w"]
            return ee_pose
        except (json.JSONDecodeError, KeyError) as e:
            rospy.logerr(f"Failed to parse get_current_ee_pose response: {e}")
            return None

    def _call_move_ee_to_pose(self, target_pose):
        """Forwards the absolute target pose to /robot/control/move_ee_to_pose."""
        req_json = json.dumps({"target_pose": target_pose})
        try:
            return self._move_ee_proxy(req_json)
        except rospy.ServiceException as e:
            rospy.logerr(f"move_ee_to_pose call failed: {e}")
            return None

    # =========================================================================
    # Request parsing
    # =========================================================================

    def _parse_request(self, req_json):
        """
        Parse {"angle_deg": float, "direction": "clockwise"|"counter-clockwise"}.
        Returns (angle_rad_signed, raw_dict) — sign convention:
            counter-clockwise (CCW) = positive rotation about +Z (right-hand rule)
            clockwise         (CW)  = negative rotation about +Z
        """
        data = json.loads(req_json)

        if "angle_deg" not in data:
            raise ValueError("Missing 'angle_deg'")
        if "direction" not in data:
            raise ValueError("Missing 'direction'")

        angle_deg = data["angle_deg"]
        direction = str(data["direction"]).strip().lower()

        if not isinstance(angle_deg, (int, float)):
            raise ValueError("'angle_deg' must be a number")
        if angle_deg < 0:
            raise ValueError("'angle_deg' must be non-negative; use 'direction' for sign")
        if angle_deg > self.max_angle_deg:
            raise ValueError(
                f"'angle_deg' ({angle_deg}) exceeds max_angle_deg ({self.max_angle_deg})"
            )

        # Accept a few common spellings
        cw_aliases  = {"clockwise", "cw"}
        ccw_aliases = {"counter-clockwise", "counterclockwise", "anti-clockwise",
                       "anticlockwise", "ccw"}

        if direction in cw_aliases:
            sign = -1.0
        elif direction in ccw_aliases:
            sign = +1.0
        else:
            raise ValueError(
                f"'direction' must be one of clockwise/counter-clockwise, got '{direction}'"
            )

        return sign * math.radians(angle_deg), {"angle_deg": angle_deg, "direction": direction}

    # =========================================================================
    # Service handler
    # =========================================================================

    def _handle_rotate_wrist(self, req):
        response = RobotCommandResponse()
        rospy.loginfo(f"rotate_wrist request: {req.req}")

        # ---- Parse ----
        try:
            angle_rad, parsed = self._parse_request(req.req)
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            response.result_code.result_code = ResultCode.INVALID_INPUT
            response.result_code.message     = f"Bad request: {e}"
            response.data = json.dumps({"success": False, "error": str(e)})
            return response

        # ---- Get current EE pose ----
        current_pose = self._get_current_ee_pose()
        if current_pose is None:
            response.result_code.result_code = ResultCode.SERVICE_NOT_RUNNING
            response.result_code.message     = "Failed to read current EE pose"
            response.data = json.dumps({"success": False, "error": "EE pose unavailable"})
            return response

        # ---- Compose new orientation: q_new = q_current * q_delta (tool frame) ----
        q_current = self._normalize_quaternion(current_pose["orientation"])
        q_delta   = self._quaternion_from_axis_angle((0.0, 0.0, 1.0), angle_rad)
        q_new     = self._normalize_quaternion(self._quaternion_multiply(q_current, q_delta))

        target_pose = {
            "position": {
                "x": float(current_pose["position"]["x"]),
                "y": float(current_pose["position"]["y"]),
                "z": float(current_pose["position"]["z"]),
            },
            "orientation": q_new,
        }

        rospy.loginfo(
            f"Wrist rotation — angle={parsed['angle_deg']}deg dir={parsed['direction']} "
            f"(signed_rad={angle_rad:.4f}) | "
            f"q_current=({q_current['x']:.4f}, {q_current['y']:.4f}, "
            f"{q_current['z']:.4f}, {q_current['w']:.4f}) -> "
            f"q_new=({q_new['x']:.4f}, {q_new['y']:.4f}, "
            f"{q_new['z']:.4f}, {q_new['w']:.4f})"
        )

        # ---- Forward to move_ee_to_pose ----
        upstream = self._call_move_ee_to_pose(target_pose)
        if upstream is None:
            response.result_code.result_code = ResultCode.FAILURE
            response.result_code.message     = "Upstream move_ee_to_pose call failed"
            response.data = json.dumps({"success": False, "error": "Upstream call failed"})
            return response

        # Pass through upstream result, but enrich the JSON data with rotation info
        response.result_code = upstream.result_code
        try:
            upstream_data = json.loads(upstream.data) if upstream.data else {}
        except json.JSONDecodeError:
            upstream_data = {"raw": upstream.data}

        upstream_data.update({
            "angle_deg": parsed["angle_deg"],
            "direction": parsed["direction"],
            "target_orientation": q_new,
        })
        response.data = json.dumps(upstream_data)
        return response


def main():
    try:
        node = RotateWristNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Shutting down...")
    except Exception as e:
        rospy.logerr(f"Failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()