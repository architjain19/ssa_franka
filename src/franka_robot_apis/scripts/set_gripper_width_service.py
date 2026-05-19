#!/usr/bin/env python3
"""
ROS1 Noetic service node: Robotiq 2F Gripper Width Control
-----------------------------------------------------------
Services
--------
1. /robot/control/set_gripper_width                    (robot_api_interfaces/RobotCommand)
2. /robot/proprioception/get_gripper_width     (robot_api_interfaces/RobotQuery)

--- set_gripper_width ---
Request (JSON string in .req field):
{
    "width": 0.04,               # target opening width in metres (required)
    "duration_seconds": 3.0,     # max seconds to wait for motion  (optional)
    "speed": 255,                # rSP 0-255, 255 = fastest        (optional)
    "force": 150                 # rFR 0-255, 150 = medium force   (optional)
}

Response (JSON string in .data field):
{
    "success": true,
    "message": "Gripper reached target width.",
    "data": {
        "target_width_m":   0.04,
        "target_pr":        120,
        "final_pr":         118,
        "final_width_m":    0.0407,
        "duration_seconds": 3.0,
        "speed":            255,
        "force":            150,
        "status":           "reached"
    }
}

--- get_gripper_width ---
Request: empty  (RobotQuery has no request fields)

Response (JSON string in .data field):
{
    "gripper_width":         0.042,
    "units":                 "meters",
    "left_finger_position":  0.021,
    "right_finger_position": 0.021
}

Robotiq 2F-85 physical limits (adjustable via ROS params):
    rPR = 0   ->  0.085 m  (fully open)
    rPR = 255 ->  0.000 m  (fully closed)
    Linear mapping is used between these two extremes.

ROS1 usage:
    rosrun franka_robot_apis set_gripper_width_service.py

    rosservice call /robot/control/set_gripper_width \
        '{"req": "{\"width\": 0.04}"}'

    rosservice call /robot/proprioception/get_gripper_width '{}'

Gripper topics:
    Publish  : /Robotiq2FGripperRobotOutput  (command)
    Subscribe: /Robotiq2FGripperRobotInput   (feedback)
"""

import json
import time
import threading
import traceback

import rospy

from robotiq_2f_gripper_control.msg import (
    Robotiq2FGripper_robot_output as OutputMsg,
    Robotiq2FGripper_robot_input  as InputMsg,
)
from robot_api_interfaces.srv import (
    RobotCommand,  RobotCommandResponse,
    RobotQuery,    RobotQueryResponse,
)
from robot_api_interfaces.msg import ResultCode


# ---------------------------------------------------------------------------
# Defaults - all overridable via ROS params
# ---------------------------------------------------------------------------
DEFAULT_MAX_WIDTH_M       = 0.085   # metres at rPR = 0   (open,   2F-85)
DEFAULT_MIN_WIDTH_M       = 0.000   # metres at rPR = 255 (closed, 2F-85)
DEFAULT_SPEED             = 190     # rSP: 0 (slow) - 255 (fast)
DEFAULT_FORCE             = 150     # rFR: 0 (light) - 255 (max)
DEFAULT_DURATION          = 3.0     # seconds to wait for motion completion
DEFAULT_POSITION_TOL_PR   = 5       # ±rPR counts to accept as "reached"


class SetGripperWidthNode:
    """
    ROS1 service node that:
      1. Accepts a target width in metres
      2. Maps it linearly to an rPR byte (0-255)
      3. Publishes a Robotiq2FGripper_robot_output command
      4. Polls Robotiq2FGripper_robot_input until motion completes or times out
      5. Returns a structured JSON response with the final gripper state.

    Gripper feedback gOBJ states
    ----------------------------
    0 - fingers moving
    1 - stopped while opening  (object detected)
    2 - stopped while closing  (object detected)
    3 - fingers reached commanded position
    """

    def __init__(self):
        # ------------------------------------------------------------------ #
        #  Parameters                                                          #
        # ------------------------------------------------------------------ #
        self.output_topic         = rospy.get_param("~output_topic",         "/Robotiq2FGripperRobotOutput")
        self.input_topic          = rospy.get_param("~input_topic",          "/Robotiq2FGripperRobotInput")
        self.max_width_m          = float(rospy.get_param("~max_width_m",          DEFAULT_MAX_WIDTH_M))
        self.min_width_m          = float(rospy.get_param("~min_width_m",          DEFAULT_MIN_WIDTH_M))
        self.default_speed        = int(rospy.get_param("~default_speed",          DEFAULT_SPEED))
        self.default_force        = int(rospy.get_param("~default_force",          DEFAULT_FORCE))
        self.default_duration     = float(rospy.get_param("~default_duration",     DEFAULT_DURATION))
        self.position_tol_pr      = int(rospy.get_param("~position_tolerance_pr",  DEFAULT_POSITION_TOL_PR))

        # ------------------------------------------------------------------ #
        #  Gripper feedback state                                              #
        # ------------------------------------------------------------------ #
        self._latest_input  = None           # InputMsg or None
        self._input_lock    = threading.Lock()

        # ------------------------------------------------------------------ #
        #  Publisher -> gripper command                                         #
        # ------------------------------------------------------------------ #
        self._cmd_pub = rospy.Publisher(
            self.output_topic,
            OutputMsg,
            queue_size=10,
        )

        # ------------------------------------------------------------------ #
        #  Subscriber ← gripper state                                          #
        # ------------------------------------------------------------------ #
        rospy.Subscriber(
            self.input_topic,
            InputMsg,
            self._input_cb,
            queue_size=10,
        )

        # Give the publisher time to connect before we send the activation
        rospy.sleep(0.5)
        self._activate()

        # ------------------------------------------------------------------ #
        #  Service                                                             #
        # ------------------------------------------------------------------ #
        self._set_width_service = rospy.Service(
            "/robot/control/set_gripper_width",
            RobotCommand,
            self._handle_set_width,
        )

        self._get_width_service = rospy.Service(
            "/robot/proprioception/get_gripper_width",
            RobotQuery,
            self._handle_get_width,
        )

        rospy.loginfo(
            "\nSetGripperWidthNode (ROS1) ready.\n"
            f"  Service (set): /robot/control/set_gripper_width\n"
            f"  Service (get): /robot/proprioception/get_gripper_width\n"
            f"  Output topic : {self.output_topic}\n"
            f"  Input  topic : {self.input_topic}\n"
            f"  Width range  : {self.min_width_m*1000:.1f} mm (rPR=255, closed)"
            f" - {self.max_width_m*1000:.1f} mm (rPR=0, open)"
        )

    # ------------------------------------------------------------------ #
    #  Feedback callback                                                   #
    # ------------------------------------------------------------------ #

    def _input_cb(self, msg):
        with self._input_lock:
            self._latest_input = msg

    def _get_latest_input(self):
        with self._input_lock:
            return self._latest_input

    # ------------------------------------------------------------------ #
    #  Activation sequence                                                 #
    # ------------------------------------------------------------------ #

    def _activate(self):
        """
        Send the two-step activation required by the Robotiq 2F driver:
            Step 1 - reset  (rACT=0)
            Step 2 - activate (rACT=1)

        Waits up to 5 s for gSTA==3 (activation complete).
        Only activates if the gripper is not already active.
        """
        # Check if already active
        inp = self._get_latest_input()
        if inp is not None and inp.gACT == 1 and inp.gSTA == 3:
            rospy.loginfo("Gripper already activated (gACT=1, gSTA=3). Skipping activation.")
            return

        rospy.loginfo("Activating Robotiq 2F gripper ...")

        # Step 1 - reset
        reset_msg      = OutputMsg()
        reset_msg.rACT = 0
        self._cmd_pub.publish(reset_msg)
        rospy.sleep(0.5)

        # Step 2 - activate
        act_msg      = OutputMsg()
        act_msg.rACT = 1
        act_msg.rGTO = 1
        act_msg.rATR = 0
        act_msg.rPR  = 0                     # open during activation
        act_msg.rSP  = self.default_speed
        act_msg.rFR  = self.default_force
        self._cmd_pub.publish(act_msg)

        # Wait for gSTA == 3
        deadline = time.time() + 5.0
        while time.time() < deadline:
            inp = self._get_latest_input()
            if inp is not None and inp.gSTA == 3:
                rospy.loginfo("Gripper activation confirmed (gSTA=3).")
                return
            rospy.sleep(0.1)

        rospy.logwarn(
            "Gripper activation timed out - gSTA never reached 3. "
            "The gripper may not be connected yet, or was already active."
        )

    # ------------------------------------------------------------------ #
    #  Width ↔ rPR conversion                                             #
    # ------------------------------------------------------------------ #

    def _width_to_pr(self, width_m):
        """
        Map a target width in metres -> rPR byte (0-255).

        Linear mapping:
            width = max_width_m  ->  rPR = 0    (open)
            width = min_width_m  ->  rPR = 255  (closed)

        Args:
            width_m (float): desired opening in metres, clamped to valid range

        Returns:
            int: rPR value in [0, 255]
        """
        width_m = max(self.min_width_m, min(self.max_width_m, float(width_m)))
        span    = self.max_width_m - self.min_width_m
        if span <= 0:
            return 0
        ratio = (self.max_width_m - width_m) / span   # 0.0 = open, 1.0 = closed
        return max(0, min(255, int(round(ratio * 255))))

    def _pr_to_width(self, pr):
        """
        Map an rPR byte (0–255) → width in metres.

        Args:
            pr (int): rPR value

        Returns:
            float: corresponding width in metres
        """
        ratio = max(0, min(255, int(pr))) / 255.0
        return self.max_width_m - ratio * (self.max_width_m - self.min_width_m)

    # ------------------------------------------------------------------ #
    #  Service handler: set_gripper_width                                  #
    # ------------------------------------------------------------------ #

    def _handle_set_width(self, request):
        """
        rospy.Service callback for /robot/control/set_gripper_width.

        Args:
            request (RobotCommand.Request): .req holds the JSON string

        Returns:
            RobotCommandResponse
        """
        rospy.loginfo(f"set_gripper_width request: {request.req}")
        response = RobotCommandResponse()

        # --- 1. Parse JSON ----------------------------------------------
        try:
            req_data = json.loads(request.req)
        except (json.JSONDecodeError, ValueError) as e:
            return self._fail(response, f"Bad request JSON: {e}")

        if "width" not in req_data:
            return self._fail(response, "Missing required field 'width' (metres).")

        try:
            target_width_m   = float(req_data["width"])
            duration_seconds = float(req_data.get("duration_seconds", self.default_duration))
            speed            = int(req_data.get("speed",  self.default_speed))
            force            = int(req_data.get("force",  self.default_force))
        except (TypeError, ValueError) as e:
            return self._fail(response, f"Invalid parameter value: {e}")

        # Clamp speed / force
        speed = max(0, min(255, speed))
        force = max(0, min(255, force))

        # Validate width range - give caller a clear error rather than silently clamping
        lo, hi = self.min_width_m, self.max_width_m
        if not (lo - 1e-6 <= target_width_m <= hi + 1e-6):
            return self._fail(
                response,
                f"Requested width {target_width_m*1000:.2f} mm is outside the "
                f"valid range [{lo*1000:.1f} mm - {hi*1000:.1f} mm]. "
                f"Adjust your request or the ~min_width_m / ~max_width_m params."
            )
        # Clamp to exact bounds after validation
        target_width_m = max(lo, min(hi, target_width_m))

        target_pr = self._width_to_pr(target_width_m)

        rospy.loginfo(
            f"Target: {target_width_m*1000:.2f} mm  ->  rPR={target_pr}  "
            f"speed={speed}  force={force}  wait={duration_seconds}s"
        )

        # --- 2. Sanity-check gripper state ------------------------------
        inp = self._get_latest_input()
        if inp is None:
            rospy.logwarn(
                "No feedback from gripper input topic yet. "
                "Sending command anyway - gripper may not be ready."
            )
        elif inp.gACT != 1 or inp.gSTA != 3:
            rospy.logwarn(
                f"Gripper may not be fully activated "
                f"(gACT={getattr(inp,'gACT','?')} gSTA={getattr(inp,'gSTA','?')}). "
                "Attempting command anyway."
            )

        # --- 3. Publish command -----------------------------------------
        cmd      = OutputMsg()
        cmd.rACT = 1
        cmd.rGTO = 1
        cmd.rATR = 0
        cmd.rPR  = target_pr
        cmd.rSP  = speed
        cmd.rFR  = force
        self._cmd_pub.publish(cmd)

        rospy.loginfo(
            f"Command published: rACT=1 rGTO=1 rPR={target_pr} rSP={speed} rFR={force}"
        )

        # --- 4. Wait for motion to complete -----------------------------
        status_str, final_pr = self._wait_for_motion(target_pr, duration_seconds)
        final_width_m        = self._pr_to_width(final_pr)

        rospy.loginfo(
            f"Motion done | status={status_str}  "
            f"final_pr={final_pr}  final_width={final_width_m*1000:.2f} mm"
        )

        # --- 5. Build response ------------------------------------------
        _STATUS_MESSAGES = {
            "reached":         "Gripper reached target width.",
            "object_detected": "Gripper stopped - object detected before target.",
            "timeout":         "Motion timed out before reaching target.",
            "fault":           "Gripper reported a fault during motion.",
            "no_feedback":     "Command sent; no gripper feedback available to confirm.",
        }
        success = status_str in ("reached", "object_detected")
        message = _STATUS_MESSAGES.get(status_str, f"Unknown status: {status_str}")

        payload = {
            "success": success,
            "message": message,
            "data": {
                "target_width_m":   target_width_m,
                "target_pr":        target_pr,
                "final_pr":         final_pr,
                "final_width_m":    round(final_width_m, 6),
                "duration_seconds": duration_seconds,
                "speed":            speed,
                "force":            force,
                "status":           status_str,
            },
        }

        response.result_code.result_code = ResultCode.SUCCESS if success else ResultCode.FAILURE
        response.result_code.message     = message
        response.data                    = json.dumps(payload)
        return response

    # ------------------------------------------------------------------ #
    #  Service handler: get_gripper_width                          #
    # ------------------------------------------------------------------ #

    def _handle_get_width(self, request):
        """
        rospy.Service callback for /robot/proprioception/get_gripper_width.

        Reads gPO (current position echo, 0-255) from the latest gripper
        feedback message and converts it to metres using the same linear
        mapping as _pr_to_width.

        The Robotiq 2F has two symmetric fingers, so each finger travels
        half the total width from the centreline.

        Args:
            request (RobotQuery.Request): no fields

        Returns:
            RobotQueryResponse
        """
        rospy.loginfo("get_gripper_width request received.")
        response = RobotQueryResponse()

        inp = self._get_latest_input()

        if inp is None:
            rospy.logwarn(
                "No feedback received from gripper input topic yet. "
                f"Is '{self.input_topic}' publishing?"
            )
            response.result_code.result_code = ResultCode.FAILURE
            response.result_code.message     = "No gripper feedback available."
            response.data = json.dumps({
                "gripper_width":         None,
                "units":                 "meters",
                "left_finger_position":  None,
                "right_finger_position": None,
                "error":                 "No feedback received from gripper input topic.",
            })
            return response

        current_pr    = int(inp.gPO)   # gPO: current position echo (0-255)
        gripper_width = self._pr_to_width(current_pr)

        # Each finger moves symmetrically - half of total width from centre
        finger_pos = gripper_width / 2.0

        rospy.loginfo(
            f"Current gripper state: gPO={current_pr}  "
            f"width={gripper_width*1000:.2f} mm  "
            f"finger_pos={finger_pos*1000:.2f} mm each"
        )

        response.result_code.result_code = ResultCode.SUCCESS
        response.result_code.message     = "Successfully retrieved gripper width"
        response.data = json.dumps({
            "gripper_width":         round(gripper_width, 3),
            "units":                 "meters",
            "left_finger_position":  round(finger_pos, 9),
            "right_finger_position": round(finger_pos, 9),
        })
        return response

    # ------------------------------------------------------------------ #
    #  Motion completion polling                                           #
    # ------------------------------------------------------------------ #

    def _wait_for_motion(self, target_pr, duration_seconds):
        """
        Poll /Robotiq2FGripperRobotInput until the gripper stops or times out.

        gOBJ interpretation
        -------------------
        0 - fingers moving
        1 - stopped while opening,  object/contact detected
        2 - stopped while closing,  object/contact detected
        3 - fingers reached the requested position

        gFLT != 0 - fault; abort immediately.

        After publishing a new command the gripper takes a few milliseconds to
        start moving.  If we read gOBJ immediately we may see the *previous*
        terminal state (gOBJ==3 from the last command) and return "reached"
        before the fingers have moved at all.

        Strategy:
          Phase 1 - wait up to MOTION_START_TIMEOUT for gOBJ to become 0
                    (fingers moving).  If motion never starts we still check
                    whether we are already at the target (gPO ≈ target_pr).
          Phase 2 - once motion is confirmed, poll until gOBJ != 0 or timeout.

        Args:
            target_pr        (int):   the rPR value we commanded
            duration_seconds (float): maximum total wait time in seconds

        Returns:
            tuple: (status_string, final_pr_int)
        """
        MOTION_START_TIMEOUT = 0.5   # seconds to wait for gOBJ==0
        POLL_INTERVAL        = 0.05  # 20 Hz

        overall_deadline  = time.time() + duration_seconds
        last_pr           = target_pr  # fallback if no feedback arrives

        # ------------------------------------------------------------------ #
        #  Phase 1: wait for motion to actually start (gOBJ == 0)            #
        # ------------------------------------------------------------------ #
        motion_started    = False
        start_deadline    = time.time() + MOTION_START_TIMEOUT

        while time.time() < start_deadline:
            inp = self._get_latest_input()
            if inp is None:
                rospy.sleep(POLL_INTERVAL)
                continue

            last_pr = inp.gPO

            if inp.gFLT != 0:
                rospy.logwarn(f"Gripper fault at motion-start: gFLT={inp.gFLT}")
                return "fault", last_pr

            if inp.gOBJ == 0:
                # Fingers are moving - proceed to phase 2
                rospy.logdebug("Gripper motion started (gOBJ=0).")
                motion_started = True
                break

            rospy.sleep(POLL_INTERVAL)

        if not motion_started:
            # Gripper never transitioned to moving.
            # Check if it is already sitting at (or very near) the target.
            inp = self._get_latest_input()
            if inp is not None:
                last_pr = inp.gPO
                already_there = abs(last_pr - target_pr) <= self.position_tol_pr
                rospy.logwarn(
                    f"Gripper did not start moving within {MOTION_START_TIMEOUT}s "
                    f"(gOBJ={inp.gOBJ} gPO={inp.gPO} target_pr={target_pr}). "
                    + ("Position already at target - returning 'reached'."
                       if already_there else
                       "Position not at target - returning 'timeout'.")
                )
                return ("reached" if already_there else "timeout"), last_pr
            return "no_feedback", target_pr

        # ------------------------------------------------------------------ #
        #  Phase 2: poll until motion stops or overall deadline               #
        # ------------------------------------------------------------------ #
        while time.time() < overall_deadline:
            inp = self._get_latest_input()

            if inp is None:
                rospy.sleep(POLL_INTERVAL)
                continue

            last_pr = inp.gPO

            if inp.gFLT != 0:
                rospy.logwarn(f"Gripper fault during motion: gFLT={inp.gFLT}")
                return "fault", last_pr

            if inp.gOBJ in (1, 2):
                # Stopped before reaching target - object or contact
                rospy.loginfo(
                    f"Object detected (gOBJ={inp.gOBJ}): "
                    f"gPO={inp.gPO}  target_pr={target_pr}  "
                    f"gCU={inp.gCU}"
                )
                return "object_detected", last_pr

            if inp.gOBJ == 3:
                # Reached the requested position
                rospy.loginfo(
                    f"Position reached (gOBJ=3): "
                    f"gPO={inp.gPO}  target_pr={target_pr}"
                )
                return "reached", last_pr

            # gOBJ == 0 -> still moving
            rospy.sleep(POLL_INTERVAL)

        # Overall timeout
        inp = self._get_latest_input()
        if inp is None:
            return "no_feedback", target_pr
        rospy.logwarn(
            f"Motion timed out after {duration_seconds}s: "
            f"gOBJ={inp.gOBJ}  gPO={inp.gPO}  target_pr={target_pr}"
        )
        return "timeout", inp.gPO

    # ------------------------------------------------------------------ #
    #  Response helper                                                     #
    # ------------------------------------------------------------------ #

    def _fail(self, response, msg):
        rospy.logerr(f"set_gripper_width error: {msg}")
        response.result_code.result_code = ResultCode.FAILURE
        response.result_code.message     = msg
        response.data = json.dumps({"success": False, "message": msg, "data": {}})
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
    rospy.init_node("set_gripper_width_service", anonymous=False)

    try:
        rospy.loginfo("Creating SetGripperWidthNode ...")
        node = SetGripperWidthNode()
        rospy.loginfo("SetGripperWidthNode spinning ...")
        node.spin()

    except rospy.ROSInterruptException:
        rospy.loginfo("ROS interrupt - shutting down SetGripperWidthNode.")
    except Exception as e:
        rospy.logerr(f"Fatal error: {e}\n{traceback.format_exc()}")
    finally:
        rospy.loginfo("SetGripperWidthNode shutdown complete.")


if __name__ == "__main__":
    main()