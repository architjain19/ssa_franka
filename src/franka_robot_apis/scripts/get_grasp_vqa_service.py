#!/usr/bin/env python3
"""
ROS1 Noetic service node that verifies a grasp using a vision-language model.

Service: /robot/perception/verify_grasp  (robot_api_interfaces/RobotCommand)

Internally:
  1. Calls /robot/proprioception/get_gripper_width to read current gripper width.
  2. Captures the latest wrist-camera image from the cached subscription.
  3. Runs a proprioception gate (cheap, no VLM call) to reject obvious empties.
  4. If proprio is plausible, sends image + discriminative prompt to Azure
     OpenAI (or OpenAI / local Qwen) and parses an Answer: yes|no response.

Request JSON format:
{
    "object_name": "baseball",                # required
    "max_tokens": 200,                        # optional
    "temperature": 0.0                        # optional
}

Response JSON format (success):
{
    "result_code": 0,
    "message": "Success",
    "data": {
        "grasped": true,
        "source": "vqa",                      # "proprio" or "vqa"
        "reason": "Object is clamped between the two fingers.",
        "gripper_width_mm": 68.3,
        "object_name": "baseball",
        "raw_response": "Answer: yes\nReason: ..."   # only when source=="vqa"
    }
}

ROS1 usage:
    rosrun franka_robot_apis verify_grasp_service.py

    rosservice call /robot/perception/verify_grasp \
        '{"req": "{\"object_name\": \"baseball\"}"}'
"""

import base64
import json
import re
import threading
import time
import traceback

import cv2
import rospy
from sensor_msgs.msg import Image
from robot_api_interfaces.srv import (
    RobotCommand,
    RobotCommandResponse,
    RobotQuery,
)
from robot_api_interfaces.msg import ResultCode

# --------------------------------------------------------------------------
# Optional dependency guards
# --------------------------------------------------------------------------
try:
    from openai import OpenAI, AzureOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helper: ROS Image message -> base64-encoded JPEG string
# ---------------------------------------------------------------------------

def ros_image_to_base64_jpeg(msg, jpeg_quality=90):
    """
    Convert a ROS1 sensor_msgs/Image to a base64-encoded JPEG string.

    Supports encodings: rgb8, bgr8, rgba8, bgra8, mono8, 8uc1, mono16, 16uc1.
    """
    if not NUMPY_AVAILABLE:
        raise RuntimeError(
            "numpy is required for image conversion. "
            "Install with: pip install numpy"
        )

    encoding = msg.encoding.lower()
    height   = msg.height
    width    = msg.width
    data     = bytes(msg.data)

    if encoding in ("rgb8", "bgr8"):
        img_np = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR) if encoding == "rgb8" else img_np
    elif encoding in ("rgba8", "bgra8"):
        img_np = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 4))
        img_bgr = (
            cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
            if encoding == "rgba8"
            else cv2.cvtColor(img_np, cv2.COLOR_BGRA2BGR)
        )
    elif encoding in ("mono8", "8uc1"):
        img_np  = np.frombuffer(data, dtype=np.uint8).reshape((height, width))
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
    elif encoding in ("mono16", "16uc1"):
        img_np  = np.frombuffer(data, dtype=np.uint16).reshape((height, width))
        img_8   = cv2.normalize(img_np, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_8, cv2.COLOR_GRAY2BGR)
    else:
        raise ValueError(
            f"Unsupported image encoding: '{encoding}'. "
            "Supported: rgb8, bgr8, rgba8, bgra8, mono8, mono16, 8uc1, 16uc1"
        )

    encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    success, buffer = cv2.imencode(".jpg", img_bgr, encode_params)
    if not success:
        raise RuntimeError("cv2.imencode failed to encode image as JPEG")

    b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")
    return b64, encoding


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_grasp_prompt(object_name, gripper_width_mm):
    """
    Build a discriminative grasp-verification prompt.

    Core idea: lean toward 'yes' when any part of the object is plausibly
    pinched between the pads, especially when the gripper width is small
    and the object overlaps the gripper region.

    Rim grasps on hollow objects (cups, bowls, plates) are common and
    valid — the wall of the object is pinched between the pads while the
    bulk hangs off to one side. From the wrist camera, these often LOOK
    like the object is draped over or resting on the gripper, which is
    expected and should be treated as success.
    """
    width_line = (
        f"Reported gripper opening: {gripper_width_mm:.1f} mm.\n"
        if gripper_width_mm is not None else ""
    )
    return (
        f"You are verifying a robot grasp from a single wrist-camera "
        f"image taken AFTER the grasp attempt. The image shows a "
        f"two-finger parallel-jaw gripper viewed from the wrist camera.\n"
        f"\n"
        f"VISUAL LANDMARK: Each gripper finger has a BLUE rubber pad on "
        f"its inner gripping surface — a LEFT blue pad and a RIGHT blue "
        f"pad. These are the contact points that press against any held "
        f"object.\n"
        f"\n"
        f"CORE QUESTION: Is the {object_name} being held by the gripper, "
        f"such that some part of it is pinched between the two pads?\n"
        f"\n"
        f"You do NOT need direct visual confirmation that each pad is "
        f"making contact. Use the overall geometry: if part of the "
        f"{object_name} is plausibly in the gap between the fingers AND "
        f"the gripper has closed on it, treat that as a successful grasp.\n"
        f"\n"
        f"SPECIAL CASE — RIM / EDGE GRASPS ON HOLLOW OBJECTS:\n"
        f"For cups, mugs, bowls, plates, buckets, and similar hollow or "
        f"thin-walled objects, it is COMMON and VALID for the robot to "
        f"grasp only the RIM, LIP, or EDGE. Visual signatures:\n"
        f"  - Only a thin sliver of the object's wall is between the "
        f"pads (the cup wall, plate edge, bowl lip).\n"
        f"  - The BULK of the object (the body of the cup, the bowl of "
        f"the plate) hangs off to ONE SIDE of the gripper — above, "
        f"below, left, or right.\n"
        f"  - From the wrist camera viewpoint, this often LOOKS like "
        f"the object is 'draped over', 'resting on', or 'hanging from' "
        f"the gripper. THIS APPEARANCE IS EXPECTED FOR A VALID RIM "
        f"GRASP — it is exactly how a successful rim grasp looks from "
        f"above.\n"
        f"  - The blue pads may be mostly visible on either side of the "
        f"pinched thin wall.\n"
        f"For these cases, answer YES even though the object 'looks "
        f"draped' or 'looks like it's resting on' the gripper. The "
        f"gripper is actually pinching the rim/wall.\n"
        f"\n"
        f"WIDTH HINT:\n"
        f"If the reported gripper opening is small (e.g., 5-25 mm) AND "
        f"the {object_name} is visibly touching or overlapping the "
        f"gripper area, the gripper has almost certainly closed on some "
        f"part of the object (likely a rim, wall, or thin edge). Lean "
        f"strongly toward YES in this case unless you can see a clear "
        f"air gap separating the object from the fingers.\n"
        f"\n"
        f"{width_line}"
        f"\n"
        f"Answer 'yes' if ANY of the following is true:\n"
        f"  - The {object_name} is pressed between both blue pads "
        f"(centered grasp).\n"
        f"  - The {object_name} clearly spans the gap between the pads, "
        f"even with one pad in shadow or partially hidden.\n"
        f"  - The {object_name} is held at the edge of the fingers "
        f"(off-center) but is clamped between them.\n"
        f"  - The RIM, LIP, EDGE, or WALL of a hollow/thin {object_name} "
        f"is pinched between the fingers, with the bulk hanging off to "
        f"one side. THIS COUNTS even if it looks like the cup/bowl is "
        f"'draped over' or 'resting on' the gripper from this view.\n"
        f"  - The gripper has closed to a small width AND the object is "
        f"visibly touching/overlapping the gripper region.\n"
        f"\n"
        f"Answer 'no' ONLY if you have CLEAR EVIDENCE the object is "
        f"NOT being held — for example:\n"
        f"  - The {object_name} is clearly SEPARATE from the gripper, "
        f"with a visible AIR GAP between it and every part of the "
        f"gripper (sitting on the table behind the gripper, etc.).\n"
        f"  - Both blue pads are clearly visible with empty space "
        f"between them and no part of the {object_name} bridging that "
        f"space.\n"
        f"  - The fingers are clearly closed on empty air and the "
        f"{object_name} is far away from the gripper.\n"
        f"\n"
        f"IGNORE the following when deciding:\n"
        f"  - Whether a human hand is visible in the frame.\n"
        f"  - Whether the gripper is near a table surface.\n"
        f"  - Whether the pads are fully or partially visible.\n"
        f"  - Whether the grasp looks centered or off-center.\n"
        f"  - Whether the bulk of the object hangs off to one side.\n"
        f"  - Whether the object 'looks draped' or 'looks like it's "
        f"resting on' the gripper — for rim grasps this is normal.\n"
        f"  - Slight darkness or shadow on one side.\n"
        f"\n"
        f"Bias: Lean toward YES when ANY part of the {object_name} is "
        f"plausibly pinched between the fingers, especially when the "
        f"gripper width is small and the object overlaps the gripper "
        f"region. Only answer NO when you can clearly see an air gap "
        f"separating the object from the gripper.\n"
        f"\n"
        f"Respond in this exact format:\n"
        f"Answer: <yes|no>\n"
        f"Reason: <one short sentence describing the spatial "
        f"relationship between the object and the gripper>"
    )


def parse_yes_no(raw):
    """
    Parse 'Answer: yes/no' and 'Reason: ...' from the VLM response.

    Returns:
        tuple[bool, str]: (verdict, reason)
    """
    if not raw:
        return False, "Empty VLM response; defaulting to no."

    answer_match = re.search(r"Answer\s*:\s*(yes|no)\b", raw, flags=re.IGNORECASE)
    reason_match = re.search(r"Reason\s*:\s*(.+?)(?:\n|$)", raw,
                             flags=re.IGNORECASE | re.DOTALL)
    if answer_match:
        verdict = answer_match.group(1).lower() == "yes"
        reason = reason_match.group(1).strip() if reason_match else raw.strip()
        return verdict, reason

    token = re.search(r"\b(yes|no)\b", raw, flags=re.IGNORECASE)
    if token:
        return token.group(1).lower() == "yes", raw.strip()

    return False, f"Could not parse VLM response: {raw.strip()[:200]}"


# ---------------------------------------------------------------------------
# Main service class
# ---------------------------------------------------------------------------

class VerifyGraspServiceNode:
    """
    ROS1 service node that verifies whether the gripper currently holds
    a named object, using gripper-width proprio gating plus a VLM check.
    """

    # Default proprio thresholds (in mm). Override via ROS params.
    DEFAULT_GRIPPER_FULLY_OPEN_MM   = 85.0
    DEFAULT_GRIPPER_FULLY_CLOSED_MM = 2.0
    # Hard minimum width below which any grasp is automatically rejected
    # (the gripper has effectively closed on nothing / on something too
    # thin to be a real object).
    DEFAULT_MIN_GRASP_WIDTH_MM      = 9.0

    def __init__(self):
        # ------------------------------------------------------------------
        # ROS1 Parameters
        # ------------------------------------------------------------------
        self.qwen_host = rospy.get_param("~qwen_host", "10.158.54.164")
        self.qwen_port = rospy.get_param("~qwen_port", 8000)

        # Only wrist camera is needed for grasp verification.
        self.wrist_image_topic = rospy.get_param(
            "~wrist_image_topic", "/zed/wrist/color/image_raw"
        )

        # Gripper-width service we call internally per request.
        self.gripper_width_service = rospy.get_param(
            "~gripper_width_service", "/robot/proprioception/get_gripper_width"
        )

        # Proprio thresholds
        self.gripper_fully_open_mm = float(rospy.get_param(
            "~gripper_fully_open_mm", self.DEFAULT_GRIPPER_FULLY_OPEN_MM,
        ))
        self.gripper_fully_closed_mm = float(rospy.get_param(
            "~gripper_fully_closed_mm", self.DEFAULT_GRIPPER_FULLY_CLOSED_MM,
        ))
        self.min_grasp_width_mm = float(rospy.get_param(
            "~min_grasp_width_mm", self.DEFAULT_MIN_GRASP_WIDTH_MM,
        ))

        # Image / inference settings
        self.image_cache_timeout = float(rospy.get_param("~image_cache_timeout", 1.0))
        self.default_max_tokens  = int(rospy.get_param("~default_max_tokens",  200))
        self.default_temperature = float(rospy.get_param("~default_temperature", 0.0))
        self.jpeg_quality        = int(rospy.get_param("~jpeg_quality",        90))
        self.request_timeout     = float(rospy.get_param("~request_timeout",   15.0))
        self.gripper_service_timeout = float(rospy.get_param(
            "~gripper_service_timeout", 2.0,
        ))

        # Which VQA backend
        self.vqa_server_name = rospy.get_param(
            "~vqa_server_name", "azure_openai"
        ).lower()

        if self.vqa_server_name == "qwen":
            self.model_id = rospy.get_param(
                "~model_id",
                "/home/hcrlab/archit/qwen-vl-stretch/models/Qwen2.5-VL-7B-Instruct-AWQ",
            )
            rospy.loginfo("Using local Qwen for grasp verification.")
        elif self.vqa_server_name == "openai":
            self._load_openai_key()
            self.model_id = rospy.get_param("~model_id", "gpt-4o")
            rospy.loginfo("Using OpenAI API for grasp verification.")
        elif self.vqa_server_name == "azure_openai":
            self._load_azure_config()
            self.model_id = self.azure_deployment_name
            rospy.loginfo("Using Azure OpenAI API for grasp verification.")
        else:
            rospy.logerr(
                f"Invalid vqa_server_name: '{self.vqa_server_name}'. "
                "Must be 'qwen', 'openai', or 'azure_openai'. "
                "Defaulting to 'azure_openai'."
            )
            self.vqa_server_name = "azure_openai"
            self._load_azure_config()
            self.model_id = self.azure_deployment_name

        # ------------------------------------------------------------------
        # Internal state
        # ------------------------------------------------------------------
        self._image_cache      = {}  # {topic: (Image_msg, timestamp)}
        self._image_cache_lock = threading.Lock()
        self._image_subs       = {}
        self._subs_lock        = threading.Lock()

        # ------------------------------------------------------------------
        # OpenAI / Azure / local Qwen client
        # ------------------------------------------------------------------
        if not OPENAI_AVAILABLE:
            rospy.logerr(
                "openai package not found. "
                "Install with: pip install openai"
            )
            self._client = None
        elif self.vqa_server_name == "qwen":
            self._client = OpenAI(
                api_key="dummy",
                base_url=f"http://{self.qwen_host}:{self.qwen_port}/v1",
                timeout=self.request_timeout,
            )
        elif self.vqa_server_name == "openai":
            self._client = OpenAI(
                api_key=self.openai_api_key,
                timeout=self.request_timeout,
            )
        elif self.vqa_server_name == "azure_openai":
            self._client = AzureOpenAI(
                api_key=self.azure_openai_api_key,
                azure_endpoint=self.azure_endpoint,
                api_version=self.api_version,
                timeout=self.request_timeout,
            )

        # ------------------------------------------------------------------
        # Subscribe to wrist camera at startup
        # ------------------------------------------------------------------
        rospy.loginfo(f"Subscribing to wrist camera topic: {self.wrist_image_topic}")
        self._ensure_subscription(self.wrist_image_topic)

        # ------------------------------------------------------------------
        # Wait briefly for the gripper-width service to be available
        # ------------------------------------------------------------------
        rospy.loginfo(
            f"Waiting up to {self.gripper_service_timeout}s for gripper width "
            f"service: {self.gripper_width_service}"
        )
        try:
            rospy.wait_for_service(
                self.gripper_width_service,
                timeout=self.gripper_service_timeout,
            )
            rospy.loginfo("Gripper width service available.")
        except rospy.ROSException:
            rospy.logwarn(
                f"Gripper width service '{self.gripper_width_service}' not "
                "available at startup; will retry per request."
            )

        # Cached ServiceProxy — uses RobotQuery (no request body)
        self._gripper_width_proxy = rospy.ServiceProxy(
            self.gripper_width_service, RobotQuery,
        )

        # ------------------------------------------------------------------
        # Advertise the service
        # ------------------------------------------------------------------
        self._service = rospy.Service(
            "/robot/perception/verify_grasp",
            RobotCommand,
            self._handle_request,
        )

        rospy.loginfo(
            "\nVerifyGraspServiceNode initialized.\n"
            f"  Service:           /robot/perception/verify_grasp\n"
            f"  Model:             {self.model_id}\n"
            f"  Wrist topic:       {self.wrist_image_topic}\n"
            f"  Gripper width svc: {self.gripper_width_service}\n"
            f"  Fully open / closed (mm): "
            f"{self.gripper_fully_open_mm:.1f} / "
            f"{self.gripper_fully_closed_mm:.1f}\n"
            f"  Min grasp width (mm):     {self.min_grasp_width_mm:.1f} "
            f"(auto-reject below this)\n"
            f"  OpenAI client:     {'available' if self._client else 'UNAVAILABLE'}\n"
            f"  numpy:             {'available' if NUMPY_AVAILABLE else 'UNAVAILABLE'}"
        )

    # ------------------------------------------------------------------
    # .env loading helpers
    # ------------------------------------------------------------------

    def _load_openai_key(self):
        try:
            import rospkg
            rospack = rospkg.RosPack()
            pkg_path = rospack.get_path("franka_robot_apis")
            env_path = f"{pkg_path}/.env"
            with open(env_path, "r") as f:
                for line in f:
                    if line.startswith("OPENAI_API_KEY="):
                        self.openai_api_key = line.strip().split("=", 1)[1]
                        return
            raise ValueError("OPENAI_API_KEY not found in .env file")
        except Exception as e:
            rospy.logerr(f"Failed to read OPENAI_API_KEY from .env: {e}")
            raise

    def _load_azure_config(self):
        try:
            import rospkg
            rospack = rospkg.RosPack()
            pkg_path = rospack.get_path("franka_robot_apis")
            env_path = f"{pkg_path}/.env"
            with open(env_path, "r") as f:
                for line in f:
                    if line.startswith("AZURE_OPENAI_API_KEY="):
                        self.azure_openai_api_key = line.strip().split("=", 1)[1]
                    elif line.startswith("AZURE_OPENAI_ENDPOINT="):
                        self.azure_endpoint = line.strip().split("=", 1)[1]
                    elif line.startswith("AZURE_OPENAI_DEPLOYMENT_NAME="):
                        self.azure_deployment_name = line.strip().split("=", 1)[1]
                    elif line.startswith("AZURE_OPENAI_API_VERSION="):
                        self.api_version = line.strip().split("=", 1)[1]

            missing = []
            if not hasattr(self, "azure_openai_api_key"):
                missing.append("AZURE_OPENAI_API_KEY")
            if not hasattr(self, "azure_endpoint"):
                missing.append("AZURE_OPENAI_ENDPOINT")
            if not hasattr(self, "azure_deployment_name"):
                missing.append("AZURE_OPENAI_DEPLOYMENT_NAME")
            if not hasattr(self, "api_version"):
                missing.append("AZURE_OPENAI_API_VERSION")
            if missing:
                raise ValueError(
                    f"Missing in .env: {', '.join(missing)}"
                )
        except Exception as e:
            rospy.logerr(f"Failed to read Azure OpenAI config from .env: {e}")
            raise

    # ------------------------------------------------------------------
    # Subscription management
    # ------------------------------------------------------------------

    def _ensure_subscription(self, topic):
        with self._subs_lock:
            if topic in self._image_subs:
                return

            rospy.loginfo(f"Creating image subscription for topic: {topic}")

            def _callback(msg, _topic=topic):
                with self._image_cache_lock:
                    self._image_cache[_topic] = (msg, time.time())

            sub = rospy.Subscriber(
                topic,
                Image,
                _callback,
                queue_size=1,
                buff_size=2**24,
            )
            self._image_subs[topic] = sub
            rospy.loginfo(f"Subscribed to image topic: {topic}")

    def _get_cached_image(self, topic):
        with self._image_cache_lock:
            entry = self._image_cache.get(topic)

        if entry is None:
            return None, (
                f"No image received yet on topic '{topic}'. "
                "Ensure the camera is publishing."
            )

        msg, timestamp = entry
        age = time.time() - timestamp

        if age > self.image_cache_timeout:
            return None, (
                f"Cached image for topic '{topic}' is stale "
                f"({age:.1f}s old, timeout={self.image_cache_timeout}s)."
            )

        return msg, ""

    # ------------------------------------------------------------------
    # Gripper width retrieval
    # ------------------------------------------------------------------

    def _get_gripper_width_mm(self):
        """
        Call the gripper-width service and return (width_mm, error_str).
        Width is converted from meters (as published) to millimeters.

        The service is robot_api_interfaces/RobotQuery — it has no request
        body and returns {result_code, data} where data is a JSON string.
        """
        try:
            # RobotQuery has an empty request — call with no args.
            response = self._gripper_width_proxy()
        except rospy.ServiceException as e:
            return None, f"Gripper width service call failed: {e}"
        except Exception as e:
            return None, f"Unexpected error calling gripper width service: {e}"

        # Check the response's result_code
        try:
            if response.result_code.result_code != ResultCode.SUCCESS:
                return None, (
                    f"Gripper width service returned failure: "
                    f"{response.result_code.message}"
                )
        except AttributeError:
            pass

        # Parse the data JSON
        try:
            data = json.loads(response.data)
        except (json.JSONDecodeError, ValueError) as e:
            return None, f"Could not parse gripper width response JSON: {e}"

        # Handle either a flat dict or a nested {"data": {...}} structure
        if isinstance(data, dict) and "data" in data and isinstance(data["data"], dict):
            data = data["data"]

        # The example response uses key "gripper_width" in meters.
        width = data.get("gripper_width")
        units = (data.get("units") or "meters").lower()

        if width is None:
            return None, f"Gripper width missing from response: {data}"

        try:
            width = float(width)
        except (TypeError, ValueError):
            return None, f"Gripper width is not numeric: {width!r}"

        # Normalize to mm
        if units in ("m", "meter", "meters"):
            width_mm = width * 1000.0
        elif units in ("mm", "millimeter", "millimeters"):
            width_mm = width
        else:
            # Heuristic: if value < 1.0 it's almost certainly meters.
            width_mm = width * 1000.0 if width < 1.0 else width

        return width_mm, ""

    # ------------------------------------------------------------------
    # Proprioception gate
    # ------------------------------------------------------------------

    def _proprio_gate(self, gripper_width_mm):
        """
        Return (success, reason) if proprio alone decides, else (None, None).

        Three unambiguous failure modes:
          - Gripper fully closed (closed on nothing).
          - Gripper width below the hard minimum threshold
            (effectively empty / on something too thin to be a real object).
          - Gripper fully open (didn't close at all).
        """
        # Fully closed -> nothing inside.
        if gripper_width_mm <= self.gripper_fully_closed_mm + 1.0:
            return False, (
                f"Gripper fully closed (width={gripper_width_mm:.1f} mm). "
                "Nothing held between fingers."
            )
        # Hard minimum -> effectively empty.
        if gripper_width_mm <= self.min_grasp_width_mm:
            return False, (
                f"Gripper width {gripper_width_mm:.1f} mm is at or below "
                f"the minimum grasp threshold ({self.min_grasp_width_mm:.1f} mm). "
                "Gripper is effectively empty."
            )
        # Fully open -> didn't close on anything.
        if gripper_width_mm >= self.gripper_fully_open_mm - 1.0:
            return False, (
                f"Gripper fully open (width={gripper_width_mm:.1f} mm). "
                "Grasp attempt did not close on an object."
            )
        return None, None

    # ------------------------------------------------------------------
    # Service handler
    # ------------------------------------------------------------------

    def _handle_request(self, request):
        rospy.loginfo(f"Received verify_grasp request: {request.req}")
        response = RobotCommandResponse()

        # --- 1. Parse request JSON ---
        try:
            req_data = json.loads(request.req)
        except (json.JSONDecodeError, ValueError) as e:
            return self._fail(response, f"Failed to parse request JSON: {e}")

        object_name = (req_data.get("object_name") or "").strip()
        if not object_name:
            return self._fail(
                response, "Missing or empty 'object_name' in request JSON."
            )

        max_tokens  = int(req_data.get("max_tokens",  self.default_max_tokens))
        temperature = float(req_data.get("temperature", self.default_temperature))

        # --- 2. Read gripper width ---
        width_mm, width_err = self._get_gripper_width_mm()
        if width_mm is None:
            return self._fail(response, width_err)

        rospy.loginfo(
            f"Gripper width: {width_mm:.1f} mm | object_name='{object_name}'"
        )

        # --- 3. Proprio gate (cheap rejection on fully-open / fully-closed) ---
        gate_success, gate_reason = self._proprio_gate(width_mm)
        if gate_success is False:
            rospy.loginfo(f"Proprio gate rejected grasp: {gate_reason}")
            return self._success(
                response,
                grasped=False,
                source="proprio",
                reason=gate_reason,
                gripper_width_mm=width_mm,
                object_name=object_name,
            )

        # --- 4. Retrieve wrist image ---
        img_msg, img_error = self._get_cached_image(self.wrist_image_topic)
        if img_msg is None:
            return self._fail(
                response,
                f"Wrist image unavailable "
                f"(topic '{self.wrist_image_topic}'): {img_error}",
            )

        # --- 5. Convert ROS image -> base64 JPEG ---
        try:
            b64_image, encoding_used = ros_image_to_base64_jpeg(
                img_msg, jpeg_quality=self.jpeg_quality,
            )
        except (ValueError, RuntimeError) as e:
            return self._fail(response, f"Image conversion failed: {e}")
        except Exception as e:
            rospy.logerr(traceback.format_exc())
            return self._fail(
                response, f"Unexpected error during image conversion: {e}",
            )

        rospy.loginfo(
            f"Wrist image converted (encoding={encoding_used}, "
            f"size={img_msg.width}x{img_msg.height}, "
            f"b64_len={len(b64_image)})"
        )

        # --- 6. Check VQA client ---
        if self._client is None:
            return self._fail(
                response,
                "VQA client is not available. "
                "Ensure 'openai' is installed: pip install openai",
            )

        # --- 7. Build prompt and call VLM ---
        prompt = build_grasp_prompt(object_name, width_mm)

        try:
            raw_answer, _usage = self._query_vlm(
                prompt=prompt,
                b64_image=b64_image,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as e:
            rospy.logerr(traceback.format_exc())
            return self._fail(response, f"VQA inference failed: {e}")

        verdict, reason = parse_yes_no(raw_answer)
        rospy.loginfo(
            f"VQA verdict: grasped={verdict} | reason={reason}"
        )

        return self._success(
            response,
            grasped=verdict,
            source="vqa",
            reason=reason,
            gripper_width_mm=width_mm,
            object_name=object_name,
            raw_response=raw_answer,
        )

    # ------------------------------------------------------------------
    # VLM inference
    # ------------------------------------------------------------------

    def _query_vlm(self, prompt, b64_image, max_tokens, temperature):
        rospy.loginfo(
            f"Sending request to VQA server "
            f"(model={self.model_id}, max_tokens={max_tokens}, "
            f"temperature={temperature})"
        )

        completion = self._client.chat.completions.create(
            model=self.model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64_image}",
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
            max_completion_tokens=max_tokens,
            temperature=temperature,
        )

        if not completion.choices:
            raise RuntimeError("VQA server returned an empty choices list.")

        answer = completion.choices[0].message.content

        usage = {}
        if completion.usage:
            usage = {
                "prompt_tokens":     completion.usage.prompt_tokens,
                "completion_tokens": completion.usage.completion_tokens,
                "total_tokens":      completion.usage.total_tokens,
            }
        return answer, usage

    # ------------------------------------------------------------------
    # Response helpers
    # ------------------------------------------------------------------

    def _success(self, response, grasped, source, reason,
                 gripper_width_mm, object_name, raw_response=None):
        payload = {
            "grasped": bool(grasped),
            "source": source,
            "reason": reason,
            "gripper_width_mm": round(float(gripper_width_mm), 2),
            "object_name": object_name,
        }
        if raw_response is not None:
            payload["raw_response"] = raw_response

        response.result_code.result_code = ResultCode.SUCCESS
        response.result_code.message     = "Success"
        response.data = json.dumps({
            "result_code": ResultCode.SUCCESS,
            "message": "Success",
            "data": payload,
        })
        return response

    def _fail(self, response, error_msg):
        rospy.logerr(f"verify_grasp service error: {error_msg}")
        response.result_code.result_code = ResultCode.FAILURE
        response.result_code.message     = error_msg
        response.data = json.dumps({
            "result_code": ResultCode.FAILURE,
            "message":     error_msg,
            "data":        {},
        })
        return response

    # ------------------------------------------------------------------
    # Spin
    # ------------------------------------------------------------------

    def spin(self):
        rospy.spin()


def main():
    rospy.init_node("verify_grasp_service", anonymous=False)

    try:
        rospy.loginfo("Creating VerifyGraspServiceNode...")
        node = VerifyGraspServiceNode()
        rospy.loginfo("VerifyGraspServiceNode spinning...")
        node.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS interrupt received - shutting down VerifyGraspServiceNode.")
    except Exception as e:
        rospy.logerr(f"Failed to start VerifyGraspServiceNode: {e}")
        rospy.logerr(traceback.format_exc())
    finally:
        rospy.loginfo("VerifyGraspServiceNode shutdown complete.")


if __name__ == "__main__":
    main()