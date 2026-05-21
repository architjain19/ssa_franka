#!/usr/bin/env python3
"""
ROS1 Noetic service node that provides a vision-language query interface using Qwen2.5-VL.

Service: /robot/perception/get_vqa_response  (robot_api_interfaces/RobotCommand)

Request JSON format:
{
    "prompt": "What objects are on the table?",
    "camera": "scene",        # "wrist" or "scene" (default: "scene")
    "max_tokens": 512,        # optional, default 512
    "temperature": 0.1        # optional, default 0.1
}

Response JSON format:
{
    "result_code": 0,
    "message": "Success",
    "data": {
        "answer": "I can see a cup, a plate...",
        "camera": "scene"
    }
}

ROS1 usage:
    rosrun franka_robot_apis get_vqa_service.py

    # Scene camera (default)
    rosservice call /robot/perception/get_vqa_response \
        '{"req": "{\"prompt\": \"Is there an aruco marker?\", \"camera\": \"scene\"}"}'

    # Wrist camera
    rosservice call /robot/perception/get_vqa_response \
        '{"req": "{\"prompt\": \"What is below the gripper?\", \"camera\": \"wrist\"}"}'
"""

import json
import base64
import threading
import time
import traceback

import rospy
from sensor_msgs.msg import Image
from robot_api_interfaces.srv import RobotCommand, RobotCommandResponse
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

import cv2  # for JPEG encoding


# Allowed values for the "camera" field in the request JSON.
VALID_CAMERAS = ("wrist", "scene")


# ---------------------------------------------------------------------------
# Helper: ROS Image message -> base64-encoded JPEG string
# ---------------------------------------------------------------------------

def ros_image_to_base64_jpeg(msg, jpeg_quality=90):
    """
    Convert a ROS1 sensor_msgs/Image to a base64-encoded JPEG string.

    Supports encodings: rgb8, bgr8, rgba8, bgra8, mono8, 8uc1, mono16, 16uc1.

    Args:
        msg (sensor_msgs.msg.Image): incoming ROS image message
        jpeg_quality (int): JPEG compression quality 0-100

    Returns:
        tuple[str, str]: (base64_jpeg_string, encoding_name_used)

    Raises:
        RuntimeError: if numpy is missing or cv2.imencode fails
        ValueError:   if the image encoding is unsupported
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

    # -- Raw bytes -> BGR numpy array ----------------------------------------
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

    # -- BGR numpy array -> JPEG bytes -> base64 string -----------------------
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    success, buffer = cv2.imencode(".jpg", img_bgr, encode_params)
    if not success:
        raise RuntimeError("cv2.imencode failed to encode image as JPEG")

    b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")
    return b64, encoding


# ---------------------------------------------------------------------------
# Main service class
# ---------------------------------------------------------------------------

class VQAServiceNode:
    """
    ROS1 service node wrapping a Qwen2.5-VL vLLM server.

    VQAServiceNode architecture:
      - Subscribes to wrist and scene camera topics at startup
      - Caches the latest frame per topic with a staleness timeout
      - Exposes /robot/perception/get_vqa_response (RobotCommand.srv)
      - Selects the camera per request via the "camera" field
        ("wrist" or "scene"); defaults to ~default_camera if unspecified.
    """

    def __init__(self):
        # ------------------------------------------------------------------
        # ROS1 Parameters  (rospy.get_param)
        # ------------------------------------------------------------------
        self.qwen_host           = rospy.get_param("~qwen_host",          "10.158.54.164")
        self.qwen_port           = rospy.get_param("~qwen_port",          8000)

        # Per-camera image topics. Both can be remapped via params.
        self.scene_image_topic   = rospy.get_param("~scene_image_topic", "/zed/scene/color/image_raw")
        self.wrist_image_topic   = rospy.get_param("~wrist_image_topic", "/zed/wrist/color/image_raw")

        # Which camera to use when the request does not specify one.
        self.default_camera      = rospy.get_param("~default_camera", "scene").lower()
        if self.default_camera not in VALID_CAMERAS:
            rospy.logwarn(
                f"~default_camera='{self.default_camera}' is invalid; "
                f"falling back to 'scene'. Valid options: {VALID_CAMERAS}"
            )
            self.default_camera = "scene"

        # Mapping from camera name -> topic.
        self.camera_topics = {
            "scene": self.scene_image_topic,
            "wrist": self.wrist_image_topic,
        }

        self.image_cache_timeout = float(rospy.get_param("~image_cache_timeout", 1.0))
        self.default_max_tokens  = int(rospy.get_param("~default_max_tokens",  512))
        self.default_temperature = float(rospy.get_param("~default_temperature", 0.1))
        self.jpeg_quality        = int(rospy.get_param("~jpeg_quality",        90))
        self.request_timeout     = float(rospy.get_param("~request_timeout",   15.0))
        self.vqa_server_name     = rospy.get_param("~vqa_server_name", "azure_openai").lower()  # "qwen", "openai", or "azure_openai"

        if self.vqa_server_name == "qwen":
            self.model_id = rospy.get_param(
                "~model_id",
                "/home/hcrlab/archit/qwen-vl-stretch/models/Qwen2.5-VL-7B-Instruct-AWQ",
            )
            rospy.loginfo("Using local VQA for VQA.")
        elif self.vqa_server_name == "openai":
            # read OPENAI_API_KEY from .env
            try:
                # switch to package share directory to find .env reliably
                import rospkg
                rospack = rospkg.RosPack()
                pkg_path = rospack.get_path("franka_robot_apis")
                env_path = f"{pkg_path}/.env"
                with open(env_path, "r") as f:
                    for line in f:
                        if line.startswith("OPENAI_API_KEY="):
                            self.openai_api_key = line.strip().split("=", 1)[1]
                            break
                    else:
                        raise ValueError("OPENAI_API_KEY not found in .env file")
            except Exception as e:
                rospy.logerr(f"Failed to read OPENAI_API_KEY from .env: {e}")
            
            self.model_id = rospy.get_param("~model_id", "gpt-4.5-preview")
            rospy.loginfo("Using OpenAI API for VQA.")
        elif self.vqa_server_name == "azure_openai":
            # read AZURE_OPENAI_API_KEY from .env
            try:
                # switch to package share directory to find .env reliably
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
                    else:
                        if not hasattr(self, 'azure_openai_api_key'):
                            raise ValueError("AZURE_OPENAI_API_KEY not found in .env file")
                        if not hasattr(self, 'azure_endpoint'):
                            raise ValueError("AZURE_OPENAI_ENDPOINT not found in .env file")
                        if not hasattr(self, 'azure_deployment_name'):
                            raise ValueError("AZURE_OPENAI_DEPLOYMENT_NAME not found in .env file")
                        if not hasattr(self, 'api_version'):
                            raise ValueError("AZURE_OPENAI_API_VERSION not found in .env file")
            except Exception as e:
                rospy.logerr(f"Failed to read Azure OpenAI config from .env: {e}")
            
            self.model_id = self.azure_deployment_name  # For Azure OpenAI, model_id is the deployment name
            rospy.loginfo("Using Azure OpenAI API for VQA.")
        else:
            rospy.logerr(f"Invalid vqa_server_name: '{self.vqa_server_name}'. Must be 'qwen', 'openai', or 'azure_openai'. Defaulting to 'qwen'.")
            self.vqa_server_name = "qwen"
            self.model_id = rospy.get_param(
                "~model_id",
                "/home/hcrlab/archit/qwen-vl-stretch/models/Qwen2.5-VL-7B-Instruct-AWQ",
            )

        # ------------------------------------------------------------------
        # Internal state
        # ------------------------------------------------------------------
        # {topic_name: (Image_msg, timestamp_float)}
        self._image_cache      = {}
        self._image_cache_lock = threading.Lock()

        # {topic_name: rospy.Subscriber}
        self._image_subs      = {}
        self._subs_lock       = threading.Lock()

        # ------------------------------------------------------------------
        # OpenAI client -> local vLLM (Qwen2.5-VL) server OR OpenAI API
        # ------------------------------------------------------------------
        if not OPENAI_AVAILABLE:
            rospy.logerr(
                "openai package not found. "
                "Install with: pip install openai"
            )
            self._client = None
        else:
            if self.vqa_server_name == "qwen":
                self._client = OpenAI(
                    api_key="dummy",
                    base_url=f"http://{self.qwen_host}:{self.qwen_port}/v1",
                    timeout=self.request_timeout,
                )
            elif self.vqa_server_name == "openai":
                self._client = OpenAI(
                    api_key=self.openai_api_key,
                    # NO base_url - defaults to api.openai.com
                    timeout=self.request_timeout,
            )
            elif self.vqa_server_name == "azure_openai":
                self._client = AzureOpenAI(
                    api_key=self.azure_openai_api_key,
                    # The endpoint provided in your Azure portal (e.g., https://your-resource.openai.azure.com/)
                    azure_endpoint=self.azure_endpoint, 
                    # Must specify an API version (e.g., "2024-08-01-preview" or "2025-01-01-preview")
                    api_version=self.api_version,
                    timeout=self.request_timeout,
                )

        # ------------------------------------------------------------------
        # Subscribe to both camera topics at startup so frames are ready
        # whichever camera the first request picks.
        # ------------------------------------------------------------------
        for cam_name, topic in self.camera_topics.items():
            rospy.loginfo(f"Subscribing to {cam_name} camera topic: {topic}")
            self._ensure_subscription(topic)

        # ------------------------------------------------------------------
        # Advertise the ROS1 service
        # ------------------------------------------------------------------
        self._service = rospy.Service(
            "/robot/perception/get_vqa_response",
            RobotCommand,
            self._handle_request,
        )

        rospy.loginfo(
            "\nVQAServiceNode initialized.\n"
            f"  Service:        /robot/perception/get_vqa_response\n"
            f"  Model:          {self.model_id}\n"
            f"  Scene topic:    {self.scene_image_topic}\n"
            f"  Wrist topic:    {self.wrist_image_topic}\n"
            f"  Default camera: {self.default_camera}\n"
            f"  OpenAI client:  {'available' if self._client else 'UNAVAILABLE'}\n"
            f"  numpy:          {'available' if NUMPY_AVAILABLE else 'UNAVAILABLE'}"
        )

    # ------------------------------------------------------------------
    # Subscription management
    # ------------------------------------------------------------------

    def _ensure_subscription(self, topic):
        """
        Create a rospy.Subscriber for *topic* if one does not already exist.
        Thread-safe.
        """
        with self._subs_lock:
            if topic in self._image_subs:
                return

            rospy.loginfo(f"Creating image subscription for topic: {topic}")

            # Use a default-argument capture so each closure binds its own topic.
            def _callback(msg, _topic=topic):
                with self._image_cache_lock:
                    self._image_cache[_topic] = (msg, time.time())

            sub = rospy.Subscriber(
                topic,
                Image,
                _callback,
                queue_size=1,
                buff_size=2**24,  # 16 MB — avoids dropped frames on HD topics
            )
            self._image_subs[topic] = sub
            rospy.loginfo(f"Subscribed to image topic: {topic}")

    def _get_cached_image(self, topic):
        """
        Return (Image_msg, error_str).  error_str is empty on success.
        """
        with self._image_cache_lock:
            entry = self._image_cache.get(topic)

        if entry is None:
            return None, (
                f"No image received yet on topic '{topic}'. "
                "Ensure the camera is publishing and the topic name is correct."
            )

        msg, timestamp = entry
        age = time.time() - timestamp

        if age > self.image_cache_timeout:
            return None, (
                f"Cached image for topic '{topic}' is stale "
                f"({age:.1f}s old, timeout={self.image_cache_timeout}s). "
                "Check that the camera is still publishing."
            )

        return msg, ""

    # ------------------------------------------------------------------
    # Service handler
    # ------------------------------------------------------------------

    def _handle_request(self, request):
        """
        rospy.Service callback — called in a dedicated thread per request.

        Args:
            request (RobotCommand.Request): .req contains the JSON string

        Returns:
            RobotCommand.Response
        """
        rospy.loginfo(f"Received VQA service request: {request.req}")

        response = RobotCommandResponse()

        # --- 1. Parse request JSON ---
        try:
            req_data = json.loads(request.req)
        except (json.JSONDecodeError, ValueError) as e:
            return self._fail(response, f"Failed to parse request JSON: {e}")

        prompt = req_data.get("prompt", "").strip()
        if not prompt:
            return self._fail(response, "Missing or empty 'prompt' in request JSON.")

        # --- 2. Resolve which camera/topic to use ---
        camera = req_data.get("camera", self.default_camera)
        if not isinstance(camera, str):
            return self._fail(
                response,
                f"'camera' must be a string, got {type(camera).__name__}.",
            )
        camera = camera.strip().lower()
        if camera not in self.camera_topics:
            return self._fail(
                response,
                f"Invalid 'camera' value: '{camera}'. "
                f"Must be one of: {list(self.camera_topics.keys())}.",
            )
        image_topic = self.camera_topics[camera]

        max_tokens  = int(req_data.get("max_tokens",  self.default_max_tokens))
        temperature = float(req_data.get("temperature", self.default_temperature))

        rospy.loginfo(
            f"Query params - camera: '{camera}', topic: '{image_topic}', "
            f"max_tokens: {max_tokens}, temperature: {temperature}\n"
            f"Prompt: {prompt}"
        )

        # --- 3. Retrieve latest image ---
        img_msg, img_error = self._get_cached_image(image_topic)
        if img_msg is None:
            return self._fail(
                response,
                f"Image unavailable for camera '{camera}' "
                f"(topic '{image_topic}'): {img_error}",
            )

        # --- 4. Convert ROS image -> base64 JPEG ---
        try:
            b64_image, encoding_used = ros_image_to_base64_jpeg(
                img_msg, jpeg_quality=self.jpeg_quality
            )
        except (ValueError, RuntimeError) as e:
            return self._fail(response, f"Image conversion failed: {e}")
        except Exception as e:
            rospy.logerr(traceback.format_exc())
            return self._fail(response, f"Unexpected error during image conversion: {e}")

        rospy.loginfo(
            f"Image converted successfully "
            f"(camera={camera}, encoding={encoding_used}, "
            f"size={img_msg.width}x{img_msg.height}, "
            f"b64_len={len(b64_image)})"
        )

        # --- 5. Check OpenAI client ---
        if self._client is None:
            return self._fail(
                response,
                "OpenAI client is not available. "
                "Ensure 'openai' is installed: pip install openai",
            )

        # --- 6. Call VQA server ---
        try:
            answer, _usage = self._query_qwen(
                prompt=prompt,
                b64_image=b64_image,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as e:
            rospy.logerr(traceback.format_exc())
            return self._fail(response, f"VQA inference failed: {e}")

        # --- 7. Build success response ---
        rospy.loginfo(f"VQA answer ({camera}): {answer}")

        response.result_code.result_code = ResultCode.SUCCESS
        response.result_code.message     = "Success"
        response.data = json.dumps({
            "result_code": ResultCode.SUCCESS,
            "message": "Success",
            "data": {
                "answer": answer,
                "camera": camera,
            },
        })
        return response

    # ------------------------------------------------------------------
    # VQA inference
    # ------------------------------------------------------------------

    def _query_qwen(self, prompt, b64_image, max_tokens, temperature):
        """
        Send prompt + base64 image to the VQA vLLM server.

        Returns:
            tuple[str, dict]: (answer_string, usage_dict)

        Raises:
            RuntimeError: on API error or empty choices list
        """
        rospy.loginfo(
            f"Sending request to VQA server "
            f"(model={self.model_id}, max_tokens={max_tokens}, temperature={temperature})"
        )
        if self.vqa_server_name == "azure_openai":
            self.model_id = f"{self.azure_deployment_name}"  # For Azure OpenAI, model_id is the deployment name

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
    # Helpers
    # ------------------------------------------------------------------

    def _fail(self, response, error_msg):
        """
        Populate *response* as a failure and log the error.

        Args:
            response (RobotCommandResponse): response object to mutate
            error_msg (str): human-readable error

        Returns:
            RobotCommandResponse: the populated failure response
        """
        rospy.logerr(f"VQA service error: {error_msg}")
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
        """Block until ROS shutdown."""
        rospy.spin()

def main():
    """Initialize and spin the VQAServiceNode."""
    rospy.init_node("get_vqa_service", anonymous=False)

    try:
        rospy.loginfo("Creating VQAServiceNode...")
        node = VQAServiceNode()
        rospy.loginfo("VQAServiceNode spinning...")
        node.spin()

    except rospy.ROSInterruptException:
        rospy.loginfo("ROS interrupt received — shutting down VQAServiceNode.")
    except Exception as e:
        rospy.logerr(f"Failed to start VQAServiceNode: {e}")
        rospy.logerr(traceback.format_exc())
    finally:
        rospy.loginfo("VQAServiceNode shutdown complete.")


if __name__ == "__main__":
    main()