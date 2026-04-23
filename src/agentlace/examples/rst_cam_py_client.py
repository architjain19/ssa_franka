import argparse
import cv2
import numpy as np
from agentlace.action import ActionClient, ActionConfig

parser = argparse.ArgumentParser()
parser.add_argument("--ip", default="localhost")
parser.add_argument("--port", type=int, default=6379)
args = parser.parse_args()

observation_keys = ['cam_123622270802_color', 'cam_947122060531_color', 'cam_032522250211_color', 'cam_123622270802_depth', 'cam_947122060531_depth', 'cam_032522250211_depth']
action_keys = ["command"]
port_number = args.port

config = ActionConfig(port_number=port_number, action_keys=action_keys, observation_keys=observation_keys)
client = ActionClient(args.ip, config=config)

print(f"Connected to {args.ip}:{port_number}, waiting for observations …")

while True:
    obs = client.obs()          # dict of all camera arrays
    if obs is None:
        continue

    # Display each color stream
    for key, frame in obs.items():
        if "color" in key and isinstance(frame, np.ndarray):
            cv2.imshow(key, frame)
        elif "depth" in key and isinstance(frame, np.ndarray):
            # Normalize depth for display
            disp = cv2.convertScaleAbs(frame, alpha=0.03)
            cv2.imshow(key, disp)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()