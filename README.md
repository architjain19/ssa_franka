# Shared Autonomy - Program Synthesized Code-as-Policies

Shared Autonomy (Code-as-Policies) - LLM-based program synthesis pipeline for task-specific shared autonomy, generating generalizable manipulation programs from a single human demonstration on the Franka robot supporting autonomous and manual handoffs based on user preference.

## Installation and Setup
Follow this guide - [setup conda env guide](setup_env_guide.txt)

## Usage

- SERL Franka Impedance Controller Launch
```bash
roslaunch serl_franka_controllers impedance.launch robot_ip:=172.16.0.2 load_gripper:=false
```

- Agentplace Camera Server
```bash
python ~/archit/ssa_ws/src/agentlace/examples/rst_cam_server.py
```

- Agentplace Camera ROS1 Client
```bash
python ~/archit/ssa_ws/src/agentlace/examples/rst_cam_ros_client.py
```

## Debug

- Sample command to control robot in impedance launch
```bash
rostopic pub /cartesian_impedance_controller/equilibrium_pose geometry_msgs/PoseStamped "header:\n  seq: 0\n  stamp:\n    secs: 0\n    nsecs: 0\n  frame_id: '0'\npose:\n  position:\n    x: 0.5\n    y: 0.0\n    z: 0.5\n  orientation:\n    x: 0.8722\n    y: -0.4867\n    z: -0.0424\n    w: 0.0264"
```