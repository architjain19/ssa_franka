# Shared Autonomy - Program Synthesized Code-as-Policies

Shared Autonomy (Code-as-Policies) - LLM-based program synthesis pipeline for task-specific shared autonomy, generating generalizable manipulation programs from a single human demonstration on the Franka robot supporting autonomous and manual handoffs based on user preference.

## Installation and Setup
Follow this guide - [setup conda env guide](setup_env_guide.txt)

## Usage

- SERL Franka Impedance Controller Launch
```bash
roslaunch serl_franka_controllers impedance.launch robot_ip:=172.16.0.2 load_gripper:=false
```

OR

- SERL Franka Position Joint Trajectory Controller Launch
```bash
roslaunch serl_franka_controllers joint.launch robot_ip:=172.16.0.2 load_gripper:=false
```

OR

- Main Core Launch for Franka Robot APIs Launch
```bash
roslaunch franka_robot_apis franka_robot_core.launch
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
rostopic pub -1 /cartesian_impedance_controller/equilibrium_pose geometry_msgs/PoseStamped "header:\n  seq: 0\n  stamp:\n    secs: 0\n    nsecs: 0\n  frame_id: '0'\npose:\n  position:\n    x: 0.5\n    y: 0.0\n    z: 0.5\n  orientation:\n    x: 0.8722\n    y: -0.4867\n    z: -0.0424\n    w: 0.0264"
```

- Sample command to contorl robot in joint position launch
```bash
rostopic pub -1 /position_joint_trajectory_controller/command trajectory_msgs/JointTrajectory "{joint_names: ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7'], points: [{positions: [0.20654942487601988, -0.14617635692439487, -0.1973732379448633, -2.0366618120164075, -0.02152645821703805, 1.7726474960871756, 1.030621456270764], time_from_start: {secs: 5, nsecs: 0}}]}"
```