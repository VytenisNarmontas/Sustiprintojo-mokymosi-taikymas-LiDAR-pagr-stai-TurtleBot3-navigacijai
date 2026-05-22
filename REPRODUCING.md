# Reproducing the Gazebo tests

The original experiments were run with ROS 2 Humble, Gazebo, TurtleBot3 Burger, Python 3.10, Gymnasium and Stable-Baselines3.

Typical Gazebo evaluation uses three terminals:

1. Launch the Gazebo world.
2. Start one controller.
3. Start `gate_experiment_manager.py`, which resets the robot and gates, records episode outcomes, and writes the CSV file.

Example controller commands:

```bash
python -m tb3_gate_rl.lidar_gate_detector_v4d_controller
```

```bash
python -m tb3_gate_rl.ppo_lidar_odom_controller_no_cheat   --ros-args   -p model_dir:=${HOME}/rl_gate_train_no_cheat/models_level421_no_cheat_v9_BEST_430of500
```

```bash
python -m tb3_gate_rl.ppo_lidar_residual_controller   --ros-args   -p model_dir:=${HOME}/rl_gate_train/rl_gate_train_lidar_feature_v2/models_lidar_residual_final
```

Example manager command:

```bash
python -m tb3_gate_rl.gate_experiment_manager   --ros-args   -p episodes:=100   -p timeout_s:=120.0   -p output_path:=${HOME}/turtlebot3_ws/gazebo_100_results/output.csv   -p seed:=4200
```

The three final CSV files in `gazebo_100_results/` are the files used for the comparison in the thesis.
