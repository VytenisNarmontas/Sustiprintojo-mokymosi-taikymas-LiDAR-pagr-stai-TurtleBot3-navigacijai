# Sustiprintojo mokymosi taikymas LiDAR pagrįstai TurtleBot3 navigacijai

This repository contains the implementation files and Gazebo result files used in the bachelor thesis:

**Sustiprintojo mokymosi taikymas LiDAR pagrįstai TurtleBot3 navigacijai trajektorijoja nužymėta vartais**  
**Application of Reinforcement Learning for LiDAR-based TurtleBot3 navigation along a gate-marked trajectory**

The project studies TurtleBot3 Burger navigation through a trajectory marked by vertical gates or poles. The compared methods are:

1. A rule-based LiDAR gate-middle tracking controller.
2. A PPO policy using raw LiDAR and odometry observations.
3. A PPO policy with a LiDAR-derived geometry helper module.

## Repository layout

- `tb3_gate_rl/` - ROS 2 Python nodes used in Gazebo evaluation.
- `training/raw_lidar_ppo/` - 2D Gymnasium environment and scripts for the raw LiDAR PPO method.
- `training/residual_lidar_ppo/` - 2D Gymnasium environment and scripts for the PPO method with LiDAR geometry helper features.
- `gazebo_100_results/` - final 100-episode Gazebo CSV result files used for the thesis comparison.
- `RESULTS.md` - short summary of the final Gazebo comparison.
- `REPRODUCING.md` - example commands for running the controllers and manager.

Trained PPO model archives are not stored directly in this repository. The code expects a model directory containing the PPO model and the matching observation-normalization statistics when running a trained controller.
