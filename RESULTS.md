# Final Gazebo Results

The final comparison used 100 Gazebo episodes with the same scenario seed family for all three methods.

| Method | Result file | Successes | Collisions | Missed gates | Timeouts |
|---|---:|---:|---:|---:|---:|
| Rule-based LiDAR gate-middle controller | `gazebo_100_results/classical_lidar_detector_NO_GUARD_ABLATION_100_seed4200_chunked.csv` | 95/100 | 5 | 0 | 0 |
| Raw LiDAR + odometry PPO | `gazebo_100_results/old_raw_lidar_ppo_100_seed4200.csv` | 48/100 | 37 | 0 | 15 |
| PPO with LiDAR geometry helper | `gazebo_100_results/residual_lidar_detector_ppo_100_seed4200.csv` | 95/100 | 2 | 3 | 0 |
