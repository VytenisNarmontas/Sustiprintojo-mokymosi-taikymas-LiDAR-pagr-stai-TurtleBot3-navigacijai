from gate_env_no_cheat import GateLidarNoCheatEnv
import numpy as np


env = GateLidarNoCheatEnv(curriculum_level=4, lidar_noise=False)
obs, info = env.reset(seed=42)

print("observation_shape:", obs.shape)
print("observation_dim:", env.observation_space.shape[0])
print("lidar_beams:", env.obs_beams)
print("odom_extra_dim:", env.odom_dim)
print("obs_min/max:", float(np.min(obs)), float(np.max(obs)))
print("contains target features: NO")
print("contains gate centers in observation: NO")
print("contains gate index in observation: NO")
print("hidden gate centers only in info/debug:", info.get("gate_centers"))
print("scan ordering: beam 0 front, 90 left, 180 back, 270 right")
print("lidar_min_range:", env.lidar_min_range)
print("lidar_max_range:", env.lidar_max_range)
print("robot footprint:", env.robot_length, env.robot_width)
print("gate geometry:")
for i, g in enumerate(env.gates, start=1):
    off = g.clear_opening / 2.0 + env.pole_radius
    print(f"  gate{i}: x={g.x:.3f}, center_y={g.center_y:.3f}, poles={g.center_y-off:.3f}, {g.center_y+off:.3f}")

env.close()
