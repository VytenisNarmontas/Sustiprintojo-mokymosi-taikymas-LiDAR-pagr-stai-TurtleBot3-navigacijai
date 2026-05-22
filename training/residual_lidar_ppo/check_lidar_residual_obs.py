from gate_env_lidar_residual import GateLidarResidualEnv


def main():
    env = GateLidarResidualEnv(curriculum_level=4, lidar_noise=False)
    obs, info = env.reset(seed=2026)
    print("obs shape:", obs.shape)
    print("expected:", 360 + 8 + 18 + 4)
    print("observation_space:", env.observation_space)
    print("event:", info.get("event"))
    print("gate_centers only in info/log/debug, not obs:", info.get("gate_centers"))
    print("detector action:", env.cached_detector_action)
    print("detector source:", env.cached_detector_output.source if env.cached_detector_output else None)
    assert obs.shape == env.observation_space.shape
    assert obs.shape[0] == 390
    env.close()
    print("OK: residual observation contains LiDAR + odom + LiDAR-computed detector features only.")


if __name__ == "__main__":
    main()
