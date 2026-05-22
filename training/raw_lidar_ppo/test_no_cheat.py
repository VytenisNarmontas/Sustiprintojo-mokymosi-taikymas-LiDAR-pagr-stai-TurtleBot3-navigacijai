import argparse
import math
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from gate_env_no_cheat import GateLidarNoCheatEnv


def load_norm_stats(model_dir: Path):
    path = model_dir / "vecnormalize_stats.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    data = np.load(path)
    return {
        "mean": data["obs_mean"].astype(np.float32),
        "var": data["obs_var"].astype(np.float32),
        "epsilon": float(data["epsilon"]),
        "clip_obs": float(data["clip_obs"]),
    }


def normalize_obs(obs, stats):
    obs = obs.astype(np.float32)
    obs = (obs - stats["mean"]) / np.sqrt(stats["var"] + stats["epsilon"])
    obs = np.clip(obs, -stats["clip_obs"], stats["clip_obs"])
    return obs.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--level", type=int, default=4)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--no-lidar-noise", action="store_true")
    parser.add_argument("--pooled-72", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    model_dir = Path(args.model_dir).expanduser()
    model = PPO.load(str(model_dir / "ppo_gate.zip"), device="cpu")
    stats = load_norm_stats(model_dir)

    results = []

    for ep in range(1, args.episodes + 1):
        env = GateLidarNoCheatEnv(
            curriculum_level=args.level,
            lidar_noise=not args.no_lidar_noise,
            use_full_360=not args.pooled_72,
            render_mode="human" if args.render else None,
        )
        obs, info = env.reset(seed=args.seed + ep)

        total_reward = 0.0
        steps = 0
        path_length = 0.0
        last_x = float(env.x)
        last_y = float(env.y)
        min_lidar = 999.0
        event = "unknown"
        passed_gates = 0

        while True:
            norm_obs = normalize_obs(obs, stats)
            action, _ = model.predict(norm_obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if args.render:
                env.render()

            x = float(env.x)
            y = float(env.y)
            step_dist = math.hypot(x - last_x, y - last_y)
            if step_dist < 0.30:
                path_length += step_dist
            last_x, last_y = x, y

            total_reward += float(reward)
            steps += 1
            min_lidar = min(min_lidar, float(np.min(env.last_scan)))

            if terminated or truncated:
                event = info.get("event", "unknown")
                passed_gates = int(info.get("passed_gates", 0))
                break

        sim_time = steps * env.control_dt
        avg_speed = path_length / sim_time if sim_time > 1e-9 else 0.0
        results.append((event, total_reward, passed_gates, steps, sim_time, path_length, avg_speed, min_lidar))
        print(
            f"episode={ep} event={event} reward={total_reward:.2f} "
            f"passed_gates={passed_gates} steps={steps} time={sim_time:.2f}s "
            f"path={path_length:.2f}m avg_speed={avg_speed:.3f} min_lidar={min_lidar:.2f}"
        )
        env.close()

    n = len(results)
    successes = sum(1 for r in results if r[0] == "success")
    collisions = sum(1 for r in results if r[0] == "collision")
    missed = sum(1 for r in results if r[0] == "missed_gate")
    stalled = sum(1 for r in results if r[0] == "stalled")
    spinning = sum(1 for r in results if r[0] == "spinning")
    timeouts = sum(1 for r in results if r[0] == "timeout")

    print("\nSummary")
    print("-------")
    print(f"successes     : {successes}/{n}")
    print(f"collisions    : {collisions}")
    print(f"missed_gate   : {missed}")
    print(f"stalled       : {stalled}")
    print(f"spinning      : {spinning}")
    print(f"timeouts      : {timeouts}")
    print(f"mean reward   : {np.mean([r[1] for r in results]):.2f}")
    print(f"mean gates    : {np.mean([r[2] for r in results]):.2f}")
    print(f"mean time     : {np.mean([r[4] for r in results]):.2f}s")
    print(f"mean speed    : {np.mean([r[6] for r in results]):.3f}m/s")


if __name__ == "__main__":
    main()
