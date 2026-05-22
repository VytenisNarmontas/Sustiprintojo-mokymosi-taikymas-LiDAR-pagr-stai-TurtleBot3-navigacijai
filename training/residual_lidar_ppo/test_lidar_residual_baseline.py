import argparse
import time

import numpy as np

from gate_env_lidar_residual import GateLidarResidualEnv, RandomLevelGateLidarResidualEnv, CloseStartGateLidarResidualEnv


def make_env(args):
    kwargs = dict(lidar_noise=not args.no_lidar_noise, use_full_360=not args.pooled_72, render_mode="human" if args.render else None)
    if args.close_start:
        return CloseStartGateLidarResidualEnv(**kwargs)
    if args.mixed_levels:
        levels = [int(x.strip()) for x in args.mixed_levels.split(",") if x.strip()]
        return RandomLevelGateLidarResidualEnv(levels=levels, **kwargs)
    return GateLidarResidualEnv(curriculum_level=args.level, **kwargs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--level", type=int, default=4)
    p.add_argument("--mixed-levels", type=str, default="")
    p.add_argument("--close-start", action="store_true")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--no-lidar-noise", action="store_true")
    p.add_argument("--pooled-72", action="store_true")
    p.add_argument("--render", action="store_true")
    p.add_argument("--sleep", type=float, default=0.01)
    args = p.parse_args()

    env = make_env(args)
    counts = {"success": 0, "collision": 0, "missed_gate": 0, "stalled": 0, "spinning": 0, "timeout": 0}
    rewards = []
    gates = []

    zero_residual = np.zeros(2, dtype=np.float32)

    for ep in range(1, args.episodes + 1):
        obs, info = env.reset(seed=args.seed + ep)
        done = False
        total_reward = 0.0
        last_info = info
        while not done:
            obs, reward, terminated, truncated, info = env.step(zero_residual)
            total_reward += float(reward)
            done = bool(terminated or truncated)
            last_info = info
            if args.render:
                env.render()
                time.sleep(args.sleep)

        event = str(last_info.get("event", "unknown"))
        if event in counts:
            counts[event] += 1
        rewards.append(total_reward)
        gates.append(int(last_info.get("passed_gates", 0)))
        print(
            f"episode={ep} event={event} reward={total_reward:.2f} "
            f"passed_gates={last_info.get('passed_gates')} min_lidar={float(last_info.get('min_lidar', 0.0)):.2f} "
            f"source={last_info.get('detector_source')} conf={float(last_info.get('detector_confidence', 0.0)):.2f}"
        )

    env.close()
    print("\nSummary")
    print("-------")
    print(f"successes     : {counts['success']}/{args.episodes}")
    print(f"collisions    : {counts['collision']}")
    print(f"missed_gate   : {counts['missed_gate']}")
    print(f"stalled       : {counts['stalled']}")
    print(f"spinning      : {counts['spinning']}")
    print(f"timeouts      : {counts['timeout']}")
    print(f"mean reward   : {np.mean(rewards):.2f}")
    print(f"mean gates    : {np.mean(gates):.2f}")


if __name__ == "__main__":
    main()
