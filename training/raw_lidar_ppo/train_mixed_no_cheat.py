import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from gate_env_no_cheat import GateLidarNoCheatEnv


class RandomLevelGateEnv(GateLidarNoCheatEnv):
    def __init__(self, levels, lidar_noise=True, use_full_360=True):
        self.random_levels = [int(x) for x in levels]
        super().__init__(
            curriculum_level=self.random_levels[0],
            lidar_noise=lidar_noise,
            use_full_360=use_full_360,
        )

    def reset(self, *, seed=None, options=None):
        self.curriculum_level = int(np.random.choice(self.random_levels))
        return super().reset(seed=seed, options=options)


def make_env(levels, lidar_noise, use_full_360):
    return lambda: RandomLevelGateEnv(
        levels=levels,
        lidar_noise=lidar_noise,
        use_full_360=use_full_360,
    )


def save_vecnormalize_stats(vecnorm: VecNormalize, save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        save_dir / "vecnormalize_stats.npz",
        obs_mean=vecnorm.obs_rms.mean,
        obs_var=vecnorm.obs_rms.var,
        epsilon=vecnorm.epsilon,
        clip_obs=vecnorm.clip_obs,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", type=str, required=True, help="Comma-separated levels, example: 421,422,423,424")
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--load-dir", type=str, required=True)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--no-lidar-noise", action="store_true")
    parser.add_argument("--pooled-72", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    levels = [int(x.strip()) for x in args.levels.split(",") if x.strip()]
    save_dir = Path(args.save_dir).expanduser()
    load_dir = Path(args.load_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)

    lidar_noise = not args.no_lidar_noise
    use_full_360 = not args.pooled_72

    env = make_vec_env(
        make_env(levels, lidar_noise, use_full_360),
        n_envs=args.n_envs,
        seed=args.seed,
    )

    env = VecNormalize.load(str(load_dir / "vecnormalize.pkl"), env)
    env.training = True
    env.norm_reward = True

    model = PPO.load(
        str(load_dir / "ppo_gate.zip"),
        env=env,
        device="auto",
        custom_objects={"learning_rate": args.learning_rate},
    )
    model.learning_rate = args.learning_rate
    model.lr_schedule = lambda _: args.learning_rate

    checkpoint_callback = CheckpointCallback(
        save_freq=max(25_000 // args.n_envs, 1),
        save_path=str(save_dir / "checkpoints"),
        name_prefix="ppo_no_cheat_mixed",
        save_vecnormalize=True,
    )

    print(f"Training mixed levels: {levels}")
    print(f"Loading from: {load_dir}")
    print(f"Saving to: {save_dir}")
    print(f"Learning rate: {args.learning_rate}")

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True,
    )

    model.save(str(save_dir / "ppo_gate.zip"))
    env.save(str(save_dir / "vecnormalize.pkl"))
    save_vecnormalize_stats(env, save_dir)

    (save_dir / "README_MODEL.txt").write_text(
        "NO-CHEAT PPO mixed-curriculum model. Observation = 360 LiDAR + relative odometry + velocity/previous command.\n"
        "No gate centers, no target point, no gate index are included in policy observation.\n"
        f"mixed_levels={levels}\n"
        f"lidar_noise={lidar_noise}\n"
        f"use_full_360={use_full_360}\n"
    )

    env.close()
    print(f"Saved mixed no-cheat model to: {save_dir}")


if __name__ == "__main__":
    main()
