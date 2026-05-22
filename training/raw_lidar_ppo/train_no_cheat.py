import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from gate_env_no_cheat import GateLidarNoCheatEnv


def make_env(level: int, lidar_noise: bool, use_full_360: bool):
    return lambda: GateLidarNoCheatEnv(
        curriculum_level=level,
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
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--load-dir", type=str, default=None)
    parser.add_argument("--no-lidar-noise", action="store_true")
    parser.add_argument("--pooled-72", action="store_true", help="Use 72 pooled beams instead of full 360 beams")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    args = parser.parse_args()

    save_dir = Path(args.save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)

    lidar_noise = not args.no_lidar_noise
    use_full_360 = not args.pooled_72

    env = make_vec_env(
        make_env(args.level, lidar_noise, use_full_360),
        n_envs=args.n_envs,
        seed=args.seed,
    )

    if args.load_dir is not None:
        load_dir = Path(args.load_dir).expanduser()
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
        print(f"Loaded model from {load_dir}")
    else:
        env = VecNormalize(
            env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=0.99,
        )

        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            seed=args.seed,
            learning_rate=args.learning_rate,
            n_steps=1024,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.20,
            ent_coef=0.010,
            vf_coef=0.50,
            max_grad_norm=0.5,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
            device="auto",
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(25_000 // args.n_envs, 1),
        save_path=str(save_dir / "checkpoints"),
        name_prefix="ppo_no_cheat_gate",
        save_vecnormalize=True,
    )

    model.learn(total_timesteps=args.total_timesteps, callback=checkpoint_callback, progress_bar=True)

    model.save(str(save_dir / "ppo_gate.zip"))
    env.save(str(save_dir / "vecnormalize.pkl"))
    save_vecnormalize_stats(env, save_dir)

    # Metadata for avoiding accidental cheat regressions.
    (save_dir / "README_MODEL.txt").write_text(
        "NO-CHEAT PPO model. Observation = 360 LiDAR + relative odometry + velocity/previous command.\n"
        "No gate centers, no target point, no gate index are included in policy observation.\n"
        f"curriculum_level={args.level}\n"
        f"lidar_noise={lidar_noise}\n"
        f"use_full_360={use_full_360}\n"
    )

    env.close()
    print(f"Saved no-cheat model to: {save_dir}")


if __name__ == "__main__":
    main()
