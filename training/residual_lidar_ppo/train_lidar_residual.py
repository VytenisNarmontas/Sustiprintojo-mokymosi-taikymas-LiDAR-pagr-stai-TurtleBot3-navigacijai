import argparse
from pathlib import Path
from typing import Optional

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from gate_env_lidar_residual import (
    CloseStartGateLidarResidualEnv,
    GateLidarResidualEnv,
    RandomLevelGateLidarResidualEnv,
)


def make_env(args):
    lidar_noise = not args.no_lidar_noise
    use_full_360 = not args.pooled_72
    kwargs = dict(
        lidar_noise=lidar_noise,
        use_full_360=use_full_360,
        residual_linear_scale=args.residual_linear_scale,
        residual_angular_scale=args.residual_angular_scale,
    )
    if args.close_start:
        return lambda: CloseStartGateLidarResidualEnv(**kwargs)
    if args.mixed_levels:
        levels = [int(x.strip()) for x in args.mixed_levels.split(",") if x.strip()]
        return lambda: RandomLevelGateLidarResidualEnv(levels=levels, **kwargs)
    return lambda: GateLidarResidualEnv(curriculum_level=args.level, **kwargs)


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
    parser.add_argument("--mixed-levels", type=str, default="")
    parser.add_argument("--close-start", action="store_true")
    parser.add_argument("--total-timesteps", type=int, default=120_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--load-dir", type=str, default="")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--no-lidar-noise", action="store_true")
    parser.add_argument("--pooled-72", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--residual-linear-scale", type=float, default=0.12)
    parser.add_argument("--residual-angular-scale", type=float, default=0.22)
    args = parser.parse_args()

    save_dir = Path(args.save_dir).expanduser()
    load_dir: Optional[Path] = Path(args.load_dir).expanduser() if args.load_dir else None
    save_dir.mkdir(parents=True, exist_ok=True)

    env = make_vec_env(make_env(args), n_envs=args.n_envs, seed=args.seed)

    if load_dir:
        env = VecNormalize.load(str(load_dir / "vecnormalize.pkl"), env)
        env.training = True
        env.norm_reward = True
        model = PPO.load(
            str(load_dir / "ppo_gate.zip"),
            env=env,
            device=args.device,
            custom_objects={"learning_rate": args.learning_rate},
        )
        model.learning_rate = args.learning_rate
        model.lr_schedule = lambda _: args.learning_rate
        print(f"Loaded residual PPO from: {load_dir}")
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            device=args.device,
            learning_rate=args.learning_rate,
            n_steps=1024,
            batch_size=256,
            n_epochs=6,
            gamma=0.995,
            gae_lambda=0.95,
            clip_range=0.16,
            ent_coef=0.001,
            vf_coef=0.50,
            max_grad_norm=0.5,
            policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
        )
        print("Created new residual PPO model.")

    callback = CheckpointCallback(
        save_freq=max(25_000 // args.n_envs, 1),
        save_path=str(save_dir / "checkpoints"),
        name_prefix="ppo_lidar_residual",
        save_vecnormalize=True,
    )

    print(f"Saving to: {save_dir}")
    print(f"level={args.level} mixed={args.mixed_levels!r} close_start={args.close_start}")
    print(f"no_lidar_noise={args.no_lidar_noise} device={args.device} n_envs={args.n_envs}")
    print(f"residual scales: linear={args.residual_linear_scale} angular={args.residual_angular_scale}")

    model.learn(total_timesteps=args.total_timesteps, callback=callback, progress_bar=True)

    model.save(str(save_dir / "ppo_gate.zip"))
    env.save(str(save_dir / "vecnormalize.pkl"))
    save_vecnormalize_stats(env, save_dir)

    mode = "close_start" if args.close_start else (f"mixed_levels={args.mixed_levels}" if args.mixed_levels else f"level={args.level}")
    (save_dir / "README_MODEL.txt").write_text(
        "NO-CHEAT residual PPO over LiDAR-only gate detector.\n"
        "Final action = LiDAR detector action + small PPO residual.\n"
        "Observation = 360 LiDAR + odom/velocity + LiDAR-computed detector features/action.\n"
        "No true gate centers, no pole coordinates, no target point, no gate index, no /gate_experiment/gates.\n"
        f"mode={mode}\n"
        f"lidar_noise={not args.no_lidar_noise}\n"
        f"use_full_360={not args.pooled_72}\n"
        f"residual_linear_scale={args.residual_linear_scale}\n"
        f"residual_angular_scale={args.residual_angular_scale}\n"
    )

    env.close()
    print(f"Saved residual PPO model to: {save_dir}")


if __name__ == "__main__":
    main()
