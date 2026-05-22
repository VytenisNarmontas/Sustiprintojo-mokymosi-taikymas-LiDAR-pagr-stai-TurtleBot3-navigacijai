"""
Residual-RL wrapper around a LiDAR-only classical gate detector.

The baseline detector computes an action from LiDAR + odometry memory only.
The PPO action is a small residual correction:

    final_action = detector_action + residual_scale * ppo_action

This is no-cheat because neither the detector nor the policy observation receives
true gate centers, pole coordinates, target points, or gate indices.
"""

from __future__ import annotations

import math
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

try:
    from .gate_env_no_cheat import Gate, GateLidarNoCheatEnv
except ImportError:
    from gate_env_no_cheat import Gate, GateLidarNoCheatEnv
from lidar_residual_detector_v4d import (
    DETECTOR_FEATURE_DIM,
    DetectorOutput,
    SoftTargetMemory,
    detector_action_v4d,
    detector_output_to_features,
)


class GateLidarResidualEnv(GateLidarNoCheatEnv):
    def __init__(
        self,
        render_mode: Optional[str] = None,
        curriculum_level: int = 4,
        lidar_noise: bool = True,
        use_full_360: bool = True,
        residual_linear_scale: float = 0.12,
        residual_angular_scale: float = 0.22,
    ):
        self.detector_memory = SoftTargetMemory()
        self.cached_detector_output: Optional[DetectorOutput] = None
        self.cached_detector_action = np.array([0.0, 0.0], dtype=np.float32)
        self.last_residual_action = np.array([0.0, 0.0], dtype=np.float32)
        self.residual_linear_scale = float(residual_linear_scale)
        self.residual_angular_scale = float(residual_angular_scale)

        super().__init__(
            render_mode=render_mode,
            curriculum_level=curriculum_level,
            lidar_noise=lidar_noise,
            use_full_360=use_full_360,
        )

        base_dim = int(self.obs_beams + self.odom_dim)
        # Detector features + previous final detector action/residual action.
        self.detector_feature_dim = DETECTOR_FEATURE_DIM + 4
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(base_dim + self.detector_feature_dim,),
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        self.detector_memory.reset()
        self.cached_detector_output = None
        self.cached_detector_action = np.array([0.0, 0.0], dtype=np.float32)
        self.last_residual_action = np.array([0.0, 0.0], dtype=np.float32)
        return super().reset(seed=seed, options=options)

    def _compute_and_cache_detector_action(self) -> None:
        out = detector_action_v4d(
            self.last_scan,
            self.detector_memory,
            v=float(getattr(self, "v", 0.0)),
            w=float(getattr(self, "w", 0.0)),
            control_dt=float(getattr(self, "control_dt", 0.05)),
            lidar_min=float(getattr(self, "lidar_min_range", 0.12)),
            lidar_max=float(getattr(self, "lidar_max_range", 3.50)),
            max_linear=float(getattr(self, "max_linear", 0.18)),
            max_angular=float(getattr(self, "max_angular", 1.20)),
        )
        self.cached_detector_output = out
        self.cached_detector_action = np.asarray(out.action, dtype=np.float32).clip(-1.0, 1.0)

    def _get_obs(self):
        base_obs = super()._get_obs().astype(np.float32)
        # Compute the detector action for the next policy step. This intentionally
        # updates only the robot's LiDAR-derived short memory.
        self._compute_and_cache_detector_action()
        assert self.cached_detector_output is not None
        det_features = detector_output_to_features(
            self.cached_detector_output,
            lidar_max=float(getattr(self, "lidar_max_range", 3.50)),
        )
        action_features = np.array(
            [
                np.clip(0.5 + 0.5 * self.cached_detector_action[0], 0.0, 1.0),
                np.clip(0.5 + 0.5 * self.cached_detector_action[1], 0.0, 1.0),
                np.clip(0.5 + 0.5 * self.last_residual_action[0], 0.0, 1.0),
                np.clip(0.5 + 0.5 * self.last_residual_action[1], 0.0, 1.0),
            ],
            dtype=np.float32,
        )
        return np.concatenate([base_obs, det_features, action_features]).astype(np.float32)

    def step(self, action):
        residual = np.asarray(action, dtype=np.float32).clip(-1.0, 1.0)
        self.last_residual_action = residual.copy()

        if self.cached_detector_output is None:
            self._compute_and_cache_detector_action()

        base_action = np.asarray(self.cached_detector_action, dtype=np.float32).copy()
        final_action = base_action.copy()
        final_action[0] += self.residual_linear_scale * float(residual[0])
        final_action[1] += self.residual_angular_scale * float(residual[1])
        final_action = np.clip(final_action, -1.0, 1.0).astype(np.float32)

        obs, reward, terminated, truncated, info = super().step(final_action)

        # Mildly encourage PPO to keep the strong LiDAR controller unless it has
        # a useful correction. This prevents policy drift from destroying baseline behavior.
        reward -= float(0.010 * (residual[0] ** 2 + residual[1] ** 2))

        info["detector_action"] = base_action.tolist()
        info["residual_action"] = residual.tolist()
        info["final_action"] = final_action.tolist()
        if self.cached_detector_output is not None:
            info["detector_source"] = self.cached_detector_output.source
            info["detector_confidence"] = float(self.cached_detector_output.confidence)
            info["detector_target_x"] = float(self.cached_detector_output.target_x)
            info["detector_target_y"] = float(self.cached_detector_output.target_y)

        return obs, float(reward), terminated, truncated, info

    def _build_curriculum_gates(self):
        lvl = int(self.curriculum_level)

        if lvl == 1:
            gate_x_positions = [2.0]
            clear_opening = 0.58
            y_low, y_high = 0.98, 1.02
        elif lvl == 2:
            gate_x_positions = [2.0]
            clear_opening = 0.54
            y_low, y_high = 0.88, 1.12
        elif lvl == 3:
            gate_x_positions = [2.0, 3.0]
            clear_opening = 0.50
            y_low, y_high = 0.84, 1.16
        elif lvl == 40:
            gate_x_positions = [2.0, 3.0, 4.0]
            clear_opening = 0.50
            y_low, y_high = 0.88, 1.12
        elif lvl == 41:
            gate_x_positions = [2.0, 3.0, 4.0]
            clear_opening = 0.48
            y_low, y_high = 0.86, 1.14
        elif lvl == 42:
            gate_x_positions = [2.0, 3.0, 4.0]
            clear_opening = 0.46
            y_low, y_high = 0.84, 1.16
        else:
            gate_x_positions = [2.0, 3.0, 4.0]
            clear_opening = 0.45
            y_low, y_high = 0.80, 1.20

        centers = []
        for idx, _gx in enumerate(gate_x_positions):
            if idx == 0 or lvl < 40:
                centers.append(float(self.np_random.uniform(y_low, y_high)))
            else:
                # Sequential gates should move smoothly enough that the next gate
                # can become visible after passing the previous one.
                jump = float(self.np_random.uniform(-0.32, 0.32))
                centers.append(float(np.clip(centers[-1] + jump, y_low, y_high)))

        self.gates = [
            Gate(x=float(gx), center_y=float(cy), clear_opening=float(clear_opening))
            for gx, cy in zip(gate_x_positions, centers)
        ]

        self.poles = []
        for gate in self.gates:
            offset = gate.clear_opening / 2.0 + self.pole_radius
            self.poles.append((gate.x, gate.center_y - offset))
            self.poles.append((gate.x, gate.center_y + offset))

    def _sample_spawn_pose(self):
        first_gate = self.gates[0]
        lvl = int(self.curriculum_level)

        if lvl == 1:
            self.x = float(self.np_random.uniform(0.35, 1.25))
            self.y = float(self.np_random.uniform(0.78, 1.22))
            self.theta = float(self.np_random.uniform(-0.15, 0.15))
            return

        if lvl == 2:
            lateral_range, close_prob, theta_limit, aim_noise = 0.42, 0.00, 0.28, 0.25
        elif lvl == 3:
            lateral_range, close_prob, theta_limit, aim_noise = 0.52, 0.00, 0.38, 0.32
        elif lvl in (40, 41, 42):
            lateral_range, close_prob, theta_limit, aim_noise = 0.58, 0.08, 0.42, 0.36
        else:
            lateral_range, close_prob, theta_limit, aim_noise = 0.70, 0.20, 0.45, 0.40

        if self.np_random.random() < close_prob:
            self.x = float(self.np_random.uniform(1.35, 1.75))
            offset = float(self.np_random.uniform(-0.58, 0.58))
            self.y = float(np.clip(first_gate.center_y + offset, self.world_y_min + 0.22, self.world_y_max - 0.22))
            desired_theta = math.atan2(first_gate.center_y - self.y, first_gate.x - self.x)
            self.theta = float(self._wrap_angle(desired_theta + self.np_random.uniform(-0.38, 0.38)))
            self.theta = float(np.clip(self.theta, -0.62, 0.62))
            return

        self.x = float(self.np_random.uniform(0.35, 1.35))
        self.y = float(np.clip(first_gate.center_y + self.np_random.uniform(-lateral_range, lateral_range), self.world_y_min + 0.22, self.world_y_max - 0.22))
        desired_theta = math.atan2(first_gate.center_y - self.y, first_gate.x - self.x)
        self.theta = float(self._wrap_angle(desired_theta + self.np_random.uniform(-aim_noise, aim_noise)))
        self.theta = float(np.clip(self.theta, -theta_limit, theta_limit))


class RandomLevelGateLidarResidualEnv(GateLidarResidualEnv):
    def __init__(self, levels, **kwargs):
        self.random_levels = [int(x) for x in levels]
        if not self.random_levels:
            raise ValueError("levels must not be empty")
        super().__init__(curriculum_level=self.random_levels[0], **kwargs)

    def reset(self, *, seed=None, options=None):
        self.curriculum_level = int(np.random.choice(self.random_levels))
        return super().reset(seed=seed, options=options)


class CloseStartGateLidarResidualEnv(GateLidarResidualEnv):
    def __init__(self, **kwargs):
        super().__init__(curriculum_level=4, **kwargs)

    def _sample_spawn_pose(self):
        first_gate = self.gates[0]
        r = float(self.np_random.random())
        if r < 0.75:
            self.x = float(self.np_random.uniform(1.35, 1.78))
            if self.np_random.random() < 0.55:
                offset = float(self.np_random.uniform(-0.42, 0.42))
            else:
                side = -1.0 if self.np_random.random() < 0.5 else 1.0
                offset = side * float(self.np_random.uniform(0.42, 0.72))
            self.y = float(np.clip(first_gate.center_y + offset, self.world_y_min + 0.22, self.world_y_max - 0.22))
            desired_theta = math.atan2(first_gate.center_y - self.y, first_gate.x - self.x)
            self.theta = float(self._wrap_angle(desired_theta + self.np_random.uniform(-0.42, 0.42)))
            self.theta = float(np.clip(self.theta, -0.65, 0.65))
            return

        self.x = float(self.np_random.uniform(0.45, 1.35))
        self.y = float(self.np_random.uniform(0.30, 1.70))
        self.theta = float(self.np_random.uniform(-0.45, 0.45))
