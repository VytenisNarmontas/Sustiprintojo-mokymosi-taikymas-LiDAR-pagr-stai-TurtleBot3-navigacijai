"""
No-cheat TurtleBot3 Burger gate-navigation RL environment.

Policy observation intentionally contains NO gate coordinates, NO target point,
NO gate index, and NO hidden simulator labels.

Allowed observations:
    - 360 LiDAR ranges, ordered like Gazebo TurtleBot3 /scan:
        beam 0 = front, beam 90 = left, beam 180 = back, beam 270 = right
    - odometry relative to episode start
    - robot velocity / previous command

Hidden simulator geometry is used only for reward, collision, and success checks.
That is normal for RL simulation training and is not available to the deployed
controller at inference time.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from matplotlib.patches import Circle, Polygon


@dataclass
class Gate:
    x: float
    center_y: float
    clear_opening: float


class GateLidarNoCheatEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 20}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        curriculum_level: int = 4,
        lidar_noise: bool = True,
        use_full_360: bool = True,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.curriculum_level = int(curriculum_level)
        self.lidar_noise = bool(lidar_noise)
        self.use_full_360 = bool(use_full_360)

        # Arena dimensions. Training y is [0, 2]. Gazebo y is usually [-1, 1].
        self.world_x_min = 0.0
        self.world_x_max = 5.0
        self.world_y_min = 0.0
        self.world_y_max = 2.0

        # Conservative TurtleBot3 Burger footprint.
        # Slightly inflated compared with ideal body to transfer better to Gazebo.
        self.robot_length = 0.160
        self.robot_width = 0.205
        self.half_length = self.robot_length / 2.0
        self.half_width = self.robot_width / 2.0

        # Match Gazebo TurtleBot3 LDS values observed from /scan.
        self.n_beams = 360
        self.obs_beams = 360 if self.use_full_360 else 72
        self.lidar_min_range = 0.12
        self.lidar_max_range = 3.50
        self.lidar_scan_hz = 5.0

        # Dynamics. Keep these calmer than idealized 2D-fast settings.
        self.control_dt = 0.05  # 20 Hz controller loop
        self.max_linear = 0.22
        self.max_angular = 1.35
        self.max_linear_accel = 0.45
        self.max_angular_accel = 2.00
        self.linear_lowpass_alpha = 0.60
        self.angular_lowpass_alpha = 0.35
        self.angular_deadband = 0.025

        self.max_steps = 1800  # 90 s at 20 Hz

        # Gate/pole geometry. clear_opening is empty space between pole surfaces.
        self.pole_radius = 0.050
        self.gate_clear_opening = 0.45
        self.gate_x_positions = [2.0, 3.0, 4.0]
        self.gate_clearance_margin = 0.050
        self.gate_pass_tolerance = 0.020

        # Spawn region.
        self.spawn_x_min = 0.30
        self.spawn_x_max = 1.70
        self.spawn_y_min = 0.30
        self.spawn_y_max = 1.70

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Observation = LiDAR + relative odometry + velocity/previous command.
        # No target, no gate center, no gate index.
        self.odom_dim = 8
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.obs_beams + self.odom_dim,),
            dtype=np.float32,
        )

        self.scan_stride = max(1, int(round((1.0 / self.lidar_scan_hz) / self.control_dt)))

        self.fig = None
        self.ax = None

        self.gates: List[Gate] = []
        self.poles: List[Tuple[float, float]] = []
        self.last_scan: Optional[np.ndarray] = None

        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.step_count = 0
        self.scan_counter = 0
        self.gate_index = 0
        self.gate_start_step = 0
        self.stall_steps = 0
        self.spin_steps = 0
        self.slow_crawl_steps = 0
        self.episode_min_scan = self.lidar_max_range

        self.v = 0.0
        self.w = 0.0
        self.filtered_target_v = 0.0
        self.filtered_target_w = 0.0
        self.prev_cmd_v = 0.0
        self.prev_cmd_w = 0.0

        self._build_curriculum_gates()
        self._sample_spawn_pose()

        self.start_x = self.x
        self.start_y = self.y
        self.start_theta = self.theta

        self._update_scan(force=True)
        return self._get_obs(), self._get_info("reset")

    def _build_curriculum_gates(self):
        """
        No-cheat curriculum.

        1      = centered easy one-gate task
        20-22  = one-gate bridge levels
        2      = full one-gate randomized task
        3      = two-gate task
        40-42  = three-gate easier bridge levels
        420-426 = micro bridge from Level 42 to final Level 4
        4      = full three-gate final task

        Hidden gate geometry is used only for reward/collision/success.
        It is NOT included in policy observation.
        """
        if self.curriculum_level == 1:
            gate_x_positions = [2.0]
            clear_opening = 0.58
            y_low, y_high = 0.98, 1.02

        elif self.curriculum_level == 20:
            gate_x_positions = [2.0]
            clear_opening = 0.58
            y_low, y_high = 0.94, 1.06

        elif self.curriculum_level == 21:
            gate_x_positions = [2.0]
            clear_opening = 0.56
            y_low, y_high = 0.90, 1.10

        elif self.curriculum_level == 22:
            gate_x_positions = [2.0]
            clear_opening = 0.54
            y_low, y_high = 0.88, 1.12

        elif self.curriculum_level == 2:
            gate_x_positions = [2.0]
            clear_opening = 0.54
            y_low, y_high = 0.88, 1.12

        elif self.curriculum_level == 3:
            gate_x_positions = [2.0, 3.0]
            clear_opening = 0.50
            y_low, y_high = 0.84, 1.16

        elif self.curriculum_level == 40:
            gate_x_positions = [2.0, 3.0, 4.0]
            clear_opening = 0.54
            y_low, y_high = 0.92, 1.08

        elif self.curriculum_level == 41:
            gate_x_positions = [2.0, 3.0, 4.0]
            clear_opening = 0.52
            y_low, y_high = 0.88, 1.12

        elif self.curriculum_level == 42:
            gate_x_positions = [2.0, 3.0, 4.0]
            clear_opening = 0.50
            y_low, y_high = 0.84, 1.16

        # Micro bridge after Level 42.
        elif self.curriculum_level == 420:
            gate_x_positions = [2.0, 3.0, 4.0]
            clear_opening = 0.50
            y_low, y_high = 0.835, 1.165

        elif self.curriculum_level == 421:
            gate_x_positions = [2.0, 3.0, 4.0]
            clear_opening = 0.495
            y_low, y_high = 0.83, 1.17

        elif self.curriculum_level == 422:
            gate_x_positions = [2.0, 3.0, 4.0]
            clear_opening = 0.490
            y_low, y_high = 0.825, 1.175

        elif self.curriculum_level == 423:
            gate_x_positions = [2.0, 3.0, 4.0]
            clear_opening = 0.480
            y_low, y_high = 0.82, 1.18

        elif self.curriculum_level == 424:
            gate_x_positions = [2.0, 3.0, 4.0]
            clear_opening = 0.470
            y_low, y_high = 0.815, 1.185

        elif self.curriculum_level == 425:
            gate_x_positions = [2.0, 3.0, 4.0]
            clear_opening = 0.460
            y_low, y_high = 0.81, 1.19

        elif self.curriculum_level == 426:
            gate_x_positions = [2.0, 3.0, 4.0]
            clear_opening = 0.455
            y_low, y_high = 0.805, 1.195

        elif self.curriculum_level == 4:
            gate_x_positions = [2.0, 3.0, 4.0]
            clear_opening = self.gate_clear_opening
            y_low, y_high = 0.80, 1.20

        else:
            raise ValueError(f"Unknown curriculum_level={self.curriculum_level}")

        self.gates = []
        for gx in gate_x_positions:
            center_y = float(self.np_random.uniform(y_low, y_high))
            self.gates.append(Gate(x=float(gx), center_y=center_y, clear_opening=float(clear_opening)))

        self.poles = []
        for gate in self.gates:
            offset = gate.clear_opening / 2.0 + self.pole_radius
            self.poles.append((gate.x, gate.center_y - offset))
            self.poles.append((gate.x, gate.center_y + offset))

    def _sample_spawn_pose(self):
        first_gate = self.gates[0]

        if self.curriculum_level == 1:
            self.x = float(self.np_random.uniform(0.35, 1.25))
            self.y = float(self.np_random.uniform(0.78, 1.22))
            self.theta = float(self.np_random.uniform(-0.15, 0.15))
            return

        if self.curriculum_level == 20:
            lateral_range = 0.18
            side_recovery_prob = 0.00
            theta_limit = 0.18
            aim_noise = 0.18

        elif self.curriculum_level == 21:
            lateral_range = 0.30
            side_recovery_prob = 0.06
            theta_limit = 0.24
            aim_noise = 0.22

        elif self.curriculum_level == 22:
            lateral_range = 0.42
            side_recovery_prob = 0.12
            theta_limit = 0.28
            aim_noise = 0.26

        elif self.curriculum_level == 2:
            lateral_range = 0.52
            side_recovery_prob = 0.18
            theta_limit = 0.28
            aim_noise = 0.30

        elif self.curriculum_level == 3:
            lateral_range = 0.60
            side_recovery_prob = 0.25
            theta_limit = 0.38
            aim_noise = 0.36

        elif self.curriculum_level == 40:
            lateral_range = 0.46
            side_recovery_prob = 0.14
            theta_limit = 0.32
            aim_noise = 0.30

        elif self.curriculum_level == 41:
            lateral_range = 0.52
            side_recovery_prob = 0.18
            theta_limit = 0.35
            aim_noise = 0.33

        elif self.curriculum_level == 42:
            lateral_range = 0.58
            side_recovery_prob = 0.22
            theta_limit = 0.38
            aim_noise = 0.36

        elif self.curriculum_level == 420:
            lateral_range = 0.59
            side_recovery_prob = 0.23
            theta_limit = 0.39
            aim_noise = 0.37

        elif self.curriculum_level == 421:
            lateral_range = 0.60
            side_recovery_prob = 0.24
            theta_limit = 0.40
            aim_noise = 0.38

        elif self.curriculum_level == 422:
            lateral_range = 0.62
            side_recovery_prob = 0.25
            theta_limit = 0.41
            aim_noise = 0.39

        elif self.curriculum_level == 423:
            lateral_range = 0.64
            side_recovery_prob = 0.26
            theta_limit = 0.42
            aim_noise = 0.40

        elif self.curriculum_level == 424:
            lateral_range = 0.66
            side_recovery_prob = 0.27
            theta_limit = 0.43
            aim_noise = 0.405

        elif self.curriculum_level == 425:
            lateral_range = 0.68
            side_recovery_prob = 0.285
            theta_limit = 0.44
            aim_noise = 0.415

        elif self.curriculum_level in (426, 4):
            lateral_range = 0.70
            side_recovery_prob = 0.30
            theta_limit = 0.45
            aim_noise = 0.42

        else:
            raise ValueError(f"Unknown curriculum_level={self.curriculum_level}")

        self.x = float(self.np_random.uniform(0.35, 1.35))

        if self.np_random.random() < side_recovery_prob:
            side = -1.0 if self.np_random.random() < 0.5 else 1.0
            side_offset = float(self.np_random.uniform(0.24, lateral_range))
            self.y = float(np.clip(
                first_gate.center_y + side * side_offset,
                self.world_y_min + 0.22,
                self.world_y_max - 0.22,
            ))
        else:
            self.y = float(np.clip(
                first_gate.center_y + self.np_random.uniform(-lateral_range, lateral_range),
                self.world_y_min + 0.22,
                self.world_y_max - 0.22,
            ))

        desired_theta = math.atan2(first_gate.center_y - self.y, first_gate.x - self.x)
        self.theta = float(self._wrap_angle(desired_theta + self.np_random.uniform(-aim_noise, aim_noise)))
        self.theta = float(np.clip(self.theta, -theta_limit, theta_limit))

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).clip(-1.0, 1.0)

        prev_x = self.x
        prev_y = self.y
        prev_theta = self.theta
        prev_rear_x = self._rear_x_at_pose(prev_x, prev_y, prev_theta)

        # Action mapping uses only the action and LiDAR-derived safety cap.
        forward01 = float((action[0] + 1.0) * 0.5)
        raw_target_v = forward01 * self.max_linear
        raw_target_w = float(action[1] * self.max_angular)

        if abs(raw_target_w) < self.angular_deadband:
            raw_target_w = 0.0

        # LiDAR-based front safety speed cap. This is allowed because it uses
        # the same scan the deployed robot has.
        front_min = self._front_min_range(self.last_scan)
        front_wide_min = self._front_wide_min_range(self.last_scan)

        if front_min < 0.22 or front_wide_min < 0.20:
            raw_target_v = min(raw_target_v, 0.020)
        elif front_min < 0.30 or front_wide_min < 0.28:
            raw_target_v = min(raw_target_v, 0.070)
        elif front_min < 0.42 or front_wide_min < 0.38:
            raw_target_v = min(raw_target_v, 0.135)

        target_v = float(np.clip(raw_target_v, 0.0, self.max_linear))
        target_w = float(np.clip(raw_target_w, -self.max_angular, self.max_angular))

        self.filtered_target_v = (
            self.linear_lowpass_alpha * target_v
            + (1.0 - self.linear_lowpass_alpha) * self.filtered_target_v
        )
        self.filtered_target_w = (
            self.angular_lowpass_alpha * target_w
            + (1.0 - self.angular_lowpass_alpha) * self.filtered_target_w
        )

        self.v = self._move_towards(
            self.v,
            self.filtered_target_v,
            self.max_linear_accel * self.control_dt,
        )
        self.w = self._move_towards(
            self.w,
            self.filtered_target_w,
            self.max_angular_accel * self.control_dt,
        )

        self.theta = self._wrap_angle(self.theta + self.w * self.control_dt)
        self.x += self.v * math.cos(self.theta) * self.control_dt
        self.y += self.v * math.sin(self.theta) * self.control_dt

        self.prev_cmd_v = target_v
        self.prev_cmd_w = target_w

        self.step_count += 1
        self.scan_counter += 1
        self._update_scan(force=False)

        reward, terminated, truncated, event = self._compute_reward_and_done(
            prev_x=prev_x,
            prev_y=prev_y,
            prev_theta=prev_theta,
            prev_rear_x=prev_rear_x,
        )

        return self._get_obs(), float(reward), terminated, truncated, self._get_info(event)

    def _compute_reward_and_done(self, prev_x, prev_y, prev_theta, prev_rear_x):
        x_progress = self.x - prev_x
        rear_x_now = self._rear_x_at_pose(self.x, self.y, self.theta)

        reward = 0.0
        terminated = False
        truncated = False
        event = "running"

        # General movement reward. This does not reveal gate positions.
        reward -= 0.018
        reward += 20.0 * max(0.0, x_progress)
        reward -= 24.0 * max(0.0, -x_progress)
        reward += 0.004 * float(np.clip(self.v / max(1e-6, self.max_linear), 0.0, 1.0))

        # Hidden training-only shaping near the current gate.
        # This is NOT in observation and does NOT exist at deployment.
        if self.gate_index < len(self.gates):
            gate = self.gates[self.gate_index]
            dist_to_gate = gate.x - self.x

            if -0.15 <= dist_to_gate <= 1.25:
                prev_center_error = abs(prev_y - gate.center_y)
                new_center_error = abs(self.y - gate.center_y)
                center_improvement = prev_center_error - new_center_error

                prev_heading_abs = abs(self._wrap_angle(prev_theta))
                new_heading_abs = abs(self._wrap_angle(self.theta))
                heading_improvement = prev_heading_abs - new_heading_abs

                # Potential-style shaping: reward improvements AND penalize regressions.
                # Do not use max(0, improvement), because that can be farmed by
                # repeatedly moving away from and back toward the gate center.
                reward += 12.0 * center_improvement
                reward -= 0.095 * new_center_error
                reward += 1.6 * heading_improvement
                reward -= 0.035 * new_heading_abs

                # If close to the gate but clearly not aligned, discourage rushing.
                if 0.0 <= dist_to_gate <= 0.45:
                    if new_center_error > 0.11 or new_heading_abs > 0.32:
                        reward -= 0.120 * float(np.clip(self.v / max(1e-6, self.max_linear), 0.0, 1.0))

                    # Extra small penalty for crossing the gate zone at a large angle.
                    # This prevents 2D policies that succeed only by clipping through
                    # a tight opening and then failing in Gazebo.
                    if new_heading_abs > 0.55:
                        reward -= 0.030


        # LiDAR proximity shaping, observation-consistent.
        min_scan = float(np.min(self.last_scan))
        self.episode_min_scan = min(self.episode_min_scan, min_scan)

        if min_scan < 0.34:
            reward -= float((0.34 - min_scan) * 3.0)
        if min_scan < 0.24:
            reward -= float((0.24 - min_scan) * 12.0)

        # danger_speed_penalty_v7
        # Do not reward fast driving when any LiDAR beam is dangerously close.
        # This reduces corner/pole clipping without adding hidden gate info.
        speed_fraction = float(np.clip(self.v / max(1e-6, self.max_linear), 0.0, 1.0))
        if min_scan < 0.30:
            reward -= 0.090 * speed_fraction
        if min_scan < 0.25:
            reward -= 0.180 * speed_fraction
        if min_scan < 0.20:
            reward -= 0.260 * speed_fraction

        # Anti-wobble / anti-spin costs.
        reward -= 0.006 * abs(self.w / max(1e-6, self.max_angular))
        reward -= 0.004 * abs((self.w - self.prev_cmd_w) / max(1e-6, self.max_angular))

        # Anti-crawl: moving extremely slowly for too long is not useful.
        if 0.006 < self.v < 0.055 and abs(self.w) < 0.45:
            self.slow_crawl_steps += 1
            if self.slow_crawl_steps > 80:
                reward -= 0.030
        else:
            self.slow_crawl_steps = 0

        if abs(x_progress) < 0.0007 and abs(self.v) < 0.012 and abs(self.w) < 0.18:
            self.stall_steps += 1
            reward -= 0.020
        else:
            self.stall_steps = 0

        if abs(self.w) > 0.70 and abs(x_progress) < 0.0012:
            self.spin_steps += 1
            reward -= 0.015
        else:
            self.spin_steps = 0

        if self._collision():
            reward -= 420.0
            terminated = True
            event = "collision"

        if not terminated and self.gate_index < len(self.gates):
            gate = self.gates[self.gate_index]
            gate_line = gate.x + self.gate_clearance_margin

            if prev_rear_x < gate_line <= rear_x_now:
                if self._crossed_current_gate_cleanly(gate):
                    self.gate_index += 1
                    self.stall_steps = 0
                    self.spin_steps = 0

                    # fast_gate_bonus
                    steps_for_gate = max(1, self.step_count - self.gate_start_step)
                    fast_gate_bonus = max(0.0, 320.0 - float(steps_for_gate)) * 0.20
                    self.gate_start_step = self.step_count

                    if self.gate_index == len(self.gates):
                        reward += 420.0 + fast_gate_bonus
                        terminated = True
                        event = "success"
                    else:
                        reward += 140.0 + fast_gate_bonus
                        event = f"passed_gate_{self.gate_index}"
                else:
                    reward -= 380.0
                    terminated = True
                    event = "missed_gate"

        if not terminated and self.stall_steps >= 160:
            reward -= 100.0
            terminated = True
            event = "stalled"

        if not terminated and self.spin_steps >= 220:
            reward -= 100.0
            terminated = True
            event = "spinning"

        if self.step_count >= self.max_steps:
            truncated = True
            if event == "running":
                reward -= 180.0
                event = "timeout"

        return reward, terminated, truncated, event

    def _get_obs(self):
        scan = self._scan_to_obs_beams(self.last_scan)
        lidar_norm = np.clip(
            (scan - self.lidar_min_range) / (self.lidar_max_range - self.lidar_min_range),
            0.0,
            1.0,
        ).astype(np.float32)

        dx_rel = self.x - self.start_x
        dy_rel = self.y - self.start_y
        theta_rel = self._wrap_angle(self.theta - self.start_theta)

        dx_norm = np.clip(dx_rel / self.world_x_max, 0.0, 1.0)
        dy_norm = np.clip((dy_rel + self.world_y_max) / (2.0 * self.world_y_max), 0.0, 1.0)
        sin_th = 0.5 + 0.5 * math.sin(theta_rel)
        cos_th = 0.5 + 0.5 * math.cos(theta_rel)
        v_norm = np.clip(self.v / max(1e-6, self.max_linear), 0.0, 1.0)
        w_norm = np.clip(0.5 + 0.5 * self.w / max(1e-6, self.max_angular), 0.0, 1.0)
        prev_v_norm = np.clip(self.prev_cmd_v / max(1e-6, self.max_linear), 0.0, 1.0)
        prev_w_norm = np.clip(0.5 + 0.5 * self.prev_cmd_w / max(1e-6, self.max_angular), 0.0, 1.0)

        odom = np.array(
            [dx_norm, dy_norm, sin_th, cos_th, v_norm, w_norm, prev_v_norm, prev_w_norm],
            dtype=np.float32,
        )
        return np.concatenate([lidar_norm, odom]).astype(np.float32)

    def _get_info(self, event):
        return {
            "event": event,
            "passed_gates": int(self.gate_index),
            "min_lidar": float(self.episode_min_scan),
            "x": float(self.x),
            "y": float(self.y),
            "theta": float(self.theta),
            "v": float(self.v),
            "w": float(self.w),
            "stall_steps": int(self.stall_steps),
            "spin_steps": int(self.spin_steps),
            "slow_crawl_steps": int(getattr(self, "slow_crawl_steps", 0)),
            # These are for logging/debug only, not observation.
            "gate_centers": [float(g.center_y) for g in self.gates],
        }

    def _scan_to_obs_beams(self, scan):
        if self.use_full_360:
            return scan.astype(np.float32)
        sector_size = self.n_beams // self.obs_beams
        return scan.reshape(self.obs_beams, sector_size).min(axis=1)

    def _update_scan(self, force: bool):
        if force or self.scan_counter >= self.scan_stride or self.last_scan is None:
            true_scan = self._lidar_scan_true()
            if self.lidar_noise:
                self.last_scan = self._apply_lidar_noise(true_scan)
            else:
                self.last_scan = true_scan
            self.scan_counter = 0

    def _lidar_scan_true(self):
        # Gazebo-style ordering: 0 rad is front, pi/2 left, pi back, 3pi/2 right.
        rel_angles = np.linspace(0.0, 2.0 * math.pi, self.n_beams, endpoint=False)
        scan = np.full(self.n_beams, self.lidar_max_range, dtype=np.float32)

        ox = self.x
        oy = self.y

        for i, rel_angle in enumerate(rel_angles):
            ang = self.theta + rel_angle
            dx = math.cos(ang)
            dy = math.sin(ang)

            dist = self._ray_to_walls(ox, oy, dx, dy)
            for px, py in self.poles:
                circle_dist = self._ray_to_circle(ox, oy, dx, dy, px, py, self.pole_radius)
                if circle_dist is not None:
                    dist = min(dist, circle_dist)

            scan[i] = float(np.clip(dist, self.lidar_min_range, self.lidar_max_range))

        return scan

    def _apply_lidar_noise(self, scan):
        noisy = np.array(scan, dtype=np.float32, copy=True)

        for i, d in enumerate(noisy):
            if d <= 0.30:
                sigma = 0.004
            elif d <= 1.5:
                sigma = max(0.004, 0.010 * d)
            else:
                sigma = 0.020
            noisy[i] = float(self.np_random.normal(float(d), sigma))
            noisy[i] = float(np.clip(noisy[i], self.lidar_min_range, self.lidar_max_range))

        # Occasional Gazebo-like max-range dropouts.
        if self.np_random.random() < 0.10:
            n_drop = int(self.np_random.integers(1, 5))
            idx = self.np_random.integers(0, self.n_beams, size=n_drop)
            noisy[idx] = self.lidar_max_range

        return noisy

    def _front_min_range(self, scan):
        # Beam 0 is front. Use +/- 18 deg sector around front.
        sector = np.concatenate([scan[:18], scan[-18:]])
        return float(np.min(sector))

    def _front_wide_min_range(self, scan):
        # Wider front sector for safer approach near gate poles.
        # Still LiDAR-only, no hidden gate information.
        sector = np.concatenate([scan[:36], scan[-36:]])
        return float(np.min(sector))

    def _collision(self):
        corners = self._robot_corners(self.x, self.y, self.theta)

        for cx, cy in corners:
            if cx <= self.world_x_min or cx >= self.world_x_max:
                return True
            if cy <= self.world_y_min or cy >= self.world_y_max:
                return True

        for px, py in self.poles:
            if self._rect_circle_collision(px, py, self.pole_radius):
                return True

        return False

    def _crossed_current_gate_cleanly(self, gate: Gate):
        projected_half_y = (
            self.half_width * abs(math.cos(self.theta))
            + self.half_length * abs(math.sin(self.theta))
        )

        allowable = gate.clear_opening / 2.0 - projected_half_y - 0.010 + self.gate_pass_tolerance
        if allowable <= 0.0:
            return False
        return abs(self.y - gate.center_y) <= allowable

    def _robot_corners(self, x, y, theta):
        c = math.cos(theta)
        s = math.sin(theta)

        local = np.array(
            [
                [self.half_length, self.half_width],
                [self.half_length, -self.half_width],
                [-self.half_length, -self.half_width],
                [-self.half_length, self.half_width],
            ],
            dtype=np.float32,
        )

        rot = np.array([[c, -s], [s, c]], dtype=np.float32)
        world = local @ rot.T
        world[:, 0] += x
        world[:, 1] += y
        return world

    def _rear_x_at_pose(self, x, y, theta):
        return float(np.min(self._robot_corners(x, y, theta)[:, 0]))

    def _rect_circle_collision(self, circle_x, circle_y, circle_r):
        dx = circle_x - self.x
        dy = circle_y - self.y

        c = math.cos(-self.theta)
        s = math.sin(-self.theta)

        local_x = dx * c - dy * s
        local_y = dx * s + dy * c

        closest_x = float(np.clip(local_x, -self.half_length, self.half_length))
        closest_y = float(np.clip(local_y, -self.half_width, self.half_width))

        err_x = local_x - closest_x
        err_y = local_y - closest_y
        return err_x * err_x + err_y * err_y <= circle_r * circle_r

    def _ray_to_walls(self, ox, oy, dx, dy):
        candidates = []

        if abs(dx) > 1e-12:
            t = (self.world_x_min - ox) / dx
            y = oy + t * dy
            if t > 0.0 and self.world_y_min <= y <= self.world_y_max:
                candidates.append(t)

            t = (self.world_x_max - ox) / dx
            y = oy + t * dy
            if t > 0.0 and self.world_y_min <= y <= self.world_y_max:
                candidates.append(t)

        if abs(dy) > 1e-12:
            t = (self.world_y_min - oy) / dy
            x = ox + t * dx
            if t > 0.0 and self.world_x_min <= x <= self.world_x_max:
                candidates.append(t)

            t = (self.world_y_max - oy) / dy
            x = ox + t * dx
            if t > 0.0 and self.world_x_min <= x <= self.world_x_max:
                candidates.append(t)

        return min(candidates) if candidates else self.lidar_max_range

    def _ray_to_circle(self, ox, oy, dx, dy, cx, cy, radius):
        ocx = ox - cx
        ocy = oy - cy
        b = 2.0 * (dx * ocx + dy * ocy)
        c = ocx * ocx + ocy * ocy - radius * radius
        disc = b * b - 4.0 * c
        if disc < 0.0:
            return None
        sqrt_disc = math.sqrt(disc)
        t1 = (-b - sqrt_disc) / 2.0
        t2 = (-b + sqrt_disc) / 2.0
        positive = [t for t in (t1, t2) if t > 0.0]
        return min(positive) if positive else None

    def render(self):
        if self.render_mode != "human":
            return

        if self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(14, 6))

        self.ax.clear()
        self.ax.plot(
            [self.world_x_min, self.world_x_max, self.world_x_max, self.world_x_min, self.world_x_min],
            [self.world_y_min, self.world_y_min, self.world_y_max, self.world_y_max, self.world_y_min],
            linewidth=2,
        )

        for px, py in self.poles:
            self.ax.add_patch(Circle((px, py), self.pole_radius, fill=True))

        for gate in self.gates:
            self.ax.plot([gate.x, gate.x], [gate.center_y - 0.03, gate.center_y + 0.03], linewidth=2)
            self.ax.plot(
                [gate.x + self.gate_clearance_margin, gate.x + self.gate_clearance_margin],
                [0.0, 2.0],
                linewidth=0.5,
            )

        # Plot every 6th LiDAR beam.
        rel_angles = np.linspace(0.0, 2.0 * math.pi, self.n_beams, endpoint=False)
        for i, (rel_angle, dist) in enumerate(zip(rel_angles, self.last_scan)):
            if i % 6 != 0:
                continue
            ang = self.theta + rel_angle
            x2 = self.x + dist * math.cos(ang)
            y2 = self.y + dist * math.sin(ang)
            self.ax.plot([self.x, x2], [self.y, y2], linewidth=0.30)

        corners = self._robot_corners(self.x, self.y, self.theta)
        self.ax.add_patch(Polygon(corners, closed=True, fill=False, linewidth=2))
        hx = self.x + 0.18 * math.cos(self.theta)
        hy = self.y + 0.18 * math.sin(self.theta)
        self.ax.plot([self.x, hx], [self.y, hy], linewidth=2)

        self.ax.set_title(
            f"NO-CHEAT LiDAR+odom | step={self.step_count} "
            f"gate={self.gate_index}/{len(self.gates)} x={self.x:.2f} y={self.y:.2f} "
            f"th={self.theta:.2f} v={self.v:.2f} w={self.w:.2f}"
        )
        self.ax.set_xlim(self.world_x_min - 0.05, self.world_x_max + 0.05)
        self.ax.set_ylim(self.world_y_min - 0.05, self.world_y_max + 0.05)
        self.ax.set_aspect("equal")
        plt.pause(0.001)

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None

    @staticmethod
    def _move_towards(current, target, max_delta):
        if target > current + max_delta:
            return current + max_delta
        if target < current - max_delta:
            return current - max_delta
        return target

    @staticmethod
    def _wrap_angle(angle):
        return (angle + math.pi) % (2.0 * math.pi) - math.pi


# Backwards-compatible alias for scripts that import GateLidarEnv.
GateLidarEnv = GateLidarNoCheatEnv
