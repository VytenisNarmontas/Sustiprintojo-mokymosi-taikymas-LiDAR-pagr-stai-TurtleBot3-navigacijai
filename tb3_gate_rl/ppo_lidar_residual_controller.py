#!/usr/bin/env python3
"""ROS2 residual PPO controller using only /scan and /odom.

Final command = LiDAR detector baseline command + PPO residual correction.
Does not subscribe to /gate_experiment/gates.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from .gate_env_lidar_residual import GateLidarResidualEnv
from .lidar_residual_detector_v4d import (
    SoftTargetMemory,
    detector_action_v4d,
    detector_output_to_features,
)


def yaw_from_quat(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class ResidualGateController(Node):
    def __init__(self):
        super().__init__("ppo_lidar_residual_controller")

        self.declare_parameter("model_dir", os.path.expanduser("~/rl_gate_train_lidar_feature/models_lidar_residual_final"))
        self.declare_parameter("control_hz", 20.0)
        self.declare_parameter("max_linear", 0.18)
        self.declare_parameter("max_angular", 1.20)
        self.declare_parameter("residual_linear_scale", 0.12)
        self.declare_parameter("residual_angular_scale", 0.22)
        self.declare_parameter("lidar_min_range", 0.12)
        self.declare_parameter("lidar_max_range", 3.50)

        self.model_dir = Path(str(self.get_parameter("model_dir").value)).expanduser()
        self.control_hz = float(self.get_parameter("control_hz").value)
        self.max_linear = float(self.get_parameter("max_linear").value)
        self.max_angular = float(self.get_parameter("max_angular").value)
        self.residual_linear_scale = float(self.get_parameter("residual_linear_scale").value)
        self.residual_angular_scale = float(self.get_parameter("residual_angular_scale").value)
        self.lidar_min_range = float(self.get_parameter("lidar_min_range").value)
        self.lidar_max_range = float(self.get_parameter("lidar_max_range").value)

        # Dummy env only for VecNormalize observation normalization shape.
        dummy = DummyVecEnv([lambda: GateLidarResidualEnv(lidar_noise=False)])
        self.vecnorm = VecNormalize.load(str(self.model_dir / "vecnormalize.pkl"), dummy)
        self.vecnorm.training = False
        self.vecnorm.norm_reward = False
        self.model = PPO.load(str(self.model_dir / "ppo_gate.zip"), device="cpu")

        self.scan: Optional[np.ndarray] = None
        self.start_x: Optional[float] = None
        self.start_y: Optional[float] = None
        self.start_yaw: Optional[float] = None
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.v = 0.0
        self.w = 0.0
        self.prev_cmd_v = 0.0
        self.prev_cmd_w = 0.0
        self.memory = SoftTargetMemory()

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.create_subscription(LaserScan, "/scan", self.on_scan, 10)
        self.create_subscription(Odometry, "/odom", self.on_odom, 10)
        self.create_timer(1.0 / self.control_hz, self.on_timer)

        self.get_logger().info(f"Loaded residual PPO model from {self.model_dir}")
        self.get_logger().info("No-cheat inputs: /scan + /odom only. No /gate_experiment/gates subscription.")

    def on_scan(self, msg: LaserScan):
        ranges = np.asarray(msg.ranges, dtype=np.float32)
        ranges[~np.isfinite(ranges)] = self.lidar_max_range
        ranges = np.clip(ranges, self.lidar_min_range, self.lidar_max_range)

        if ranges.size != 360:
            old_x = np.linspace(0.0, 1.0, ranges.size, endpoint=False)
            new_x = np.linspace(0.0, 1.0, 360, endpoint=False)
            ranges = np.interp(new_x, old_x, ranges).astype(np.float32)

        # TurtleBot3 Gazebo /scan is normally 0=front for this project setup.
        self.scan = ranges.astype(np.float32)

    def on_odom(self, msg: Odometry):
        p = msg.pose.pose.position
        yaw = yaw_from_quat(msg.pose.pose.orientation)
        if self.start_x is None:
            self.start_x = float(p.x)
            self.start_y = float(p.y)
            self.start_yaw = float(yaw)
            self.memory.reset()
        self.x = float(p.x)
        self.y = float(p.y)
        self.yaw = float(yaw)
        self.v = float(msg.twist.twist.linear.x)
        self.w = float(msg.twist.twist.angular.z)

    def make_obs(self, det_out) -> np.ndarray:
        assert self.scan is not None
        sx = self.start_x if self.start_x is not None else self.x
        sy = self.start_y if self.start_y is not None else self.y
        syaw = self.start_yaw if self.start_yaw is not None else self.yaw

        lidar_norm = np.clip((self.scan - self.lidar_min_range) / (self.lidar_max_range - self.lidar_min_range), 0.0, 1.0).astype(np.float32)
        dx_rel = self.x - sx
        dy_rel = self.y - sy
        theta_rel = (self.yaw - syaw + math.pi) % (2 * math.pi) - math.pi

        odom = np.array([
            np.clip(dx_rel / 5.0, 0.0, 1.0),
            np.clip((dy_rel + 2.0) / 4.0, 0.0, 1.0),
            0.5 + 0.5 * math.sin(theta_rel),
            0.5 + 0.5 * math.cos(theta_rel),
            np.clip(self.v / max(self.max_linear, 1e-6), 0.0, 1.0),
            np.clip(0.5 + 0.5 * self.w / max(self.max_angular, 1e-6), 0.0, 1.0),
            np.clip(self.prev_cmd_v / max(self.max_linear, 1e-6), 0.0, 1.0),
            np.clip(0.5 + 0.5 * self.prev_cmd_w / max(self.max_angular, 1e-6), 0.0, 1.0),
        ], dtype=np.float32)

        det_features = detector_output_to_features(det_out, lidar_max=self.lidar_max_range)
        action_features = np.array([
            np.clip(0.5 + 0.5 * det_out.action[0], 0.0, 1.0),
            np.clip(0.5 + 0.5 * det_out.action[1], 0.0, 1.0),
            0.5,
            0.5,
        ], dtype=np.float32)
        return np.concatenate([lidar_norm, odom, det_features, action_features]).astype(np.float32)

    def action_to_cmd(self, action: np.ndarray) -> Twist:
        forward01 = float((float(action[0]) + 1.0) * 0.5)
        v = float(np.clip(forward01 * self.max_linear, 0.0, self.max_linear))
        w = float(np.clip(float(action[1]) * self.max_angular, -self.max_angular, self.max_angular))
        msg = Twist()
        msg.linear.x = v
        msg.angular.z = w
        self.prev_cmd_v = v
        self.prev_cmd_w = w
        return msg

    def on_timer(self):
        if self.scan is None or self.start_x is None:
            return

        det_out = detector_action_v4d(
            self.scan,
            self.memory,
            v=self.v,
            w=self.w,
            control_dt=1.0 / max(self.control_hz, 1e-6),
            lidar_min=self.lidar_min_range,
            lidar_max=self.lidar_max_range,
            max_linear=self.max_linear,
            max_angular=self.max_angular,
        )
        obs = self.make_obs(det_out)
        norm_obs = self.vecnorm.normalize_obs(obs.reshape(1, -1))
        residual, _ = self.model.predict(norm_obs, deterministic=True)
        residual = np.asarray(residual[0], dtype=np.float32).clip(-1.0, 1.0)

        final_action = np.asarray(det_out.action, dtype=np.float32).copy()
        final_action[0] += self.residual_linear_scale * float(residual[0])
        final_action[1] += self.residual_angular_scale * float(residual[1])
        final_action = np.clip(final_action, -1.0, 1.0)

        self.cmd_pub.publish(self.action_to_cmd(final_action))


def main(args=None):
    rclpy.init(args=args)
    node = ResidualGateController()
    try:
        rclpy.spin(node)
    finally:
        stop = Twist()
        node.cmd_pub.publish(stop)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
