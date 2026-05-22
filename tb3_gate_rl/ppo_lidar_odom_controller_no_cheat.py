"""
ROS2 no-cheat PPO controller.

Inputs used by policy:
    /scan  -> 360 LiDAR ranges
    /odom  -> odometry relative to controller start/reset
    previous command

Not used:
    /gate_experiment/gates
    hidden gate centers
    target point
    gate index
    ground-truth pose

This file is intended to be copied into:
    ~/turtlebot3_ws/src/tb3_gate_rl/tb3_gate_rl/ppo_lidar_odom_controller_no_cheat.py
"""

import math
from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from stable_baselines3 import PPO
from std_msgs.msg import Empty


class NoCheatPpoController(Node):
    def __init__(self):
        super().__init__("ppo_lidar_odom_controller_no_cheat")

        self.declare_parameter("model_dir", str(Path.home() / "rl_gate_train_no_cheat/models_level4_no_cheat"))
        self.declare_parameter("controller_max_linear", 0.22)
        self.declare_parameter("controller_max_angular", 1.35)
        self.declare_parameter("controller_max_linear_accel", 0.45)
        self.declare_parameter("controller_max_angular_accel", 2.00)
        self.declare_parameter("linear_lowpass_alpha", 0.60)
        self.declare_parameter("angular_lowpass_alpha", 0.35)
        self.declare_parameter("angular_deadband", 0.025)

        self.model_dir = Path(self.get_parameter("model_dir").value).expanduser()
        self.max_linear = float(self.get_parameter("controller_max_linear").value)
        self.max_angular = float(self.get_parameter("controller_max_angular").value)
        self.max_linear_accel = float(self.get_parameter("controller_max_linear_accel").value)
        self.max_angular_accel = float(self.get_parameter("controller_max_angular_accel").value)
        self.linear_lowpass_alpha = float(self.get_parameter("linear_lowpass_alpha").value)
        self.angular_lowpass_alpha = float(self.get_parameter("angular_lowpass_alpha").value)
        self.angular_deadband = float(self.get_parameter("angular_deadband").value)

        self.control_dt = 0.05
        self.n_beams = 360
        self.lidar_min_range = 0.12
        self.lidar_max_range = 3.50
        self.world_x_max = 5.0
        self.world_y_max = 2.0

        self.model = PPO.load(str(self.model_dir / "ppo_gate.zip"), device="cpu")
        self.norm_stats = self.load_norm_stats(self.model_dir / "vecnormalize_stats.npz")

        self.last_scan: Optional[np.ndarray] = None
        self.last_odom: Optional[Odometry] = None
        self.start_x: Optional[float] = None
        self.start_y: Optional[float] = None
        self.start_yaw: Optional[float] = None

        self.v = 0.0
        self.w = 0.0
        self.filtered_target_v = 0.0
        self.filtered_target_w = 0.0
        self.prev_cmd_v = 0.0
        self.prev_cmd_w = 0.0

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.scan_sub = self.create_subscription(LaserScan, "/scan", self.scan_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, "/odom", self.odom_callback, 10)
        self.reset_sub = self.create_subscription(Empty, "/gate_experiment/reset", self.reset_callback, 10)
        self.timer = self.create_timer(self.control_dt, self.control_loop)

        self.get_logger().info(f"Loaded no-cheat PPO model from {self.model_dir}")
        self.get_logger().info("Policy inputs: /scan + /odom + previous cmd only. No gate topic is subscribed.")

    def load_norm_stats(self, path: Path):
        data = np.load(path)
        return {
            "mean": data["obs_mean"].astype(np.float32),
            "var": data["obs_var"].astype(np.float32),
            "epsilon": float(data["epsilon"]),
            "clip_obs": float(data["clip_obs"]),
        }

    def scan_callback(self, msg: LaserScan):
        ranges = np.asarray(msg.ranges, dtype=np.float32)
        ranges[~np.isfinite(ranges)] = self.lidar_max_range
        ranges = np.clip(ranges, self.lidar_min_range, self.lidar_max_range)

        if ranges.size != self.n_beams:
            old_x = np.linspace(0.0, 1.0, ranges.size, endpoint=False)
            new_x = np.linspace(0.0, 1.0, self.n_beams, endpoint=False)
            ranges = np.interp(new_x, old_x, ranges).astype(np.float32)

        self.last_scan = ranges

    def odom_callback(self, msg: Odometry):
        self.last_odom = msg
        x, y, yaw = self.read_odom_pose(msg)
        if self.start_x is None:
            self.start_x = x
            self.start_y = y
            self.start_yaw = yaw

    def reset_callback(self, _msg: Empty):
        self.start_x = None
        self.start_y = None
        self.start_yaw = None
        self.v = 0.0
        self.w = 0.0
        self.filtered_target_v = 0.0
        self.filtered_target_w = 0.0
        self.prev_cmd_v = 0.0
        self.prev_cmd_w = 0.0
        self.publish_cmd(0.0, 0.0)

    def read_odom_pose(self, msg: Odometry):
        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        return x, y, yaw

    def make_obs(self):
        x, y, yaw = self.read_odom_pose(self.last_odom)

        if self.start_x is None:
            self.start_x = x
            self.start_y = y
            self.start_yaw = yaw

        dx_rel = x - self.start_x
        dy_rel = y - self.start_y
        yaw_rel = self.wrap_angle(yaw - self.start_yaw)

        lidar_norm = np.clip(
            (self.last_scan - self.lidar_min_range) / (self.lidar_max_range - self.lidar_min_range),
            0.0,
            1.0,
        ).astype(np.float32)

        dx_norm = np.clip(dx_rel / self.world_x_max, 0.0, 1.0)
        dy_norm = np.clip((dy_rel + self.world_y_max) / (2.0 * self.world_y_max), 0.0, 1.0)
        sin_th = 0.5 + 0.5 * math.sin(yaw_rel)
        cos_th = 0.5 + 0.5 * math.cos(yaw_rel)
        v_norm = np.clip(self.v / max(1e-6, self.max_linear), 0.0, 1.0)
        w_norm = np.clip(0.5 + 0.5 * self.w / max(1e-6, self.max_angular), 0.0, 1.0)
        prev_v_norm = np.clip(self.prev_cmd_v / max(1e-6, self.max_linear), 0.0, 1.0)
        prev_w_norm = np.clip(0.5 + 0.5 * self.prev_cmd_w / max(1e-6, self.max_angular), 0.0, 1.0)

        odom = np.array(
            [dx_norm, dy_norm, sin_th, cos_th, v_norm, w_norm, prev_v_norm, prev_w_norm],
            dtype=np.float32,
        )
        obs = np.concatenate([lidar_norm, odom]).astype(np.float32)
        return obs

    def normalize_obs(self, obs):
        s = self.norm_stats
        obs = (obs - s["mean"]) / np.sqrt(s["var"] + s["epsilon"])
        obs = np.clip(obs, -s["clip_obs"], s["clip_obs"])
        return obs.astype(np.float32)

    def control_loop(self):
        if self.last_scan is None or self.last_odom is None:
            self.publish_cmd(0.0, 0.0)
            return

        obs = self.make_obs()
        norm_obs = self.normalize_obs(obs)
        action, _ = self.model.predict(norm_obs, deterministic=True)
        action = np.asarray(action, dtype=np.float32).clip(-1.0, 1.0)

        forward01 = float((action[0] + 1.0) * 0.5)
        raw_target_v = forward01 * self.max_linear
        raw_target_w = float(action[1] * self.max_angular)

        if abs(raw_target_w) < self.angular_deadband:
            raw_target_w = 0.0

        front_min = self.front_min_range(self.last_scan)
        if front_min < 0.22:
            raw_target_v = min(raw_target_v, 0.020)
        elif front_min < 0.30:
            raw_target_v = min(raw_target_v, 0.075)
        elif front_min < 0.42:
            raw_target_v = min(raw_target_v, 0.145)

        target_v = float(np.clip(raw_target_v, 0.0, self.max_linear))
        target_w = float(np.clip(raw_target_w, -self.max_angular, self.max_angular))

        self.filtered_target_v = self.linear_lowpass_alpha * target_v + (1.0 - self.linear_lowpass_alpha) * self.filtered_target_v
        self.filtered_target_w = self.angular_lowpass_alpha * target_w + (1.0 - self.angular_lowpass_alpha) * self.filtered_target_w

        self.v = self.move_towards(self.v, self.filtered_target_v, self.max_linear_accel * self.control_dt)
        self.w = self.move_towards(self.w, self.filtered_target_w, self.max_angular_accel * self.control_dt)

        self.prev_cmd_v = target_v
        self.prev_cmd_w = target_w
        self.publish_cmd(self.v, self.w)

    def publish_cmd(self, v, w):
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        self.cmd_pub.publish(msg)

    @staticmethod
    def front_min_range(scan):
        sector = np.concatenate([scan[:18], scan[-18:]])
        return float(np.min(sector))

    @staticmethod
    def move_towards(current, target, max_delta):
        if target > current + max_delta:
            return current + max_delta
        if target < current - max_delta:
            return current - max_delta
        return target

    @staticmethod
    def wrap_angle(angle):
        return (angle + math.pi) % (2.0 * math.pi) - math.pi


def main():
    rclpy.init()
    node = NoCheatPpoController()
    try:
        rclpy.spin(node)
    finally:
        node.publish_cmd(0.0, 0.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
