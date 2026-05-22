import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Empty


@dataclass
class Cluster:
    x: float
    y: float
    width: float
    n: int


@dataclass
class GateCandidate:
    x: float
    y: float
    sep: float
    score: float
    left: Cluster
    right: Cluster


@dataclass
class SoftTargetMemory:
    target: Optional[Tuple[float, float]] = None
    age: int = 0
    missed: int = 0

    max_age: int = 26
    max_missed: int = 6
    old_weight: float = 0.60
    new_weight: float = 0.40

    def reset(self) -> None:
        self.target = None
        self.age = 0
        self.missed = 0

    def predict_from_odometry(self, v: float, w: float, dt: float) -> None:
        if self.target is None:
            return

        x, y = self.target

        dx = v * dt
        dtheta = w * dt

        px = x - dx
        py = y

        ca = math.cos(dtheta)
        sa = math.sin(dtheta)

        new_x = ca * px + sa * py
        new_y = -sa * px + ca * py

        self.target = (new_x, new_y)
        self.age += 1

    def unsafe_or_expired(self, front_min: float) -> bool:
        if self.target is None:
            return True

        tx, ty = self.target
        angle = math.atan2(ty, tx)

        if front_min < 0.30:
            return True
        if tx < 0.15:
            return True
        if abs(angle) > 0.95:
            return True
        if self.age > self.max_age:
            return True
        if self.missed > self.max_missed:
            return True

        return False

    def release_if_bad(self, front_min: float) -> None:
        if self.target is not None and self.unsafe_or_expired(front_min):
            self.reset()

    def observe_gate_target(self, observed: Tuple[float, float], front_min: float) -> Tuple[float, float]:
        ox, oy = observed

        if self.target is None or self.unsafe_or_expired(front_min):
            self.target = (ox, oy)
            self.age = 0
            self.missed = 0
            return self.target

        tx, ty = self.target

        old_angle = math.atan2(ty, tx)
        new_angle = math.atan2(oy, ox)

        if abs(new_angle - old_angle) > 0.55:
            self.target = (ox, oy)
            self.age = 0
            self.missed = 0
            return self.target

        sx = self.old_weight * tx + self.new_weight * ox
        sy = self.old_weight * ty + self.new_weight * oy

        self.target = (sx, sy)
        self.age = 0
        self.missed = 0
        return self.target

    def mark_no_gate_seen(self, front_min: float) -> Optional[Tuple[float, float]]:
        if self.target is None:
            return None

        self.missed += 1
        self.release_if_bad(front_min)
        return self.target


def safe_min(arr: np.ndarray, default: float = 8.0) -> float:
    if arr is None or len(arr) == 0:
        return default
    return float(np.min(arr))


def safe_percentile(arr: np.ndarray, q: float, default: float = 0.0) -> float:
    if arr is None or len(arr) == 0:
        return default
    return float(np.percentile(arr, q))


def front_min_from_scan(scan: np.ndarray) -> float:
    n = len(scan)
    c = n // 2
    front = scan[max(0, c - 12):min(n, c + 13)]
    return float(np.min(front))


def scan_msg_to_centered_scan(msg: LaserScan, n_beams: int = 360) -> np.ndarray:
    """
    Converts Gazebo LaserScan angle layout to the training-style layout:

      index n//2 = front / angle 0
      indexes > center = left
      indexes < center = right

    Works for Gazebo scans using 0..2pi or -pi..pi.
    """
    ranges = np.asarray(msg.ranges, dtype=np.float32)

    lidar_min = float(msg.range_min)
    lidar_max = float(msg.range_max)

    ranges = np.nan_to_num(ranges, nan=lidar_max, posinf=lidar_max, neginf=lidar_max)
    ranges = np.clip(ranges, lidar_min, lidar_max)

    raw_angles = float(msg.angle_min) + np.arange(len(ranges), dtype=np.float32) * float(msg.angle_increment)

    # Normalize angles to [-pi, pi).
    angles = np.arctan2(np.sin(raw_angles), np.cos(raw_angles))

    order = np.argsort(angles)
    angles_sorted = angles[order]
    ranges_sorted = ranges[order]

    target_angles = np.linspace(-math.pi, math.pi, n_beams, endpoint=False, dtype=np.float32)

    centered = np.interp(target_angles, angles_sorted, ranges_sorted).astype(np.float32)

    return centered


def extract_clusters(
    scan: np.ndarray,
    lidar_min: float,
    lidar_max: float,
    max_x: float = 2.80,
    max_abs_y: float = 1.45,
    jump_dist: float = 0.14,
) -> List[Cluster]:
    n = len(scan)
    angles = np.linspace(-math.pi, math.pi, n, endpoint=False, dtype=np.float32)

    pts = []

    for r, a in zip(scan, angles):
        rr = float(r)

        if not np.isfinite(rr):
            continue
        if rr <= lidar_min + 0.005:
            continue
        if rr >= lidar_max - 0.02:
            continue

        x = rr * math.cos(float(a))
        y = rr * math.sin(float(a))

        if x <= 0.03:
            continue
        if x > max_x:
            continue
        if abs(y) > max_abs_y:
            continue

        pts.append((x, y))

    if not pts:
        return []

    clusters_raw = []
    current = [pts[0]]

    for p in pts[1:]:
        prev = current[-1]
        d = math.hypot(p[0] - prev[0], p[1] - prev[1])

        if d <= jump_dist:
            current.append(p)
        else:
            if len(current) >= 2:
                clusters_raw.append(current)
            current = [p]

    if len(current) >= 2:
        clusters_raw.append(current)

    clusters = []

    for group in clusters_raw:
        xs = np.array([p[0] for p in group], dtype=np.float32)
        ys = np.array([p[1] for p in group], dtype=np.float32)

        width = float(max(np.max(xs) - np.min(xs), np.max(ys) - np.min(ys)))
        npts = len(group)

        # Pole-like compact obstacle filter.
        if npts < 2:
            continue
        if width > 0.26:
            continue

        cx = float(np.mean(xs))
        cy = float(np.mean(ys))

        clusters.append(Cluster(x=cx, y=cy, width=width, n=npts))

    clusters.sort(key=lambda c: c.x)

    return clusters


def find_gate_candidates(clusters: List[Cluster]) -> List[GateCandidate]:
    candidates = []

    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            a = clusters[i]
            b = clusters[j]

            if a.x <= 0.05 or b.x <= 0.05:
                continue

            xdiff = abs(a.x - b.x)
            sep = abs(a.y - b.y)
            mid_x = 0.5 * (a.x + b.x)
            mid_y = 0.5 * (a.y + b.y)

            if xdiff > 0.35:
                continue
            if not (0.42 <= sep <= 0.78):
                continue
            if mid_x < 0.15:
                continue

            sep_error = abs(sep - 0.58)
            score = mid_x + 0.65 * abs(mid_y) + 1.10 * xdiff + 0.60 * sep_error

            left, right = (a, b) if a.y > b.y else (b, a)

            candidates.append(
                GateCandidate(
                    x=mid_x,
                    y=mid_y,
                    sep=sep,
                    score=score,
                    left=left,
                    right=right,
                )
            )

    candidates.sort(key=lambda g: g.score)

    return candidates


def relaxed_gate_candidate(clusters: List[Cluster]) -> Optional[GateCandidate]:
    if len(clusters) < 2:
        return None

    best = None
    best_score = 999.0

    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            a = clusters[i]
            b = clusters[j]

            if a.x <= 0.05 or b.x <= 0.05:
                continue

            xdiff = abs(a.x - b.x)
            sep = abs(a.y - b.y)
            mid_x = 0.5 * (a.x + b.x)
            mid_y = 0.5 * (a.y + b.y)

            if xdiff > 0.45:
                continue
            if not (0.35 <= sep <= 0.90):
                continue
            if mid_x < 0.15:
                continue

            sep_error = abs(sep - 0.58)
            score = mid_x + 0.55 * abs(mid_y) + 0.80 * xdiff + 0.50 * sep_error

            if score < best_score:
                left, right = (a, b) if a.y > b.y else (b, a)
                best_score = score
                best = GateCandidate(
                    x=mid_x,
                    y=mid_y,
                    sep=sep,
                    score=score,
                    left=left,
                    right=right,
                )

    return best


def one_pole_target(clusters: List[Cluster], scan: np.ndarray) -> Optional[Tuple[float, float]]:
    if not clusters:
        return None

    front_clusters = [
        c for c in clusters
        if 0.15 < c.x < 2.40 and abs(c.y) < 1.20 and c.width <= 0.22
    ]

    if not front_clusters:
        return None

    pole = min(front_clusters, key=lambda c: c.x)

    n = len(scan)
    center = n // 2

    left_scan = scan[center + 20:center + 95]
    right_scan = scan[center - 95:center - 20]

    left_clear = safe_percentile(left_scan, 70)
    right_clear = safe_percentile(right_scan, 70)

    if pole.y > 0.08:
        target_y = pole.y - 0.34
    elif pole.y < -0.08:
        target_y = pole.y + 0.34
    else:
        target_y = 0.34 if left_clear >= right_clear else -0.34

    target_x = pole.x + 0.20

    return target_x, target_y


def forward_or_gap_fallback(scan: np.ndarray) -> Tuple[float, float]:
    n = len(scan)
    c = n // 2

    front = scan[max(0, c - 16):min(n, c + 17)]
    left = scan[c + 20:c + 100]
    right = scan[c - 100:c - 20]

    front_min = safe_min(front)
    left_clear = safe_percentile(left, 75)
    right_clear = safe_percentile(right, 75)

    if front_min > 0.65:
        return 1.0, 0.0

    if left_clear >= right_clear:
        return 0.80, 0.65
    else:
        return 0.80, -0.65


class LidarGateDetectorV4DController(Node):
    def __init__(self):
        super().__init__("lidar_gate_detector_v4d_controller")

        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")

        self.declare_parameter("controller_max_linear", 0.22)
        self.declare_parameter("controller_max_angular", 1.35)
        self.declare_parameter("timer_hz", 10.0)

        self.declare_parameter("n_beams", 360)
        self.declare_parameter("debug", False)

        self.declare_parameter("front_stop_distance", 0.23)
        self.declare_parameter("front_slow_distance", 0.32)

        self.scan_topic = self.get_parameter("scan_topic").value
        self.odom_topic = self.get_parameter("odom_topic").value
        self.cmd_vel_topic = self.get_parameter("cmd_vel_topic").value

        self.max_linear = float(self.get_parameter("controller_max_linear").value)
        self.max_angular = float(self.get_parameter("controller_max_angular").value)
        self.timer_hz = float(self.get_parameter("timer_hz").value)

        self.n_beams = int(self.get_parameter("n_beams").value)
        self.debug = bool(self.get_parameter("debug").value)

        self.front_stop_distance = float(self.get_parameter("front_stop_distance").value)
        self.front_slow_distance = float(self.get_parameter("front_slow_distance").value)

        self.scan: Optional[np.ndarray] = None
        self.lidar_min = 0.12
        self.lidar_max = 3.50

        self.v = 0.0
        self.w = 0.0

        self.memory = SoftTargetMemory()

        self.last_time = self.get_clock().now()

        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)

        self.scan_sub = self.create_subscription(
            LaserScan,
            self.scan_topic,
            self.scan_callback,
            10,
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_callback,
            10,
        )

        # If the manager publishes reset as std_msgs/Empty, memory resets.
        # If the topic is absent, this does nothing harmful.
        self.reset_sub = self.create_subscription(
            Empty,
            "/gate_experiment/reset",
            self.reset_callback,
            10,
        )

        self.timer = self.create_timer(1.0 / self.timer_hz, self.control_step)

        self.get_logger().info("Started LiDAR gate detector V4d controller")
        self.get_logger().info("Subscribed: /scan, /odom")
        self.get_logger().info("Publishing: /cmd_vel")
        self.get_logger().info("Does NOT use /gate_experiment/gates")

    def scan_callback(self, msg: LaserScan) -> None:
        self.lidar_min = float(msg.range_min)
        self.lidar_max = float(msg.range_max)
        self.scan = scan_msg_to_centered_scan(msg, n_beams=self.n_beams)

    def odom_callback(self, msg: Odometry) -> None:
        self.v = float(msg.twist.twist.linear.x)
        self.w = float(msg.twist.twist.angular.z)

    def reset_callback(self, msg: Empty) -> None:
        self.memory.reset()
        self.publish_stop()
        if self.debug:
            self.get_logger().info("Reset received; memory cleared.")

    def publish_stop(self) -> None:
        cmd = Twist()
        self.cmd_pub.publish(cmd)

    def target_to_cmd(self, tx: float, ty: float, front_min: float) -> Twist:
        target_angle = math.atan2(ty, tx)
        target_dist = math.hypot(tx, ty)

        # Base steering.
        angular = 1.75 * target_angle
        angular = float(np.clip(angular, -self.max_angular, self.max_angular))

        # Forward speed reduced while turning.
        turn_ratio = min(1.0, abs(angular) / max(1e-6, self.max_angular))
        linear = self.max_linear * max(0.25, 1.0 - 0.65 * turn_ratio)

        # Slow down when target is very near.
        if target_dist < 0.45:
            linear *= 0.65

        # Front obstacle slow-down.
        if front_min < self.front_slow_distance:
            linear = min(linear, 0.06)

        if front_min < self.front_stop_distance:
            linear = 0.0

        cmd = Twist()
        cmd.linear.x = float(np.clip(linear, 0.0, self.max_linear))
        cmd.angular.z = float(np.clip(angular, -self.max_angular, self.max_angular))

        # Ablation: return the raw gate-target command without the close-obstacle
        # collision guard. This file is only for measuring how much the safety
        # layer contributes to the classical detector controller.
        return cmd

    def collision_guard(self, cmd: Twist, target_angle: float, front_min: float) -> Twist:
        if self.scan is None:
            return cmd

        scan = self.scan
        n = len(scan)
        c = n // 2

        left_front = safe_min(scan[c + 8:c + 50])
        right_front = safe_min(scan[c - 50:c - 8])

        left_mid = safe_min(scan[c + 50:c + 95])
        right_mid = safe_min(scan[c - 95:c - 50])

        left_clear = safe_percentile(scan[c + 20:c + 115], 75)
        right_clear = safe_percentile(scan[c - 115:c - 20], 75)

        turn_to_clear = 0.75 if left_clear >= right_clear else -0.75

        lin = float(cmd.linear.x)
        ang = float(cmd.angular.z)

        if front_min < 0.23:
            lin = min(lin, 0.00)
            ang = turn_to_clear * self.max_angular
        elif front_min < 0.32:
            lin = min(lin, 0.07)
            if abs(ang) < 0.45:
                ang = 0.60 * self.max_angular if turn_to_clear > 0 else -0.60 * self.max_angular

        if left_front < 0.27 and target_angle > 0.08:
            lin = min(lin, 0.08)
            ang = min(ang, -0.35 * self.max_angular)

        if right_front < 0.27 and target_angle < -0.08:
            lin = min(lin, 0.08)
            ang = max(ang, 0.35 * self.max_angular)

        if left_front < 0.30 and right_front < 0.30:
            lin = min(lin, 0.06)
            ang = turn_to_clear * self.max_angular

        if left_mid < 0.23:
            lin = min(lin, 0.10)
            ang = min(ang, -0.20 * self.max_angular)

        if right_mid < 0.23:
            lin = min(lin, 0.10)
            ang = max(ang, 0.20 * self.max_angular)

        cmd.linear.x = float(np.clip(lin, 0.0, self.max_linear))
        cmd.angular.z = float(np.clip(ang, -self.max_angular, self.max_angular))

        return cmd

    def control_step(self) -> None:
        if self.scan is None:
            self.publish_stop()
            return

        now = self.get_clock().now()
        dt = max(1e-3, (now - self.last_time).nanoseconds * 1e-9)
        self.last_time = now

        scan = self.scan
        front_min = front_min_from_scan(scan)

        self.memory.predict_from_odometry(self.v, self.w, dt)
        self.memory.release_if_bad(front_min)

        clusters = extract_clusters(
            scan,
            lidar_min=self.lidar_min,
            lidar_max=self.lidar_max,
            max_x=2.80,
            max_abs_y=1.45,
            jump_dist=0.14,
        )

        gates = find_gate_candidates(clusters)

        if gates:
            gate = gates[0]
            tx, ty = self.memory.observe_gate_target((gate.x + 0.22, gate.y), front_min)

            if self.debug:
                self.get_logger().info(
                    f"STRICT target=({tx:.2f},{ty:.2f}) "
                    f"gate=({gate.x:.2f},{gate.y:.2f}) sep={gate.sep:.2f} "
                    f"front={front_min:.2f} clusters={len(clusters)}"
                )

            self.cmd_pub.publish(self.target_to_cmd(tx, ty, front_min))
            return

        relaxed = relaxed_gate_candidate(clusters)

        if relaxed is not None:
            tx, ty = self.memory.observe_gate_target((relaxed.x + 0.18, relaxed.y), front_min)

            if self.debug:
                self.get_logger().info(
                    f"RELAXED target=({tx:.2f},{ty:.2f}) "
                    f"gate=({relaxed.x:.2f},{relaxed.y:.2f}) sep={relaxed.sep:.2f} "
                    f"front={front_min:.2f} clusters={len(clusters)}"
                )

            self.cmd_pub.publish(self.target_to_cmd(tx, ty, front_min))
            return

        remembered = self.memory.mark_no_gate_seen(front_min)

        if remembered is not None:
            tx, ty = remembered
            angle = math.atan2(ty, tx)

            safe_bridge = (
                front_min > 0.40
                and 0.20 < tx < 2.10
                and abs(angle) < 0.65
                and self.memory.missed <= 5
                and self.memory.age <= 22
            )

            if safe_bridge:
                if self.debug:
                    self.get_logger().info(
                        f"MEMORY target=({tx:.2f},{ty:.2f}) "
                        f"front={front_min:.2f} missed={self.memory.missed}"
                    )

                self.cmd_pub.publish(self.target_to_cmd(tx, ty, front_min))
                return

            self.memory.reset()

        one_pole = one_pole_target(clusters, scan)

        if one_pole is not None:
            tx, ty = one_pole

            if self.debug:
                self.get_logger().info(
                    f"ONE_POLE target=({tx:.2f},{ty:.2f}) "
                    f"front={front_min:.2f} clusters={len(clusters)}"
                )

            self.cmd_pub.publish(self.target_to_cmd(tx, ty, front_min))
            return

        tx, ty = forward_or_gap_fallback(scan)

        if self.debug:
            self.get_logger().info(
                f"FALLBACK target=({tx:.2f},{ty:.2f}) "
                f"front={front_min:.2f} clusters={len(clusters)}"
            )

        self.cmd_pub.publish(self.target_to_cmd(tx, ty, front_min))


def main(args=None):
    rclpy.init(args=args)

    node = LidarGateDetectorV4DController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
