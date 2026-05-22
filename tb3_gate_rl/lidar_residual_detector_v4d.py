"""
LiDAR-only active gate detector + baseline controller for residual RL.

Allowed inputs:
  - 360 range scan with beam 0 = front, 90 = left, 180 = rear, 270 = right
  - robot odometry-derived velocity v/w and control_dt for short target memory

Not used:
  - true gate coordinates
  - true pole coordinates
  - gate index
  - /gate_experiment/gates
  - simulator hidden labels

This is intentionally close in spirit to the previous successful V4d detector,
but adapted to the no-cheat Gazebo-style scan convention used by
GateLidarNoCheatEnv.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


DETECTOR_FEATURE_DIM = 18


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def beam_bearing(idx: int, n: int = 360) -> float:
    # 0 is front, +pi/2 is left, -pi/2 is right.
    return wrap_angle(2.0 * math.pi * float(idx) / float(n))


def front_order_indices(n: int = 360, half_angle_deg: float = 115.0) -> List[int]:
    half = int(round(half_angle_deg))
    return list(range(n - half, n)) + list(range(0, half + 1))


def front_min_from_scan(scan: Sequence[float], half_deg: int = 18, default: float = 3.5) -> float:
    arr = np.asarray(scan, dtype=np.float32)
    if arr.size == 0:
        return float(default)
    half = int(max(1, half_deg))
    sector = np.concatenate([arr[:half], arr[-half:]]) if arr.size >= 2 * half else arr
    return float(np.min(sector))


def safe_min(arr: Sequence[float], default: float = 3.5) -> float:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return float(default)
    return float(np.min(arr))


def safe_percentile(arr: Sequence[float], q: float, default: float = 0.0) -> float:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return float(default)
    return float(np.percentile(arr, q))


@dataclass
class Cluster:
    indices: List[int]
    x: float
    y: float
    range: float
    bearing: float
    width: float
    angular_span: float
    min_range: float


@dataclass
class GateCandidate:
    x: float
    y: float
    sep: float
    score: float
    left: Cluster
    right: Cluster
    confidence: float
    source: str = "pair"


@dataclass
class DetectorOutput:
    target_x: float
    target_y: float
    action: np.ndarray
    confidence: float
    source: str
    front_min: float
    clusters: int
    gate_sep: float
    memory_age: int
    memory_missed: int


@dataclass
class SoftTargetMemory:
    """Short LiDAR-derived target memory, dead-reckoned using only v/w/dt."""

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
        dx = float(v) * float(dt)
        dtheta = float(w) * float(dt)

        # Previous robot frame -> current robot frame.
        px = x - dx
        py = y
        ca = math.cos(dtheta)
        sa = math.sin(dtheta)
        new_x = ca * px + sa * py
        new_y = -sa * px + ca * py

        self.target = (float(new_x), float(new_y))
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
            self.target = (float(ox), float(oy))
            self.age = 0
            self.missed = 0
            return self.target

        tx, ty = self.target
        old_angle = math.atan2(ty, tx)
        new_angle = math.atan2(oy, ox)

        if abs(wrap_angle(new_angle - old_angle)) > 0.55:
            self.target = (float(ox), float(oy))
            self.age = 0
            self.missed = 0
            return self.target

        sx = self.old_weight * tx + self.new_weight * ox
        sy = self.old_weight * ty + self.new_weight * oy
        self.target = (float(sx), float(sy))
        self.age = 0
        self.missed = 0
        return self.target

    def mark_no_gate_seen(self, front_min: float) -> Optional[Tuple[float, float]]:
        if self.target is None:
            return None
        self.missed += 1
        self.release_if_bad(front_min)
        return self.target


def scan_to_360(scan: Sequence[float], lidar_min: float = 0.12, lidar_max: float = 3.5) -> np.ndarray:
    arr = np.asarray(scan, dtype=np.float32).copy()
    if arr.size != 360:
        old_x = np.linspace(0.0, 1.0, arr.size, endpoint=False)
        new_x = np.linspace(0.0, 1.0, 360, endpoint=False)
        arr = np.interp(new_x, old_x, arr).astype(np.float32)
    arr[~np.isfinite(arr)] = lidar_max
    return np.clip(arr, lidar_min, lidar_max).astype(np.float32)


def extract_clusters(
    scan: Sequence[float],
    lidar_min: float = 0.12,
    lidar_max: float = 3.5,
    max_x: float = 2.80,
    max_abs_y: float = 1.45,
    jump_dist: float = 0.14,
    half_angle_deg: float = 115.0,
) -> List[Cluster]:
    arr = scan_to_360(scan, lidar_min, lidar_max)
    n = int(arr.size)
    ids = front_order_indices(n, half_angle_deg)

    clusters: List[List[int]] = []
    cur: List[int] = []
    last_r: Optional[float] = None

    for idx in ids:
        r = float(arr[idx])
        b = beam_bearing(idx, n)
        x = r * math.cos(b)
        y = r * math.sin(b)
        ok = (
            np.isfinite(r)
            and lidar_min <= r < 0.97 * lidar_max
            and 0.03 < x < max_x
            and abs(y) < max_abs_y
        )

        if ok:
            if cur and last_r is not None and abs(r - last_r) > jump_dist:
                clusters.append(cur)
                cur = []
            cur.append(idx)
            last_r = r
        else:
            if cur:
                clusters.append(cur)
                cur = []
            last_r = None

    if cur:
        clusters.append(cur)

    out: List[Cluster] = []
    for cids in clusters:
        if len(cids) < 1 or len(cids) > 40:
            continue
        pts = []
        bearings = []
        ranges = []
        for idx in cids:
            r = float(arr[idx])
            b = beam_bearing(idx, n)
            bearings.append(b)
            ranges.append(r)
            pts.append((r * math.cos(b), r * math.sin(b)))
        pts_arr = np.asarray(pts, dtype=np.float32)
        xs = pts_arr[:, 0]
        ys = pts_arr[:, 1]
        x = float(0.5 * np.median(xs) + 0.5 * np.mean(xs))
        y = float(0.5 * np.median(ys) + 0.5 * np.mean(ys))
        rng = float(math.hypot(x, y))
        bearing = float(math.atan2(y, x))
        width = float(max(np.max(xs) - np.min(xs), np.max(ys) - np.min(ys))) if len(cids) > 1 else 0.03
        span = float(max(bearings) - min(bearings)) if len(bearings) > 1 else math.radians(1.0)
        min_r = float(np.min(ranges))

        # Poles are compact. Long wall-like clusters are rejected.
        if x <= 0.03 or rng < 0.10 or width > 0.32:
            continue
        out.append(Cluster(list(cids), x, y, rng, bearing, width, abs(span), min_r))

    return out


def find_gate_candidates(clusters: List[Cluster], scan: Sequence[float]) -> List[GateCandidate]:
    arr = scan_to_360(scan)
    out: List[GateCandidate] = []

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
            mid_angle = math.atan2(mid_y, mid_x)

            if xdiff > 0.45:
                continue
            if not (0.34 <= sep <= 0.95):
                continue
            if mid_x < 0.12:
                continue
            if abs(mid_angle) > 1.15:
                continue

            # Check that beams between poles are relatively open.
            lo = min(a.bearing, b.bearing)
            hi = max(a.bearing, b.bearing)
            sector = []
            for idx in range(360):
                bb = beam_bearing(idx)
                if lo <= bb <= hi:
                    sector.append(float(arr[idx]))
            gap_clear = float(np.percentile(sector, 60)) if sector else 0.0

            sep_error = abs(sep - 0.58)
            score = mid_x + 0.55 * abs(mid_y) + 0.80 * xdiff + 0.50 * sep_error - 0.10 * min(gap_clear, 3.5)
            conf = float(np.clip(1.0 - score / 3.2, 0.25, 1.0))

            left, right = (a, b) if a.y > b.y else (b, a)
            out.append(GateCandidate(mid_x, mid_y, sep, score, left, right, conf, "strict_pair"))

    out.sort(key=lambda g: g.score)
    return out


def relaxed_gate_candidate(clusters: List[Cluster], scan: Sequence[float]) -> Optional[GateCandidate]:
    if len(clusters) < 2:
        return None

    best: Optional[GateCandidate] = None
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
            if xdiff > 0.55:
                continue
            if not (0.32 <= sep <= 1.02):
                continue
            if mid_x < 0.12 or abs(math.atan2(mid_y, mid_x)) > 1.20:
                continue

            sep_error = abs(sep - 0.58)
            score = mid_x + 0.65 * abs(mid_y) + 0.95 * xdiff + 0.60 * sep_error
            if score < best_score:
                left, right = (a, b) if a.y > b.y else (b, a)
                best_score = score
                best = GateCandidate(mid_x, mid_y, sep, score, left, right, 0.55, "relaxed_pair")
    return best


def one_pole_target(clusters: List[Cluster], scan: Sequence[float]) -> Optional[Tuple[float, float]]:
    if not clusters:
        return None

    arr = scan_to_360(scan)
    front_clusters = [
        c for c in clusters
        if 0.15 < c.x < 2.40 and abs(c.y) < 1.20 and c.width <= 0.24
    ]
    if not front_clusters:
        return None

    pole = min(front_clusters, key=lambda c: c.x)

    left_scan = arr[20:100]
    right_scan = arr[-100:-20]
    left_clear = safe_percentile(left_scan, 70)
    right_clear = safe_percentile(right_scan, 70)

    if pole.y > 0.08:
        target_y = pole.y - 0.34
    elif pole.y < -0.08:
        target_y = pole.y + 0.34
    else:
        target_y = 0.34 if left_clear >= right_clear else -0.34

    target_x = pole.x + 0.20
    return float(target_x), float(target_y)


def forward_or_gap_fallback(scan: Sequence[float]) -> Tuple[float, float]:
    arr = scan_to_360(scan)
    front = np.concatenate([arr[:16], arr[-16:]])
    left = arr[20:105]
    right = arr[-105:-20]

    front_min = safe_min(front)
    left_clear = safe_percentile(left, 75)
    right_clear = safe_percentile(right, 75)

    if front_min > 0.65:
        return 1.0, 0.0
    if left_clear >= right_clear:
        return 0.80, 0.65
    return 0.80, -0.65


def v_to_action(v: float, max_linear: float) -> float:
    max_linear = max(float(max_linear), 1e-6)
    return float(np.clip(2.0 * (float(v) / max_linear) - 1.0, -1.0, 1.0))


def action_from_target_angle(
    target_angle: float,
    target_dist: float,
    front_min: float,
    max_linear: float = 0.18,
    max_angular: float = 1.20,
) -> np.ndarray:
    angle_abs = abs(float(target_angle))

    if front_min < 0.24:
        v = 0.020
    elif angle_abs < 0.18:
        v = 0.175
    elif angle_abs < 0.42:
        v = 0.145
    elif angle_abs < 0.72:
        v = 0.090
    else:
        v = 0.045

    if target_dist < 0.35:
        v = min(v, 0.070)

    # Normalize angular command for the environment action.
    w = float(np.clip(1.85 * float(target_angle), -max_angular, max_angular))
    a0 = v_to_action(v, max_linear)
    a1 = float(np.clip(w / max(max_angular, 1e-6), -1.0, 1.0))
    return np.array([a0, a1], dtype=np.float32)


def collision_guard(
    scan: Sequence[float],
    action: Sequence[float],
    target_angle: float,
    front_min: float,
    max_linear: float = 0.18,
) -> np.ndarray:
    arr = scan_to_360(scan)
    action = np.asarray(action, dtype=np.float32).copy()

    left_front = safe_min(arr[8:50])
    right_front = safe_min(arr[-50:-8])
    left_mid = safe_min(arr[50:95])
    right_mid = safe_min(arr[-95:-50])
    left_clear = safe_percentile(arr[20:115], 75)
    right_clear = safe_percentile(arr[-115:-20], 75)

    turn_to_clear = 0.75 if left_clear >= right_clear else -0.75

    def cap_v(max_v: float) -> None:
        action[0] = min(float(action[0]), v_to_action(max_v, max_linear))

    if front_min < 0.23:
        cap_v(0.025)
        action[1] = turn_to_clear
        return np.clip(action, -1.0, 1.0)

    if front_min < 0.32:
        cap_v(0.060)
        if abs(float(action[1])) < 0.45:
            action[1] = 0.60 if turn_to_clear > 0 else -0.60

    if left_front < 0.27 and target_angle > 0.08:
        cap_v(0.080)
        action[1] = min(float(action[1]), -0.35)

    if right_front < 0.27 and target_angle < -0.08:
        cap_v(0.080)
        action[1] = max(float(action[1]), 0.35)

    if left_front < 0.30 and right_front < 0.30:
        cap_v(0.055)
        action[1] = turn_to_clear

    if left_mid < 0.23:
        cap_v(0.095)
        action[1] = min(float(action[1]), -0.20)

    if right_mid < 0.23:
        cap_v(0.095)
        action[1] = max(float(action[1]), 0.20)

    return np.clip(action, -1.0, 1.0)


def action_to_target(
    scan: Sequence[float],
    tx: float,
    ty: float,
    front_min: float,
    max_linear: float = 0.18,
    max_angular: float = 1.20,
) -> np.ndarray:
    target_angle = math.atan2(float(ty), float(tx))
    target_dist = math.hypot(float(tx), float(ty))
    action = action_from_target_angle(target_angle, target_dist, front_min, max_linear, max_angular)
    return collision_guard(scan, action, target_angle, front_min, max_linear)


def detector_action_v4d(
    scan: Sequence[float],
    memory: SoftTargetMemory,
    v: float = 0.0,
    w: float = 0.0,
    control_dt: float = 0.05,
    lidar_min: float = 0.12,
    lidar_max: float = 3.5,
    max_linear: float = 0.18,
    max_angular: float = 1.20,
    debug: bool = False,
) -> DetectorOutput:
    arr = scan_to_360(scan, lidar_min, lidar_max)
    front_min = front_min_from_scan(arr, default=lidar_max)

    memory.predict_from_odometry(v=float(v), w=float(w), dt=float(control_dt))
    memory.release_if_bad(front_min)

    clusters = extract_clusters(
        arr,
        lidar_min=lidar_min,
        lidar_max=lidar_max,
        max_x=2.80,
        max_abs_y=1.45,
        jump_dist=0.14,
    )

    gates = find_gate_candidates(clusters, arr)
    if gates:
        gate = gates[0]
        observed = (gate.x + 0.22, gate.y)
        tx, ty = memory.observe_gate_target(observed, front_min)
        action = action_to_target(arr, tx, ty, front_min, max_linear, max_angular)
        if debug:
            print(f"STRICT target=({tx:.2f},{ty:.2f}) gate=({gate.x:.2f},{gate.y:.2f}) sep={gate.sep:.2f} clusters={len(clusters)}")
        return DetectorOutput(tx, ty, action, gate.confidence, "strict_pair", front_min, len(clusters), gate.sep, memory.age, memory.missed)

    relaxed = relaxed_gate_candidate(clusters, arr)
    if relaxed is not None:
        observed = (relaxed.x + 0.18, relaxed.y)
        tx, ty = memory.observe_gate_target(observed, front_min)
        action = action_to_target(arr, tx, ty, front_min, max_linear, max_angular)
        if debug:
            print(f"RELAXED target=({tx:.2f},{ty:.2f}) gate=({relaxed.x:.2f},{relaxed.y:.2f}) sep={relaxed.sep:.2f} clusters={len(clusters)}")
        return DetectorOutput(tx, ty, action, relaxed.confidence, "relaxed_pair", front_min, len(clusters), relaxed.sep, memory.age, memory.missed)

    remembered = memory.mark_no_gate_seen(front_min)
    if remembered is not None:
        tx, ty = remembered
        angle = math.atan2(ty, tx)
        safe_bridge = (
            front_min > 0.40
            and 0.20 < tx < 2.10
            and abs(angle) < 0.65
            and memory.missed <= 5
            and memory.age <= 22
        )
        if safe_bridge:
            action = action_to_target(arr, tx, ty, front_min, max_linear, max_angular)
            if debug:
                print(f"MEMORY target=({tx:.2f},{ty:.2f}) angle={angle:.2f} clusters={len(clusters)}")
            return DetectorOutput(tx, ty, action, 0.42, "memory", front_min, len(clusters), 0.0, memory.age, memory.missed)
        memory.reset()

    one = one_pole_target(clusters, arr)
    if one is not None:
        tx, ty = one
        action = action_to_target(arr, tx, ty, front_min, max_linear, max_angular)
        if debug:
            print(f"ONE_POLE target=({tx:.2f},{ty:.2f}) clusters={len(clusters)}")
        return DetectorOutput(tx, ty, action, 0.35, "one_pole", front_min, len(clusters), 0.0, memory.age, memory.missed)

    tx, ty = forward_or_gap_fallback(arr)
    action = action_to_target(arr, tx, ty, front_min, max_linear, max_angular)
    if debug:
        print(f"FALLBACK target=({tx:.2f},{ty:.2f}) clusters={len(clusters)}")
    return DetectorOutput(tx, ty, action, 0.15, "fallback", front_min, len(clusters), 0.0, memory.age, memory.missed)


def detector_output_to_features(out: DetectorOutput, lidar_max: float = 3.5) -> np.ndarray:
    angle = math.atan2(out.target_y, out.target_x)
    dist = math.hypot(out.target_x, out.target_y)
    source_id = {
        "strict_pair": 1.0,
        "relaxed_pair": 0.8,
        "memory": 0.6,
        "one_pole": 0.4,
        "fallback": 0.2,
    }.get(out.source, 0.0)
    return np.array(
        [
            np.clip(out.confidence, 0.0, 1.0),
            np.clip(0.5 + 0.5 * np.clip(angle, -math.pi / 2, math.pi / 2) / (math.pi / 2), 0.0, 1.0),
            np.clip(dist / lidar_max, 0.0, 1.0),
            np.clip(out.target_x / lidar_max, 0.0, 1.0),
            np.clip(0.5 + out.target_y / 2.0, 0.0, 1.0),
            np.clip(out.front_min / lidar_max, 0.0, 1.0),
            np.clip(out.clusters / 8.0, 0.0, 1.0),
            np.clip(out.gate_sep / 1.2, 0.0, 1.0),
            np.clip(out.action[0] * 0.5 + 0.5, 0.0, 1.0),
            np.clip(out.action[1] * 0.5 + 0.5, 0.0, 1.0),
            np.clip(out.memory_age / 30.0, 0.0, 1.0),
            np.clip(out.memory_missed / 8.0, 0.0, 1.0),
            source_id,
            1.0 if out.source in ("strict_pair", "relaxed_pair") else 0.0,
            1.0 if out.source == "memory" else 0.0,
            1.0 if out.source == "one_pole" else 0.0,
            1.0 if out.source == "fallback" else 0.0,
            1.0,
        ],
        dtype=np.float32,
    )


def stateless_detector_features(
    scan: Sequence[float],
    lidar_min: float = 0.12,
    lidar_max: float = 3.5,
    max_linear: float = 0.18,
    max_angular: float = 1.20,
) -> np.ndarray:
    # For observation-only diagnostics; uses a temporary memory so it does not mutate controller memory.
    mem = SoftTargetMemory()
    out = detector_action_v4d(
        scan,
        mem,
        v=0.0,
        w=0.0,
        control_dt=0.05,
        lidar_min=lidar_min,
        lidar_max=lidar_max,
        max_linear=max_linear,
        max_angular=max_angular,
    )
    return detector_output_to_features(out, lidar_max=lidar_max)
