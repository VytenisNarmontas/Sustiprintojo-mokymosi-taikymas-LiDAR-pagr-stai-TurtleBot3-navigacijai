import csv
import math
import random
import time
from pathlib import Path

import numpy as np
import rclpy

from gazebo_msgs.msg import ContactsState
from gazebo_msgs.srv import DeleteEntity, SpawnEntity
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Empty, Float32MultiArray


class GateExperimentManager(Node):
    def __init__(self):
        super().__init__("gate_experiment_manager")

        self.declare_parameter("episodes", 20)
        self.declare_parameter("timeout_s", 75.0)
        self.declare_parameter(
            "output_path",
            str(Path.home() / "turtlebot3_ws" / "gazebo_gate_results.txt"),
        )
        self.declare_parameter("seed", 42)

        self.episodes = int(self.get_parameter("episodes").value)
        self.timeout_s = float(self.get_parameter("timeout_s").value)
        self.output_path = Path(str(self.get_parameter("output_path").value)).expanduser()
        self.seed = int(self.get_parameter("seed").value)

        random.seed(self.seed)
        np.random.seed(self.seed)

        self.world_x_min = 0.0
        self.world_x_max = 5.0
        self.world_y_min = 0.0
        self.world_y_max = 2.0

        self.robot_length = 0.138
        self.robot_width = 0.178
        self.half_length = self.robot_length / 2.0
        self.half_width = self.robot_width / 2.0

        self.wall_margin = 0.015

        # Small tolerance for pass/miss classification.
        # This is evaluator tolerance, not physical collision tolerance.
        self.gate_pass_tolerance = 0.080
        self.gate_clearance_margin = 0.050

        self.pole_radius = 0.05
        self.clear_opening = 0.45
        self.gate_x_positions = [2.0, 3.0, 4.0]

        self.gate_center_train_low = 0.80
        self.gate_center_train_high = 1.20

        self.spawn_x_min = 0.30
        self.spawn_x_max = 1.70
        self.spawn_y_train_min = 0.30
        self.spawn_y_train_max = 1.70
        self.spawn_theta_min = -0.20
        self.spawn_theta_max = 0.20

        self.robot_entity_name = "turtlebot3_burger"

        self.burger_sdf_path = (
            Path.home()
            / "turtlebot3_ws"
            / "install"
            / "turtlebot3_gazebo"
            / "share"
            / "turtlebot3_gazebo"
            / "models"
            / "turtlebot3_burger"
            / "model.sdf"
        )

        if not self.burger_sdf_path.exists():
            raise FileNotFoundError(f"Missing TurtleBot3 Burger SDF: {self.burger_sdf_path}")

        self.gate_models = [
            "gate1_lower",
            "gate1_upper",
            "gate2_lower",
            "gate2_upper",
            "gate3_lower",
            "gate3_upper",
        ]

        self.gt_msg = None
        self.last_gate_contact_time = -999.0
        self.last_gate_contact_count = 0

        self.create_subscription(Odometry, "/tb3_ground_truth", self.ground_truth_callback, 10)
        self.create_subscription(ContactsState, "/gate_contacts", self.gate_contact_callback, 10)

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.reset_pub = self.create_publisher(Empty, "/gate_experiment/reset", 10)
        self.gates_pub = self.create_publisher(Float32MultiArray, "/gate_experiment/gates", 10)

        self.spawn_client = self.create_client(SpawnEntity, "/spawn_entity")
        self.delete_client = self.create_client(DeleteEntity, "/delete_entity")

    def ground_truth_callback(self, msg: Odometry):
        self.gt_msg = msg

    def gate_contact_callback(self, msg: ContactsState):
        count = len(msg.states)
        self.last_gate_contact_count = count

        if count > 0:
            self.last_gate_contact_time = time.monotonic()

    def run(self):
        self.wait_for_services()

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self.output_path.exists()

        with self.output_path.open("a", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "episode",
                    "event",
                    "passed_gates",
                    "elapsed_s",
                    "path_length_m",
                    "avg_speed_mps",
                    "time_per_meter_s_per_m",
                    "collision_type",
                    "miss_direction",
                    "spawn_x",
                    "spawn_y_gz",
                    "spawn_y_train",
                    "spawn_theta",
                    "gate1_center_y_train",
                    "gate2_center_y_train",
                    "gate3_center_y_train",
                    "end_x",
                    "end_y_gz",
                    "end_y_train",
                    "end_theta",
                ],
            )

            if not file_exists:
                writer.writeheader()

            for episode in range(1, self.episodes + 1):
                config = self.reset_episode(episode)
                result = self.monitor_episode(episode, config)

                writer.writerow(result)
                f.flush()

                self.get_logger().info(
                    f"episode={episode} "
                    f"event={result['event']} "
                    f"passed_gates={result['passed_gates']} "
                    f"elapsed_s={result['elapsed_s']:.2f} "
                    f"path_m={result['path_length_m']:.2f} "
                    f"avg_speed={result['avg_speed_mps']:.3f} "
                    f"collision={result['collision_type']} "
                    f"miss={result['miss_direction']}"
                )

                self.publish_stop()
                time.sleep(0.5)

        self.get_logger().info(f"Finished. Results saved to: {self.output_path}")

    def wait_for_services(self):
        for name, client in [
            ("/spawn_entity", self.spawn_client),
            ("/delete_entity", self.delete_client),
        ]:
            self.get_logger().info(f"Waiting for {name}...")
            while rclpy.ok() and not client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info(f"Still waiting for {name}...")

    def wait_for_ground_truth(self):
        self.get_logger().info("Waiting for /tb3_ground_truth...")
        while rclpy.ok() and self.gt_msg is None:
            rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().info("/tb3_ground_truth received.")

    def debug_print_gate_geometry(self):
        print("")
        print("=" * 80)
        print("GEOMETRY_DEBUG_POLES")
        print("=" * 80)

        pole_radius = getattr(self, "pole_radius", 0.05)
        clear_opening = getattr(self, "gate_clear_opening", getattr(self, "clear_opening", 0.45))
        correct_offset = clear_opening / 2.0 + pole_radius
        wrong_offset = clear_opening / 2.0

        print(f"pole_radius={pole_radius}")
        print(f"clear_opening={clear_opening}")
        print(f"correct_offset={correct_offset}")
        print(f"wrong_offset={wrong_offset}")

        if hasattr(self, "gates"):
            for i, gate in enumerate(self.gates, start=1):
                if isinstance(gate, dict):
                    gx = gate.get("x")
                    cy_train = gate.get("center_y")
                else:
                    gx = getattr(gate, "x", None)
                    cy_train = getattr(gate, "center_y", None)

                if gx is None or cy_train is None:
                    print(f"gate{i}: could not read gate object: {gate}")
                    continue

                cy_gz = cy_train - 1.0

                print(f"gate{i}: x={gx:.4f}")
                print(f"  center_y_train={cy_train:.4f}")
                print(f"  center_y_gazebo={cy_gz:.4f}")
                print(f"  EXPECTED lower_pole_y_gz={cy_gz - correct_offset:.4f}")
                print(f"  EXPECTED upper_pole_y_gz={cy_gz + correct_offset:.4f}")
                print(f"  WRONG    lower_pole_y_gz={cy_gz - wrong_offset:.4f}")
                print(f"  WRONG    upper_pole_y_gz={cy_gz + wrong_offset:.4f}")

        print("=" * 80)
        print("")

    def reset_episode(self, episode):
        self.publish_stop()

        self.delete_entity(self.robot_entity_name)

        for name in self.gate_models:
            self.delete_entity(name)

        time.sleep(0.5)

        gate_centers_train = [
            float(random.uniform(self.gate_center_train_low, self.gate_center_train_high))
            for _ in self.gate_x_positions
        ]

        self.spawn_gates(gate_centers_train)

        spawn_x = float(random.uniform(self.spawn_x_min, self.spawn_x_max))
        spawn_y_train = float(random.uniform(self.spawn_y_train_min, self.spawn_y_train_max))
        spawn_y_gz = self.train_y_to_gazebo_y(spawn_y_train)
        spawn_theta = float(random.uniform(self.spawn_theta_min, self.spawn_theta_max))

        self.gt_msg = None
        self.last_gate_contact_time = -999.0
        self.last_gate_contact_count = 0

        self.spawn_robot(spawn_x, spawn_y_gz, spawn_theta)
        self.wait_for_ground_truth()

        for _ in range(15):
            self.publish_stop()
            self.publish_gate_info(gate_centers_train)
            self.reset_pub.publish(Empty())
            rclpy.spin_once(self, timeout_sec=0.03)
            time.sleep(0.02)

        self.get_logger().info(
            f"Reset episode {episode}: "
            f"spawn=({spawn_x:.2f}, {spawn_y_gz:.2f}, {spawn_theta:.2f}), "
            f"gates_train_y={[round(v, 3) for v in gate_centers_train]}"
        )

        return {
            "spawn_x": spawn_x,
            "spawn_y_gz": spawn_y_gz,
            "spawn_y_train": spawn_y_train,
            "spawn_theta": spawn_theta,
            "gate_centers_train": gate_centers_train,
        }

    def monitor_episode(self, episode, config):
        passed_gates = 0
        event = "running"
        collision_type = ""
        miss_direction = ""

        start_time = time.monotonic()
        last_xy = None
        previous_gate_pose = None  # previous robot center pose for gate-line interpolation
        path_length = 0.0

        end_x = 0.0
        end_y_gz = 0.0
        end_y_train = 0.0
        end_theta = 0.0

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.03)

            if self.gt_msg is None:
                continue

            x_gz, y_gz, theta = self.read_ground_truth_pose()
            x_train = x_gz
            y_train = self.gazebo_y_to_train_y(y_gz)

            end_x = x_train
            end_y_gz = y_gz
            end_y_train = y_train
            end_theta = theta

            elapsed_s = time.monotonic() - start_time

            if last_xy is not None:
                dx = x_train - last_xy[0]
                dy = y_train - last_xy[1]
                step_dist = math.hypot(dx, dy)

                if step_dist < 0.25:
                    path_length += step_dist

            last_xy = (x_train, y_train)

            # Real Gazebo pole collision from contact sensor.
            if time.monotonic() - self.last_gate_contact_time < 0.25:
                collision_type = "gate_pole"
                event = "collision"
                break

            # Wall collision from ground-truth pose and arena bounds.
            wall_collision = self.detect_wall_collision(x_train, y_train, theta)
            if wall_collision:
                collision_type = "wall"
                event = "collision"
                break

            # Gate pass / miss.
            #
            # Judge the gate only after the robot's REAR has cleared the gate
            # line. This avoids false missed_gate_above/below while the robot
            # is still entering the opening at an angle.
            if passed_gates < len(self.gate_x_positions):
                gate_x = self.gate_x_positions[passed_gates]
                gate_center_y = config["gate_centers_train"][passed_gates]

                rear_x = self.rear_x_at_pose(x_train, y_train, theta)

                if rear_x >= gate_x + self.gate_clearance_margin:
                    if self.crossed_gate_cleanly(gate_center_y, y_train, theta):
                        passed_gates += 1

                        self.get_logger().info(
                            f"Passed gate {passed_gates}: "
                            f"robot_y={y_train:.3f}, "
                            f"gate_center_y={gate_center_y:.3f}, "
                            f"theta={theta:.3f}, "
                            f"rear_x={rear_x:.3f}"
                        )

                        if passed_gates == len(self.gate_x_positions):
                            event = "success"
                            break
                    else:
                        if y_train > gate_center_y:
                            miss_direction = "above"
                            event = "missed_gate_above"
                        else:
                            miss_direction = "below"
                            event = "missed_gate_below"

                        self.get_logger().warn(
                            f"Missed gate {passed_gates + 1}: "
                            f"direction={miss_direction}, "
                            f"robot_y={y_train:.3f}, "
                            f"gate_center_y={gate_center_y:.3f}, "
                            f"theta={theta:.3f}, "
                            f"rear_x={rear_x:.3f}"
                        )
                        break

            if elapsed_s >= self.timeout_s:
                event = "timeout"
                break

        elapsed_s = max(1e-6, time.monotonic() - start_time)
        avg_speed = path_length / elapsed_s
        time_per_meter = elapsed_s / path_length if path_length > 1e-6 else 0.0

        gate_centers = config["gate_centers_train"]

        return {
            "episode": episode,
            "event": event,
            "passed_gates": passed_gates,
            "elapsed_s": round(elapsed_s, 4),
            "path_length_m": round(path_length, 4),
            "avg_speed_mps": round(avg_speed, 4),
            "time_per_meter_s_per_m": round(time_per_meter, 4),
            "collision_type": collision_type,
            "miss_direction": miss_direction,
            "spawn_x": round(config["spawn_x"], 4),
            "spawn_y_gz": round(config["spawn_y_gz"], 4),
            "spawn_y_train": round(config["spawn_y_train"], 4),
            "spawn_theta": round(config["spawn_theta"], 4),
            "gate1_center_y_train": round(gate_centers[0], 4),
            "gate2_center_y_train": round(gate_centers[1], 4),
            "gate3_center_y_train": round(gate_centers[2], 4),
            "end_x": round(end_x, 4),
            "end_y_gz": round(end_y_gz, 4),
            "end_y_train": round(end_y_train, 4),
            "end_theta": round(end_theta, 4),
        }

    def publish_gate_info(self, gate_centers_train):
        msg = Float32MultiArray()
        msg.data = [float(v) for v in gate_centers_train]
        self.gates_pub.publish(msg)

    def publish_stop(self):
        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = 0.0
        self.cmd_pub.publish(msg)

    def spawn_gates(self, gate_centers_train):
        for i, gate_x in enumerate(self.gate_x_positions):
            center_train = gate_centers_train[i]
            center_gz = self.train_y_to_gazebo_y(center_train)

            offset = self.clear_opening / 2.0 + self.pole_radius

            lower_y_gz = center_gz - offset
            upper_y_gz = center_gz + offset

            self.spawn_pole(f"gate{i + 1}_lower", gate_x, lower_y_gz)
            self.spawn_pole(f"gate{i + 1}_upper", gate_x, upper_y_gz)

    def spawn_robot(self, x, y_gz, yaw):
        base_sdf = self.burger_sdf_path.read_text()

        p3d_plugin = """
    <plugin name="tb3_ground_truth_p3d" filename="libgazebo_ros_p3d.so">
      <ros>
        <namespace>/</namespace>
        <remapping>odom:=/tb3_ground_truth</remapping>
      </ros>
      <body_name>base_footprint</body_name>
      <frame_name>world</frame_name>
      <update_rate>50.0</update_rate>
      <xyz_offset>0 0 0</xyz_offset>
      <rpy_offset>0 0 0</rpy_offset>
      <gaussian_noise>0.0</gaussian_noise>
    </plugin>
"""

        if "tb3_ground_truth_p3d" not in base_sdf:
            base_sdf = base_sdf.replace("</model>", p3d_plugin + "\n  </model>", 1)

        req = SpawnEntity.Request()
        req.name = self.robot_entity_name
        req.xml = base_sdf
        req.robot_namespace = ""
        req.reference_frame = "world"

        req.initial_pose.position.x = float(x)
        req.initial_pose.position.y = float(y_gz)
        req.initial_pose.position.z = 0.01

        qz, qw = self.yaw_to_quaternion_z_w(yaw)
        req.initial_pose.orientation.z = qz
        req.initial_pose.orientation.w = qw

        self.call_service(self.spawn_client, req)

    def spawn_pole(self, name, x, y):
        sdf = f"""
<sdf version="1.6">
  <model name="{name}">
    <static>true</static>
    <link name="link">
      <collision name="collision">
        <geometry>
          <cylinder>
            <radius>{self.pole_radius}</radius>
            <length>0.60</length>
          </cylinder>
        </geometry>
      </collision>

      <visual name="visual">
        <geometry>
          <cylinder>
            <radius>{self.pole_radius}</radius>
            <length>0.60</length>
          </cylinder>
        </geometry>
        <material>
          <script>
            <uri>file://media/materials/scripts/gazebo.material</uri>
            <name>Gazebo/Red</name>
          </script>
        </material>
      </visual>

      <sensor name="{name}_contact_sensor" type="contact">
        <always_on>true</always_on>
        <update_rate>100.0</update_rate>
        <contact>
          <collision>collision</collision>
        </contact>
        <plugin name="{name}_bumper" filename="libgazebo_ros_bumper.so">
          <ros>
            <namespace>/</namespace>
            <remapping>bumper_states:=/gate_contacts</remapping>
          </ros>
          <frame_name>world</frame_name>
        </plugin>
      </sensor>
    </link>
  </model>
</sdf>
"""

        req = SpawnEntity.Request()
        req.name = name
        req.xml = sdf
        req.robot_namespace = ""
        req.reference_frame = "world"

        req.initial_pose.position.x = float(x)
        req.initial_pose.position.y = float(y)
        req.initial_pose.position.z = 0.30
        req.initial_pose.orientation.w = 1.0

        self.call_service(self.spawn_client, req)

    def delete_entity(self, name):
        req = DeleteEntity.Request()
        req.name = name

        try:
            self.call_service(self.delete_client, req)
        except Exception:
            pass

    def call_service(self, client, req):
        future = client.call_async(req)

        while rclpy.ok() and not future.done():
            rclpy.spin_once(self, timeout_sec=0.05)

        result = future.result()

        if result is None:
            raise RuntimeError("Service call failed.")

        return result

    def read_ground_truth_pose(self):
        msg = self.gt_msg

        p = msg.pose.pose.position
        q = msg.pose.pose.orientation

        yaw = self.yaw_from_quaternion(q.x, q.y, q.z, q.w)

        return float(p.x), float(p.y), yaw

    def detect_wall_collision(self, x, y, theta):
        corners = self.robot_corners(x, y, theta)

        for cx, cy in corners:
            if cx <= self.world_x_min + self.wall_margin:
                return True
            if cx >= self.world_x_max - self.wall_margin:
                return True
            if cy <= self.world_y_min + self.wall_margin:
                return True
            if cy >= self.world_y_max - self.wall_margin:
                return True

        return False

    def crossed_gate_cleanly(self, gate_center_y, robot_y, theta):
        projected_half_y = (
            self.half_width * abs(math.cos(theta))
            + self.half_length * abs(math.sin(theta))
        )

        allowable = (
            self.clear_opening / 2.0
            - projected_half_y
            - 0.005
            + self.gate_pass_tolerance
        )

        if allowable <= 0.0:
            return False

        return abs(robot_y - gate_center_y) <= allowable

    def robot_corners(self, x, y, theta):
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

    def front_x_at_pose(self, x, y, theta):
        corners = self.robot_corners(x, y, theta)
        return float(np.max(corners[:, 0]))

    def rear_x_at_pose(self, x, y, theta):
        corners = self.robot_corners(x, y, theta)
        return float(np.min(corners[:, 0]))

    @staticmethod
    def train_y_to_gazebo_y(y_train):
        return y_train - 1.0

    @staticmethod
    def gazebo_y_to_train_y(y_gz):
        return y_gz + 1.0

    @staticmethod
    def yaw_to_quaternion_z_w(yaw):
        return math.sin(yaw / 2.0), math.cos(yaw / 2.0)

    @staticmethod
    def yaw_from_quaternion(x, y, z, w):
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def wrap_angle(angle):
        return (angle + math.pi) % (2.0 * math.pi) - math.pi


def main(args=None):
    rclpy.init(args=args)

    node = GateExperimentManager()

    try:
        node.run()
    finally:
        node.publish_stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
