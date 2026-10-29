"""ROS 2 node: reference line + LiDAR -> /drive via MPPI.

Parameters (the only things to set per deployment):
  waypoints_file  CSV with x, y in the first two columns
  sim             true: /ego_racecar/odom (pose + speed), gym vehicle model
                  false: /pf/viz/inferred_pose for pose, /odom for speed, our car's
                  measured model (vehicle.CAR), /odom speed corrected against the PF
  max_speed       m/s
  friction        tyre-floor friction coefficient; sets the grip limit (friction * g)
  samples, iterations   0 (default): picked at startup to fit the solve budget; set them
                  to skip that search (each count tried is one more compile)
  viz             false: publish no markers (RViz or a bag recording /mppi/* markers costs
                  the Jetson's CPU several milliseconds per tick)

Prediction check: every tick the plan's start pose (the pose predicted through the
actuation latency from commands already sent, so a pure test of the vehicle model
and latency) and the plan itself are published on /mppi/prediction (nav_msgs/Path,
each pose stamped with the time it predicts). The status line every 5 s compares
them with the poses that then arrived: position error (median, p90), its along-track
part (+: the model is ahead of the car, i.e. predicts it faster) and cross-track part
(+: the model is left of the car), and heading error. The plan horizons (0.5, 1 s)
also include replanning, so they are larger. scripts/model_check.py does the same
offline from a bag and fits the model to it.

Everything else (vehicle geometry and dynamics, latency) is in vehicle.py.
The pose's age is measured from its header stamp, so only the actuation part
of the latency is a constant.
"""
import math
import os
import time
from collections import deque

# JAX grabs 75% of GPU memory up front by default; on the Jetson that memory is
# the RAM ROS runs in. Must be set before JAX is imported (by the controller).
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
# Keep compiled solvers on disk: compiling takes minutes on a Jetson, loading seconds.
os.environ.setdefault('JAX_COMPILATION_CACHE_DIR', os.path.expanduser('~/.cache/scan_mppi/jax'))
os.environ.setdefault('JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS', '0')
os.environ.setdefault('JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES', '-1')

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Point, PoseStamped
from std_msgs.msg import ColorRGBA
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray

from . import planner as mp
from . import vehicle as veh
from .controller import Controller

CONTROL_HZ = 40.0
STALE_POSE = 0.25   # s without a pose -> stop
STALE_SCAN = 0.25   # s without a scan -> stop
VIZ_HZ = 10.0       # plan and reference
SLOW_VIZ_HZ = 2.0   # sampled rollouts and scan points: thousands of points, slow to build in Python
VIZ_ROLLOUTS = 24   # of the planner's VIZ_SAMPLES
MAX_AGE = 0.12      # s: pose ages beyond this are clock trouble, not latency
PRED_HORIZONS = (0.5, 1.0)   # s along the plan scored besides its start pose
PRED_MIN_SPEED = 0.5         # m/s: only score predictions while the car is moving


def stamp_of(msg):
    s = msg.header.stamp
    return s.sec + 1e-9 * s.nanosec


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class ScanMPPI(Node):
    def __init__(self):
        super().__init__('scan_mppi')
        sim = self.declare_parameter('sim', True).value
        max_speed = float(self.declare_parameter('max_speed', 8.0).value)
        friction = float(self.declare_parameter('friction', 1.0489 if sim else 0.6).value)
        path = self.declare_parameter('waypoints_file', '').value
        samples = int(self.declare_parameter('samples', 0).value)
        iterations = int(self.declare_parameter('iterations', 0).value)
        self.viz_on = bool(self.declare_parameter('viz', True).value)

        self.ctrl = Controller(path, sim, max_speed, friction)
        if samples:
            self.ctrl.n_samples = samples
        if iterations:
            self.ctrl.iterations = iterations
        self.get_logger().info(f'{path}: {self.ctrl.describe()}')
        self.pose = None          # (x, y, yaw)
        self.pose_time = None
        self.pose_stamp = None    # header time of the pose, for its age
        self.solve_time = 0.01    # running estimate (s)
        self.last_age = 0.0
        self.speed = 0.0
        self.yaw_rate = None      # measured, from odometry
        self.scan_time = None
        self.stats = []
        self.pose_hist = deque(maxlen=500)   # (stamp, x, y, yaw): ~10 s at 50 Hz
        self.preds = deque(maxlen=3000)      # (stamp predicted for, horizon, x, y, yaw)
        self.pred_errs = []                  # scored (horizon, err, along, cross, dyaw) arrays, for the run summary

        sensor = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                            durability=DurabilityPolicy.VOLATILE)
        self.pose_topic = '/ego_racecar/odom' if sim else '/pf/viz/inferred_pose'
        if sim:
            self.create_subscription(Odometry, '/ego_racecar/odom', self.on_odom, sensor)
        else:
            self.create_subscription(PoseStamped, '/pf/viz/inferred_pose', self.on_pose, sensor)
            self.create_subscription(Odometry, '/odom', self.on_speed, sensor)
        self.create_subscription(LaserScan, '/scan', self.on_scan, sensor)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 1)
        self.pred_pub = self.create_publisher(Path, '/mppi/prediction', 10)
        # Same topics and BEST_EFFORT QoS as the displays in f1tenth_gym_ros/launch/gym_bridge.rviz
        self.viz = {name: self.create_publisher(MarkerArray, '/mppi/' + name, sensor)
                    for name in ('reference', 'optimal_trajectory', 'sampled_trajectories', 'scan_obstacles')}

        t0 = time.time()
        self.get_logger().info('compiling the planner (minutes on a Jetson the first time, then cached)')
        self.ctrl.warm_up(self.get_logger().info, search=not samples and not iterations)
        self.get_logger().info(f'MPPI ready in {time.time() - t0:.1f} s, {self.ctrl.n_samples} samples, '
                               f'{self.ctrl.iterations} iteration(s)')
        self.create_timer(1.0 / CONTROL_HZ, self.tick)
        self.create_timer(5.0, self.report)
        if self.viz_on:
            self.create_timer(1.0 / VIZ_HZ, self.publish_viz)
            self.create_timer(1.0 / SLOW_VIZ_HZ, self.publish_slow_viz)

    def on_odom(self, msg):
        p = msg.pose.pose
        self.pose = (p.position.x, p.position.y, yaw_of(p.orientation))
        self.pose_time = time.time()
        self.pose_stamp = stamp_of(msg)
        self.speed = msg.twist.twist.linear.x
        self.yaw_rate = msg.twist.twist.angular.z
        self.pose_hist.append((self.pose_stamp,) + self.pose)

    def on_pose(self, msg):
        p = msg.pose
        self.pose = (p.position.x, p.position.y, yaw_of(p.orientation))
        self.pose_time = time.time()
        self.pose_stamp = stamp_of(msg)
        self.pose_hist.append((self.pose_stamp,) + self.pose)

    def on_speed(self, msg):
        self.speed = msg.twist.twist.linear.x

    def on_scan(self, msg):
        if self.pose is None:
            return
        self.ctrl.set_scan(msg.ranges, msg.angle_min, msg.angle_increment, msg.range_min, msg.range_max, self.pose)
        self.scan_time = time.time()

    def tick(self):
        now = time.time()
        if (self.pose_time is None or now - self.pose_time > STALE_POSE
                or self.scan_time is None or now - self.scan_time > STALE_SCAN):
            missing = [name for name, last, limit in (('pose (' + self.pose_topic + ')', self.pose_time, STALE_POSE),
                                                      ('/scan', self.scan_time, STALE_SCAN))
                       if last is None or now - last > limit]
            self.get_logger().warn(f'stopped: no {" or ".join(missing)}', throttle_duration_sec=2.0)
            self.publish(0.0, 0.0)
            self.ctrl.reset_timing()
            return
        # How old the pose is when the command goes out: header age (the pose
        # filter stamps with its scan's time) plus the solve. Ignore a stamp
        # from an unsynchronised clock and fall back to the receive time.
        ros_now = self.get_clock().now().nanoseconds * 1e-9
        age = ros_now - self.pose_stamp
        if not 0.0 <= age <= MAX_AGE:
            age = now - self.pose_time
        age = min(age, MAX_AGE) + self.solve_time
        self.last_age = age
        steer, speed = self.ctrl.step(self.pose, self.speed, now, age, self.yaw_rate)
        self.publish(steer, speed)
        if self.ctrl.planned_at == now:
            self.log_prediction(ros_now)
        dt = time.time() - now
        self.solve_time += 0.1 * (dt - self.solve_time)
        self.stats.append(dt)

    def publish(self, steer, speed):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.drive.steering_angle = steer
        msg.drive.speed = speed
        self.drive_pub.publish(msg)

    def log_prediction(self, ros_now):
        """Keep this tick's predictions for scoring and publish them for bags. The
        plan starts where the car is predicted to be when the command acts:
        the pose's stamp + its age + solve + actuation latency = now + solve + latency."""
        ctrl = self.ctrl
        t0 = ros_now + self.solve_time + ctrl.latency
        x, y, yaw = ctrl.plan_pose
        c, s = math.cos(yaw), math.sin(yaw)
        plan = ctrl.plan[:, :2]
        pts = np.concatenate([[[0.0, 0.0]], plan])
        heading = np.arctan2(*np.gradient(pts, axis=0)[:, ::-1].T)
        moving = np.hypot(*np.diff(pts, axis=0).T) > 1e-3
        heading[1:][~moving] = 0.0
        heading[0] = 0.0
        times = t0 + np.arange(len(pts)) * mp.DT
        mx = x + c * pts[:, 0] - s * pts[:, 1]
        my = y + s * pts[:, 0] + c * pts[:, 1]
        myaw = yaw + heading
        self.preds.append((t0, 0.0, x, y, yaw))
        for h in PRED_HORIZONS:
            k = int(round(h / mp.DT))
            if k < len(pts):
                self.preds.append((times[k], h, mx[k], my[k], myaw[k]))
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        for tk, px, py, pyaw in zip(times, mx, my, myaw):
            ps = PoseStamped()
            ps.header.frame_id = 'map'
            ps.header.stamp.sec = int(tk)
            ps.header.stamp.nanosec = int((tk - int(tk)) * 1e9)
            ps.pose.position.x, ps.pose.position.y = float(px), float(py)
            ps.pose.orientation.z, ps.pose.orientation.w = math.sin(pyaw / 2), math.cos(pyaw / 2)
            msg.poses.append(ps)
        self.pred_pub.publish(msg)

    def prediction_report(self):
        """Score the predictions whose time has come against the poses received."""
        if len(self.pose_hist) < 10 or not self.preds:
            return ''
        h = np.array(self.pose_hist)
        ts, hx, hy, hyaw = h[:, 0], h[:, 1], h[:, 2], np.unwrap(h[:, 3])
        rows = []
        while self.preds and self.preds[0][0] <= ts[-1]:
            rows.append(self.preds.popleft())
        rows = [r for r in rows if r[0] >= ts[0]]
        if not rows:
            return ''
        r = np.array(rows)
        t, ax, ay, ayaw = r[:, 0], np.interp(r[:, 0], ts, hx), np.interp(r[:, 0], ts, hy), np.interp(r[:, 0], ts, hyaw)
        v = np.hypot(np.interp(t + 0.05, ts, hx) - np.interp(t - 0.05, ts, hx),
                     np.interp(t + 0.05, ts, hy) - np.interp(t - 0.05, ts, hy)) / 0.1
        dx, dy = r[:, 2] - ax, r[:, 3] - ay
        along = dx * np.cos(ayaw) + dy * np.sin(ayaw)
        cross = -dx * np.sin(ayaw) + dy * np.cos(ayaw)
        dyaw = np.degrees(np.angle(np.exp(1j * (r[:, 4] - ayaw))))
        e = np.stack([r[:, 1], np.hypot(dx, dy), along, cross, dyaw], 1)[v > PRED_MIN_SPEED]
        self.pred_errs.append(e)
        return self.grade_predictions(e, 'pred err')

    def grade_predictions(self, e, title):
        """One line per call: median/p90 error per horizon, its grade and any systematic offset."""
        out, statuses = [], []
        for hz in (0.0,) + PRED_HORIZONS:
            m = e[:, 0] == hz
            if m.sum() < 5:
                continue
            err, along, cross = 100 * e[m, 1], 100 * e[m, 2].mean(), 100 * e[m, 3].mean()
            lat = self.ctrl.latency + self.solve_time
            status, hints = veh.grade(hz if hz > 0.0 else lat, np.median(err), along, cross)
            statuses.append(status)
            label = f'{1e3 * lat:.0f} ms' if hz == 0.0 else f'{hz:.1f} s'
            out.append(f'@{label} {np.median(err):.0f}/{np.percentile(err, 90):.0f} cm {status} '
                       f'(along {along:+.0f} cross {cross:+.0f}, yaw {np.median(np.abs(e[m, 4])):.1f} deg'
                       + ''.join(f', {h}' for h in hints) + ')')
        return f'{title} [{veh.worst(statuses)}] ' + '  '.join(out) if out else ''

    def run_summary(self):
        e = np.concatenate(self.pred_errs) if self.pred_errs else np.zeros((0, 5))
        return self.grade_predictions(e, f'whole run ({len(e)} predictions): pred err') or 'no predictions scored (car never moving?)'

    def marker(self, ns, kind, width, rgba, points, colors=None, mid=0):
        m = Marker()
        m.header.frame_id = 'map'
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns, m.id, m.type = ns, mid, kind
        m.scale.x = m.scale.y = width
        m.color.r, m.color.g, m.color.b, m.color.a = rgba
        m.pose.orientation.w = 1.0
        m.points = [Point(x=float(x), y=float(y), z=0.05) for x, y in points]
        if colors is not None:
            m.colors = colors
        return m

    def publish_viz(self):
        ctrl = self.ctrl
        if ctrl.plan is None:
            return
        x, y, yaw = ctrl.plan_pose
        c, s = math.cos(yaw), math.sin(yaw)

        def to_map(local):
            return np.stack([x + c * local[..., 0] - s * local[..., 1],
                             y + s * local[..., 0] + c * local[..., 1]], -1)

        pubs = self.viz
        if pubs['optimal_trajectory'].get_subscription_count():
            # coloured by planned speed: blue slow -> red at max_speed
            f = np.clip(ctrl.plan[:, 2] / ctrl.max_speed, 0.0, 1.0)
            colors = [ColorRGBA(r=float(k), g=0.2, b=float(1 - k), a=1.0) for k in f]
            pubs['optimal_trajectory'].publish(MarkerArray(markers=[self.marker(
                'plan', Marker.LINE_STRIP, 0.08, (1.0, 1.0, 1.0, 1.0), to_map(ctrl.plan[:, :2]), colors)]))
        if pubs['reference'].get_subscription_count() and ctrl.ref_idx is not None:
            pts, _ = ctrl.ref.window(ctrl.ref_idx - mp.REF_BACK, mp.REF_POINTS)
            pubs['reference'].publish(MarkerArray(markers=[self.marker(
                'reference', Marker.LINE_STRIP, 0.04, (0.0, 0.9, 0.9, 0.8), pts)]))

    def publish_slow_viz(self):
        ctrl, pubs = self.ctrl, self.viz
        if ctrl.plan is None:
            return
        if pubs['sampled_trajectories'].get_subscription_count() and ctrl.samples is not None:
            x, y, yaw = ctrl.plan_pose
            c, s = math.cos(yaw), math.sin(yaw)
            rollouts = ctrl.samples[::max(len(ctrl.samples) // VIZ_ROLLOUTS, 1)]
            pubs['sampled_trajectories'].publish(MarkerArray(markers=[
                self.marker('samples', Marker.LINE_STRIP, 0.015, (0.6, 0.6, 0.6, 0.5),
                            np.stack([x + c * tr[:, 0] - s * tr[:, 1], y + s * tr[:, 0] + c * tr[:, 1]], -1), mid=i)
                for i, tr in enumerate(rollouts)]))
        if pubs['scan_obstacles'].get_subscription_count() and ctrl.scan_xy is not None:
            pubs['scan_obstacles'].publish(MarkerArray(markers=[self.marker(
                'scan', Marker.POINTS, 0.06, (1.0, 0.5, 0.0, 0.9), ctrl.scan_xy[::3])]))

    def report(self):
        if self.stats:
            a = np.array(self.stats) * 1e3
            self.get_logger().info(f'{len(a) / 5.0:.0f} Hz  tick mean {a.mean():.1f} ms  max {a.max():.1f} ms  '
                                   f'latency {1e3 * (self.ctrl.latency + self.last_age):.0f} ms  '
                                   f'v {self.speed * self.ctrl.speed_scale:.2f}  cmd {self.ctrl.u_nom[0, 1]:.2f}  '
                                   f'odom scale {self.ctrl.speed_scale:.2f}  recoveries {self.ctrl.recoveries}  odom dropouts {self.ctrl.dropouts}')
            self.stats = []
        pred = self.prediction_report()
        if pred:
            self.get_logger().info(pred)


def main():
    rclpy.init()
    node = ScanMPPI()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():  # under ros2 launch, SIGINT has already shut the context down
            node.publish(0.0, 0.0)
        print(node.run_summary(), flush=True)
        node.destroy_node()
        rclpy.try_shutdown()
