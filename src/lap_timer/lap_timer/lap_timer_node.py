#!/usr/bin/env python3
import csv
import os
import sys
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from ament_index_python.packages import get_package_share_directory

MAX_TRACKING_ERROR = 0.5  # m
START_POSE = (0.0, 0.0)


class LapTimer(Node):
    def __init__(self):
        super().__init__('lap_timer')
        default_wp = os.path.join(get_package_share_directory('lap_timer'), 'waypoints', 'levine.csv')
        wp_file = self.declare_parameter('waypoint_file', default_wp).value
        self.log_dir = os.path.abspath(self.declare_parameter('log_dir', 'lap_logs').value)
        self.tag = self.declare_parameter('tag', 'lap').value

        with open(wp_file) as f:
            rows = list(csv.reader(f))[1:]
        self.wp = np.array([[float(r[0]), float(r[1])] for r in rows])
        self.seg = np.roll(self.wp, -1, axis=0) - self.wp
        self.seg_len = np.linalg.norm(self.seg, axis=1)
        self.s0 = np.concatenate([[0.0], np.cumsum(self.seg_len)[:-1]])
        self.loop = float(self.seg_len.sum())

        self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_cb, 10)
        self.create_subscription(PoseWithCovarianceStamped, '/initialpose', self.reset_cb, 10)
        self.reset_pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
        self.reset_pos = None
        self.arm()
        self.get_logger().info(f'{len(self.wp)} waypoints, lap length {self.loop:.2f} m, '
                               f'max tracking error {MAX_TRACKING_ERROR} m. Logs go to {self.log_dir}')
        self.get_logger().info('Press Enter in this terminal to put the car back at the start.')
        threading.Thread(target=self.read_keys, daemon=True).start()

    def read_keys(self):
        while sys.stdin.readline():
            msg = PoseWithCovarianceStamped()
            msg.header.frame_id = 'map'
            msg.pose.pose.position.x, msg.pose.pose.position.y = START_POSE
            msg.pose.pose.orientation.w = 1.0
            self.reset_pub.publish(msg)

    def arm(self):
        self.state = 'waiting'
        self.t_start = None
        self.s_prev = None
        self.progress = 0.0
        self.log = []
        self.still_since = None
        self.get_logger().info('Waiting for the car to be placed at the start.')

    def reset_cb(self, msg):
        self.get_logger().info('Car was reset.')
        self.reset_pos = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        self.arm()

    def project(self, p):
        d = p - self.wp
        t = np.clip(np.einsum('ij,ij->i', d, self.seg) / self.seg_len ** 2, 0.0, 1.0)
        dist = np.linalg.norm(p - (self.wp + self.seg * t[:, None]), axis=1)
        i = int(np.argmin(dist))
        return self.s0[i] + t[i] * self.seg_len[i], float(dist[i])

    def odom_cb(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        pose = msg.pose.pose
        p = np.array([pose.position.x, pose.position.y])
        q = pose.orientation
        yaw = np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y ** 2 + q.z ** 2))
        v = msg.twist.twist.linear.x

        if self.state == 'done':
            return

        if self.state == 'waiting':
            at_reset = self.reset_pos is not None and np.linalg.norm(p - self.reset_pos) < 0.2
            if not (at_reset or abs(v) < 0.05):
                return
            self.state = 'armed'
            self.get_logger().info('Ready: timing starts when the car starts moving.')

        if self.state == 'armed':
            if abs(v) < 0.1:
                return
            if np.linalg.norm(p - START_POSE) > 0.5:
                self.get_logger().warn('Lap started away from the spawn pose (0, 0): this lap does not count.')
            self.state = 'running'
            self.t_start = t
            self.s_prev, _ = self.project(p)

        s, err = self.project(p)
        self.progress += (s - self.s_prev + self.loop / 2) % self.loop - self.loop / 2
        self.s_prev = s
        self.log.append((t - self.t_start, p[0], p[1], yaw, v, err))

        if abs(v) < 0.05:
            self.still_since = self.still_since or t
            if t - self.still_since > 1.0:
                self.finish(t, f'Car stopped after {self.progress:.1f} m (crash?)', 'dnf')
            return
        self.still_since = None

        if self.progress >= self.loop:
            e = np.array([r[5] for r in self.log])
            valid = e.max() <= MAX_TRACKING_ERROR
            self.finish(t, f'{t - self.t_start:.2f} s | max tracking error {e.max():.3f} m, '
                           f'mean {e.mean():.3f} m | '
                           + ('VALID' if valid else f'INVALID (max error > {MAX_TRACKING_ERROR} m)'),
                        'valid' if valid else 'invalid')

    def finish(self, t, summary, status):
        self.state = 'done'
        os.makedirs(self.log_dir, exist_ok=True)
        path = os.path.join(self.log_dir, f'{self.tag}_{time.strftime("%Y%m%d-%H%M%S")}_{status}.csv')
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['t', 'x', 'y', 'yaw', 'v', 'tracking_error'])
            w.writerows(self.log)
        self.get_logger().info(f'Lap: {summary}')
        self.get_logger().info(f'Saved {path}. Press Enter to reset the car and time another lap.')


def main():
    rclpy.init()
    try:
        rclpy.spin(LapTimer())
    except (KeyboardInterrupt, ExternalShutdownException):
        pass


if __name__ == '__main__':
    main()
