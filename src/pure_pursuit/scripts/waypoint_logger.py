#!/usr/bin/env python3
"""
Waypoint logger

Drive the car around a track and this node records its path as a list of (x, y)
waypoints in a CSV file, which your Pure Pursuit node can then load and follow.
A new waypoint is saved each time the car moves at least `min_distance` meters.

This logger uses the simulator's ground-truth odometry. On the real car,
pose comes from the particle filter instead (/pf/viz/inferred_pose, a
PoseStamped), adapt this node for it as needed.

By default the CSV is written to the `waypoints/` folder inside the pure_pursuit
package. Override the location with the `output_file` parameter.

Usage:
  ros2 run pure_pursuit waypoint_logger.py
"""

import csv
import os

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry


def default_output_file():
    """Return pure_pursuit/waypoints/waypoints.csv, resolved from this script's
    location. Falls back to ./waypoints/waypoints.csv if the package root cannot
    be found."""
    d = os.path.dirname(os.path.abspath(__file__))
    # Walk up to the package root (the directory containing package.xml)
    for _ in range(5):
        if os.path.exists(os.path.join(d, 'package.xml')):
            return os.path.join(d, 'waypoints', 'waypoints.csv')
        d = os.path.dirname(d)
    return os.path.join('waypoints', 'waypoints.csv')


class WaypointLogger(Node):
    def __init__(self):
        super().__init__('waypoint_logger')

        self.output_file = self.declare_parameter(
            'output_file', default_output_file()).value
        self.min_distance = self.declare_parameter('min_distance', 0.1).value

        # Make sure the output directory exists before we write to it.
        os.makedirs(os.path.dirname(os.path.abspath(self.output_file)), exist_ok=True)

        self.waypoints = []
        self.last_x = None
        self.last_y = None

        self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 10)

        self.get_logger().info('Waypoint logger started - drive the car!')
        self.get_logger().info(f'Saving to: {self.output_file}')

    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        if self.last_x is None or \
           ((x - self.last_x) ** 2 + (y - self.last_y) ** 2) ** 0.5 >= self.min_distance:
            self.waypoints.append([x, y])
            self.last_x = x
            self.last_y = y
            self.get_logger().info(
                f'Logged waypoint {len(self.waypoints)}: ({x:.2f}, {y:.2f})')
            self.save_waypoints()

    def save_waypoints(self):
        with open(self.output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['x', 'y'])
            writer.writerows(self.waypoints)


def main(args=None):
    rclpy.init(args=args)
    node = WaypointLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
