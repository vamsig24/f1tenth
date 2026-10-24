"""Closed-loop MPPI vs f110_gym without the ROS graph, faster than real time.

Builds the real MPPI_Node (same params file, same control_step) and feeds it
odometry + scans straight from f110_gym, stepping the sim at 100 Hz like
gym_bridge and calling control_step at 40 Hz on sim time. Deterministic for a
given seed, and the planner's internals (rollouts, a_opt, obstacle field) are
right there to inspect.

Run inside the container after `colcon build`:
    python3 mppi_offline_harness.py [map] [sim_secs] [-p name:=value ...] [--save=log.npz]
                                    [--debug=t0:t1]   (also save planner internals in [t0, t1])
e.g.
    python3 mppi_offline_harness.py levine_combined 60 -p obstacle_cost_weight:=60.0
"""
import sys
import time

import gym
import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

sys.path.insert(0, "/sim_ws/src")
from mppi_obstacle_eval import CSV, P, w, score  # noqa: E402

SHARE = "/sim_ws/install/mppi_bringup/share/mppi_bringup"
PARAMS = f"{SHARE}/config/params_sim_lev_blocked.yaml"
WALL_YAML = f"{SHARE}/maps/levine_blocked.yaml"
MAPS = "/sim_ws/src/f1tenth_gym_ros/maps/"
SIM_DT = 0.01          # gym_bridge drive_timer
CONTROL_HZ = 40.0      # control_loop_hz
SCAN_FOV, SCAN_BEAMS, LIDAR_OFFSET = 4.7, 1080, 0.275  # sim.yaml


def parse_args(argv):
    pos, overrides, save, debug = [], [], None, None
    it = iter(argv)
    for a in it:
        if a == "-p":
            overrides += ["-p", next(it)]
        elif a.startswith("--save="):
            save = a.split("=", 1)[1]
        elif a.startswith("--debug="):
            debug = tuple(float(x) for x in a.split("=", 1)[1].split(":"))
        else:
            pos.append(a)
    map_name = pos[0] if pos else "levine_combined"
    secs = float(pos[1]) if len(pos) > 1 else 60.0
    return map_name, secs, overrides, save, debug


def yaw_to_quat(yaw):
    return 0.0, 0.0, float(np.sin(yaw / 2)), float(np.cos(yaw / 2))


def make_odom(obs, t):
    m = Odometry()
    m.header.stamp.sec = int(t)
    m.header.stamp.nanosec = int((t - int(t)) * 1e9)
    m.header.frame_id = "map"
    m.pose.pose.position.x = float(obs["poses_x"][0])
    m.pose.pose.position.y = float(obs["poses_y"][0])
    q = yaw_to_quat(float(obs["poses_theta"][0]))
    (m.pose.pose.orientation.x, m.pose.pose.orientation.y,
     m.pose.pose.orientation.z, m.pose.pose.orientation.w) = q
    m.twist.twist.linear.x = float(obs["linear_vels_x"][0])
    m.twist.twist.linear.y = float(obs["linear_vels_y"][0])
    m.twist.twist.angular.z = float(obs["ang_vels_z"][0])
    return m


def make_scan(obs):
    s = LaserScan()
    s.angle_min = -SCAN_FOV / 2
    s.angle_max = SCAN_FOV / 2
    s.angle_increment = SCAN_FOV / SCAN_BEAMS
    s.range_max = 30.0
    s.ranges = [float(r) for r in obs["scans"][0]]
    return s


def main():
    map_name, secs, overrides, save, debug = parse_args(sys.argv[1:])
    rclpy.init(args=[
        "--ros-args", "-r", "__node:=lmppi_node", "--params-file", PARAMS,
        "-p", "is_sim:=true", "-p", f"wpt_path:={CSV}", "-p", "wpt_path_absolute:=true",
        "-p", f"wall_cost_map_yaml:={WALL_YAML}",
        # Wall time and sim time run at different rates here by design.
        "-p", "mppi_guard_on_timing_jump:=false",
        "-p", "stats_log_interval:=0.0",
        *overrides,
    ])
    from mppi_example.mppi_node import MPPI_Node
    node = MPPI_Node()

    env = gym.make("f110_gym:f110-v0", map=MAPS + map_name, map_ext=".png",
                   num_agents=1, lidar_dist=LIDAR_OFFSET)
    obs, _, _, _ = env.reset(np.array([[P[0, 0], P[0, 1], w[0, 3]]]))

    log, plans, snaps = [], [], []
    t, next_control = 0.0, 0.0
    steer, speed = 0.0, 0.0
    wall0 = time.time()
    while t < secs:
        if t >= next_control - 1e-9:
            node.latest_scan = make_scan(obs)
            node.latest_scan_time = node.now_sec()
            node.control_step(make_odom(obs, t + 1.0))
            steer, speed = float(node.control[0]), float(node.control[1])
            plans.append(np.asarray(node.mppi.traj_opt)[:, :2])
            if debug and debug[0] <= t <= debug[1]:
                snaps.append(dict(
                    t=t, state=np.asarray([obs["poses_x"][0], obs["poses_y"][0], obs["poses_theta"][0],
                                           obs["linear_vels_x"][0]]),
                    reference=np.asarray(node.infer_env.reference), traj_opt=np.asarray(node.mppi.traj_opt),
                    samples=np.asarray(node.mppi.sampled_states), a_opt=np.asarray(node.mppi.a_opt),
                    points=node.last_obstacle_points, control=np.asarray(node.control)))
            next_control += 1.0 / CONTROL_HZ
        obs, _, _, _ = env.step(np.array([[steer, speed]]))
        t += SIM_DT
        log.append((t, obs["poses_x"][0], obs["poses_y"][0], obs["poses_theta"][0],
                    obs["linear_vels_x"][0], steer, speed))

    log = np.asarray(log)
    print(f"sim {secs:.0f} s in {time.time() - wall0:.1f} s wall")
    metrics = score(map_name, log[:, :5])
    if save:
        np.savez(save, log=log, plans=np.asarray(plans), snaps=np.asarray(snaps, dtype=object))
    node.destroy_node()
    rclpy.shutdown()
    return metrics


if __name__ == "__main__":
    main()
