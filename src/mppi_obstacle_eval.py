"""Score an MPPI run on an obstacle map: laps, footprint clearance, contact.

gym_bridge ignores the collision flag, so the car can drive straight through an
obstacle without the sim noticing. This measures it instead: the 0.58 x 0.31 m
footprint is checked against the true map (static walls + obstacles), split
into wall clearance and obstacle clearance.

Usage: mppi_obstacle_eval.py [map] [secs] [--reset] [--save=path.npy]
  --reset   teleport the car to the raceline start first (via /initialpose)
  --save    dump the (t, x, y, yaw, v) log for plotting
"""
import os
import sys
import time

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from PIL import Image
from scipy.ndimage import distance_transform_edt

CSV = "/sim_ws/install/mppi_bringup/share/mppi_bringup/waypoints/old/lev_testing/lev_blocked.csv"
if not os.path.exists(CSV):  # mppi_bringup not built, e.g. when scoring a non-MPPI stack
    CSV = "/sim_ws/src/opp_aware_mppi_f1tenth/mppi_bringup/waypoints/old/lev_testing/lev_blocked.csv"
w = np.loadtxt(CSV, delimiter=";", skiprows=3)
P, S, TOTAL = w[:, 1:3], w[:, 0], w[-1, 0]

MD = "/sim_ws/src/f1tenth_gym_ros/maps/"


def load_occ(name):
    cfg = yaml.safe_load(open(f"{MD}{name}.yaml"))
    img = np.flipud(np.array(Image.open(MD + cfg["image"]).convert("L")))
    return ((255.0 - img) / 255.0) >= cfg["occupied_thresh"], cfg


static, cfg = load_occ("levine_blocked")
res, ox, oy = cfg["resolution"], cfg["origin"][0], cfg["origin"][1]
sdf_wall = distance_transform_edt(~static) * res

# Footprint perimeter, body frame, centred on the pose (f110_gym convention).
L, W = 0.58, 0.31
xs = np.linspace(-L / 2, L / 2, 13)
ys = np.linspace(-W / 2, W / 2, 7)
FOOT = np.concatenate([
    np.stack([xs, np.full_like(xs, -W / 2)], 1), np.stack([xs, np.full_like(xs, W / 2)], 1),
    np.stack([np.full_like(ys, -L / 2), ys], 1), np.stack([np.full_like(ys, L / 2), ys], 1),
])


def footprint_clearance(sdf, x, y, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    px = x + c * FOOT[:, 0] - s * FOOT[:, 1]
    py = y + s * FOOT[:, 0] + c * FOOT[:, 1]
    col = np.clip(((px - ox) / res).astype(int), 0, sdf.shape[1] - 1)
    row = np.clip(((py - oy) / res).astype(int), 0, sdf.shape[0] - 1)
    return sdf[row, col].min()


def score(map_name, log, quiet=False):
    """log: (n, 5) rows of (t, x, y, yaw, v). Prints a summary, returns the metrics."""
    occ, _ = load_occ(map_name)
    sdf_obs = distance_transform_edt(~(occ & ~static)) * res
    a = np.asarray(log)
    t, xy, yaw, v = a[:, 0] - a[0, 0], a[:, 1:3], a[:, 3], a[:, 4]
    dd = np.linalg.norm(P[None, :, :] - xy[:, None, :], axis=2)
    idx = dd.argmin(axis=1)
    err = dd.min(axis=1)
    cw = np.array([footprint_clearance(sdf_wall, *p, h) for p, h in zip(xy, yaw)])
    co = np.array([footprint_clearance(sdf_obs, *p, h) for p, h in zip(xy, yaw)])

    # Obstacle zones: raceline stretches within 0.6 m of an obstacle.
    on_line_obs = sdf_obs[((P[:, 1] - oy) / res).astype(int), ((P[:, 0] - ox) / res).astype(int)] < 0.6
    in_zone = on_line_obs[idx]

    ds = np.diff(S[idx])
    ds[ds < -TOTAL / 2] += TOTAL
    ds[ds > TOTAL / 2] -= TOTAL
    prog = np.concatenate([[0.0], np.cumsum(ds)])
    lap_marks = [t[np.argmax(prog >= k * TOTAL)] for k in range(1, int(prog[-1] // TOTAL) + 1)]
    lap_times = np.diff([0.0] + lap_marks)

    if not quiet:
        print(f"map {map_name}   duration {t[-1]:.1f} s   samples {len(a)}")
        print(f"progress        {prog[-1]:.1f} m = {prog[-1] / TOTAL:.2f} laps")
        print(f"lap times       {' '.join(f'{x:.2f}' for x in lap_times) or '-'} s")
        print(f"speed           mean {v.mean():.2f}  max {v.max():.2f} m/s   stalls(<0.3) {int((v < 0.3).sum())}")
        print(f"obstacle clr    min {co.min():.3f}  p5 {np.percentile(co, 5):.3f} m   "
              f"contact {100 * (co <= 0.0).mean():.1f}%  (<0.10m {100 * (co < 0.10).mean():.1f}%)")
        print(f"wall clr        min {cw.min():.3f}  p5 {np.percentile(cw, 5):.3f} m   "
              f"contact {100 * (cw <= 0.0).mean():.1f}%  (<0.10m {100 * (cw < 0.10).mean():.1f}%)")
        print(f"raceline err    clear-track mean {err[~in_zone].mean():.3f} p95 {np.percentile(err[~in_zone], 95):.3f}  "
              f"| obstacle-zone mean {err[in_zone].mean() if in_zone.any() else 0:.3f} m")
    return dict(laps=prog[-1] / TOTAL, lap_times=lap_times, v_mean=v.mean(), obs_min=co.min(),
                obs_contact=(co <= 0.0).mean(), wall_min=cw.min(), wall_contact=(cw <= 0.0).mean(),
                err_clear=err[~in_zone].mean(), err_zone=err[in_zone].mean() if in_zone.any() else 0.0,
                obs_clr=co, wall_clr=cw)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    map_name = args[0] if len(args) > 0 else "levine_combined"
    dur = float(args[1]) if len(args) > 1 else 90.0
    reset = "--reset" in sys.argv
    save = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--save=")), None)

    rclpy.init()
    n = rclpy.create_node("mppi_obstacle_eval")
    rec = []

    def on_odom(m):
        q = m.pose.pose.orientation
        yaw = np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        rec.append((time.time(), m.pose.pose.position.x, m.pose.pose.position.y, yaw,
                    m.twist.twist.linear.x))

    n.create_subscription(Odometry, "/ego_racecar/odom", on_odom, 10)

    if reset:
        pub = n.create_publisher(PoseWithCovarianceStamped, "/initialpose", 10)
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = "map"
        msg.pose.pose.position.x = float(P[0, 0])
        msg.pose.pose.position.y = float(P[0, 1])
        msg.pose.pose.orientation.z = float(np.sin(w[0, 3] / 2))
        msg.pose.pose.orientation.w = float(np.cos(w[0, 3] / 2))
        t_end = time.time() + 1.0
        while time.time() < t_end:
            pub.publish(msg)
            rclpy.spin_once(n, timeout_sec=0.1)
        rec.clear()

    t0 = time.time()
    while time.time() - t0 < dur:
        rclpy.spin_once(n, timeout_sec=0.05)

    a = np.array(rec)
    if save:
        np.save(save, a)
    score(map_name, a)


if __name__ == "__main__":
    main()
