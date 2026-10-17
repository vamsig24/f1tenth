"""Measure MPPI raceline tracking: error, wall clearance, laps. Usage: soak.py [map] [secs]"""
import sys
import time

import numpy as np
import rclpy
import yaml
from nav_msgs.msg import Odometry
from PIL import Image
from scipy.ndimage import distance_transform_edt

MAP = sys.argv[1] if len(sys.argv) > 1 else "levine_blocked"
DUR = float(sys.argv[2]) if len(sys.argv) > 2 else 90.0

CSV = "/sim_ws/install/mppi_bringup/share/mppi_bringup/waypoints/old/lev_testing/lev_blocked.csv"
w = np.loadtxt(CSV, delimiter=";", skiprows=3)
P, S, TOTAL = w[:, 1:3], w[:, 0], w[-1, 0]

MD = "/sim_ws/src/f1tenth_gym_ros/maps/"
cfg = yaml.safe_load(open(f"{MD}{MAP}.yaml"))
img = np.flipud(np.array(Image.open(MD + cfg["image"]).convert("L")))
res, ox, oy = cfg["resolution"], cfg["origin"][0], cfg["origin"][1]
sdf = distance_transform_edt(((255.0 - img) / 255.0) < cfg["occupied_thresh"]) * res

rclpy.init()
n = rclpy.create_node("soak")
rec = []
n.create_subscription(
    Odometry, "/ego_racecar/odom",
    lambda m: rec.append((time.time(), m.pose.pose.position.x,
                          m.pose.pose.position.y, m.twist.twist.linear.x)), 10)

t0 = time.time()
while time.time() - t0 < DUR:
    rclpy.spin_once(n, timeout_sec=0.05)

a = np.array(rec)
xy = a[:, 1:3]
dd = np.linalg.norm(P[None, :, :] - xy[:, None, :], axis=2)
err = dd.min(axis=1)
clr = sdf[((xy[:, 1] - oy) / res).astype(int), ((xy[:, 0] - ox) / res).astype(int)]

ds = np.diff(S[dd.argmin(axis=1)])
ds[ds < -TOTAL / 2] += TOTAL
ds[ds > TOTAL / 2] -= TOTAL
prog = ds.sum()
laps = prog / TOTAL

print(f"map {MAP}   duration {a[-1,0]-a[0,0]:.1f} s   samples {len(a)}")
print(f"tracking error  mean {err.mean():.3f}  p95 {np.percentile(err,95):.3f}  max {err.max():.3f} m")
print(f"wall clearance  min {clr.min():.2f}  mean {clr.mean():.2f} m")
print(f"speed           mean {a[:,3].mean():.2f}  max {a[:,3].max():.2f} m/s")
print(f"stalls (<0.3m/s) {int((a[:,3] < 0.3).sum())} samples")
print(f"progress        {prog:.1f} m = {laps:.2f} laps (lap {TOTAL:.1f} m)")
if laps > 0.05:
    print(f"avg lap time    {(a[-1,0]-a[0,0])/laps:.1f} s")
