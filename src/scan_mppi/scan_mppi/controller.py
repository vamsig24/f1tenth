"""ROS-free controller: pose + speed + scan in, (steering, speed) command out.

The ROS node and the offline gym harness both drive this class, so what is
tested offline is exactly what runs on the car.
"""
import math
import time

import jax
import numpy as np
from scipy.ndimage import distance_transform_edt

from . import planner as mp
from . import vehicle as veh
from .reference import Reference

# Recovery: when the planner holds the car still with no way forward (a gap it
# judges too tight, or it ended up nose-in to something), reverse a short way
# with the wheels turned so the nose swings toward the line, then replan.
STUCK_SPEED = 0.15       # m/s
STUCK_TIME = 1.0         # s stopped with a stopped plan before reversing
REVERSE_SPEED = -0.45    # m/s, slow: also keeps the gym in its kinematic (stable) regime
REVERSE_TIME = 1.2       # s
REVERSE_CLEARANCE = 0.12 # m of rear clearance needed to keep reversing

# Solve time budget: warm_up() halves the sample count until a solve fits
# (the Jetson's GPU is several times slower than a desktop's).
SOLVE_BUDGET = 0.018     # s, inside a 25 ms (40 Hz) control period
HALVING_GAIN = 0.85      # halve only while it cuts the solve time by at least 15%
MIN_SAMPLES = 256

# Odometry speed scale. The VESC converts motor rpm to m/s with a calibrated
# gain; on our car /odom reads ~1.2x the speed the particle filter sees (bags
# 2026-10-25), and the same gain turns our speed command into rpm, so the car
# also drives ~0.84x what it is told. On the car the controller compares
# distance along the PF pose with distance integrated from /odom and corrects
# both directions by the ratio.
SCALE_WINDOW = 4.0       # m of odometry distance per estimate
SCALE_MIN_SPEED = 1.0    # m/s: only measure while moving properly
SCALE_LIMITS = (0.7, 1.2)
SCALE_JUMP = 0.3         # m beyond the expected step: a pose filter jump, not motion. Poses
                         # reach the node in bursts, so a tick can see two at once.

# Odometry dropouts. The car's /odom speed falls to ~0 for 20-300 ms a few times
# every 10 s while the car rolls on (2026-10-28 bag: PF and IMU show no slowdown;
# also in the pure-pursuit bags). Planning from that "stopped" state made MPPI
# command a surge, then brake once the reading came back. A reading below what
# the car can brake to is held off for up to ODOM_HOLD, and the estimate follows
# the speed loop model meanwhile.
ODOM_BRAKE = 12.0        # m/s^2, harder than the car brakes (kp_down 4 from 3 m/s)
ODOM_NOISE = 0.3         # m/s
ODOM_HOLD = 0.3          # s


class Controller:
    def __init__(self, waypoints_file, sim, max_speed, friction, latency=None, model=None):
        self.ref = Reference(waypoints_file, friction, max_speed)
        self.max_speed = float(max_speed)
        self.mu = float(friction)
        self.model = model or (veh.SIM if sim else veh.CAR)
        self.car = self.model.array()
        self.latency = self.model.latency if latency is None else float(latency)
        self.n_samples = mp.SAMPLES
        self.iterations = mp.ITERATIONS
        # odometry speed scale (car only; the gym's odometry is exact)
        self.calibrate_speed = not sim
        self.speed_scale = 1.0
        self._scale_acc = None    # (last xy, pf distance, odom distance)
        self.v_est = None         # odometry speed with dropouts removed (odometry units)
        self.dropout_since = None
        self.dropouts = 0
        self.sent = []            # (time, steer, speed) of recent commands
        self.scan_xy = None       # latest scan returns, map frame
        self.ref_idx = None
        self.steer_est = 0.0      # steering angle, integrated from our own commands
        self.last_steer_cmd = 0.0
        self.last_time = None
        self.u_nom = np.zeros((mp.STEPS, 2), np.float32)
        # The small per-tick calls (latency prediction, key split) run on the CPU:
        # on a GPU each is a chain of tiny kernels that costs milliseconds on a Jetson.
        self.cpu = jax.devices('cpu')[0]
        with jax.default_device(self.cpu):
            self.key = jax.random.PRNGKey(0)
        self.plan = None          # (T, 3) local x, y, v of the chosen plan
        self.plan_pose = None     # (x, y, yaw) the plan is relative to (the latency-predicted pose)
        self.samples = None       # (VIZ_SAMPLES, T, 3) some rollouts, same frame
        self.plan_cost = 0.0
        self.stuck_since = None
        self.reverse_until = None
        self.reverse_steer = 0.0
        self.recoveries = 0

    def describe(self):
        r = self.ref
        return (f'{r.n} points, {r.length:.1f} m, {"closed" if r.closed else "open"}, '
                f'speed cap {r.v.min():.1f}..{r.v.max():.1f} m/s')

    def set_scan(self, ranges, angle_min, angle_inc, range_min, range_max, pose):
        """Store the scan in the map frame, using the pose at scan time."""
        r = np.asarray(ranges, dtype=np.float32)
        th = angle_min + angle_inc * np.arange(len(r), dtype=np.float32)
        ok = np.isfinite(r) & (r > range_min) & (r < min(range_max, 15.0))
        lx = veh.LASER_X + r[ok] * np.cos(th[ok])
        ly = r[ok] * np.sin(th[ok])
        body = (np.abs(lx - veh.BODY_CENTER_X) < veh.LENGTH / 2 + 0.05) & (np.abs(ly) < veh.WIDTH / 2 + 0.05)
        lx, ly = lx[~body], ly[~body]  # returns off the car itself
        x, y, yaw = pose
        c, s = math.cos(yaw), math.sin(yaw)
        self.scan_xy = np.stack([x + c * lx - s * ly, y + s * lx + c * ly], 1)

    def clearance_grid(self, x, y, yaw):
        g = np.ones((mp.GRID_NY, mp.GRID_NX), bool)
        if self.scan_xy is not None and len(self.scan_xy):
            c, s = math.cos(yaw), math.sin(yaw)
            dx, dy = self.scan_xy[:, 0] - x, self.scan_xy[:, 1] - y
            lx, ly = c * dx + s * dy, -s * dx + c * dy
            ix = ((lx - mp.GRID_X0) / mp.GRID_RES).astype(int)
            iy = ((ly - mp.GRID_Y0) / mp.GRID_RES).astype(int)
            ok = (ix >= 0) & (ix < mp.GRID_NX) & (iy >= 0) & (iy < mp.GRID_NY)
            g[iy[ok], ix[ok]] = False
        return (distance_transform_edt(g) * mp.GRID_RES).astype(np.float32)

    def local_reference(self, x, y, yaw):
        self.ref_idx = self.ref.nearest(np.array([x, y]), self.ref_idx)
        pts, v = self.ref.window(self.ref_idx - mp.REF_BACK, mp.REF_POINTS)
        c, s = math.cos(yaw), math.sin(yaw)
        dx, dy = pts[:, 0] - x, pts[:, 1] - y
        local = np.stack([c * dx + s * dy, -s * dx + c * dy], 1).astype(np.float32)
        return local, v.astype(np.float32)

    def _solve(self, x, y, yaw, v, steer, yaw_rate):
        ref_xy, ref_v = self.local_reference(x, y, yaw)
        grid = self.clearance_grid(x, y, yaw)
        with jax.default_device(self.cpu):
            self.key, sub = jax.random.split(self.key)
        sub = np.asarray(sub)   # uncommitted, so the solve still runs on the default (GPU) device
        out = mp.solve(sub, self.u_nom, np.float32(v), np.float32(steer), np.float32(yaw_rate), np.float32(self.last_steer_cmd),
                       ref_xy, ref_v, np.float32(self.ref.ds), grid, np.float32(self.mu), self.car, self.max_speed,
                       self.n_samples, self.iterations)
        return [np.asarray(o) for o in out]

    def warm_up(self, log=print, search=True):
        """Compile the solver before the car is live, and pick the largest
        sample count whose solve fits SOLVE_BUDGET. On a GPU the solve is mostly
        a fixed chain of small sequential steps (a Jetson takes ~24 ms at any
        count), so stop halving once it no longer buys a real saving."""
        x, y = self.ref.xy[0]
        with jax.default_device(self.cpu):
            mp.propagate(np.float32(0), np.float32(0), np.float32(0), np.zeros((mp.PREDICT_MAX, 2), np.float32), 1,
                         np.float32(self.mu), self.car)
        def timed():
            self.ref_idx = 0
            self._solve(x, y, 0.0, 0.0, 0.0, 0.0)   # compiles for this sample count
            times = []
            for _ in range(10):
                t0 = time.perf_counter()
                self._solve(x, y, 0.0, 1.0, 0.0, 0.0)
                times.append(time.perf_counter() - t0)
            solve = float(np.median(times))
            log(f'{self.n_samples} samples, {self.iterations} iteration(s): solve {1e3 * solve:.1f} ms '
                f'on {jax.devices()[0].platform}')
            return solve

        if not search:   # sample count and iterations set by the caller
            timed()
            self.u_nom[:] = 0.0
            self.ref_idx = None
            return
        last = None
        while True:
            solve = timed()
            if last is not None and solve > HALVING_GAIN * last:
                self.n_samples *= 2   # halving saved too little: keep the larger count
                solve = last
                log(f'fewer samples barely help on this machine; using {self.n_samples}')
                break
            if solve <= SOLVE_BUDGET or self.n_samples <= MIN_SAMPLES:
                break
            last = solve
            self.n_samples //= 2
        if solve > SOLVE_BUDGET and self.iterations > 1:
            # One MPPI round per tick: the warm start from the last tick stands in for the
            # second, and drives the same in the car tests while saving a third of the solve.
            self.iterations = 1
            timed()
        self.u_nom[:] = 0.0
        self.ref_idx = None

    def reset_timing(self):
        self.last_time = None
        self.sent = []

    def _cmd_at(self, t):
        """Command sent most recently before time t (zero before the first)."""
        out = (0.0, 0.0)
        for ts, steer, speed in self.sent:
            if ts > t:
                break
            out = (steer, speed)
        return out

    def _wheel_cmds(self, t0, n):
        """(steering, speed) commands acting on the car for n substeps from t0:
        those sent `latency` earlier (steering a further STEER_DELAY)."""
        h = mp.H_SUB
        return np.array([(self._cmd_at(t0 + k * h - self.latency - veh.STEER_DELAY)[0],
                          self._cmd_at(t0 + k * h - self.latency)[1]) for k in range(n)], np.float32)

    def _send(self, now, steer, speed):
        """Record a command (true speed) and return it as sent (odometry units)."""
        self.last_steer_cmd = steer
        self.sent.append((now, steer, speed))
        keep = now - self.latency - veh.STEER_DELAY - 0.5
        while len(self.sent) > 1 and self.sent[1][0] < keep:
            self.sent.pop(0)
        return steer, speed / self.speed_scale

    def predict(self, pose, speed, yaw_rate, now, latency):
        """Pose, speed, steering and yaw rate `latency` s after the pose was
        measured, when a command sent now takes effect."""
        n = min(int(round(latency / mp.H_SUB)), mp.PREDICT_MAX)
        if n == 0 or not self.sent:
            return pose, speed, self.steer_est, yaw_rate
        cmds = np.zeros((mp.PREDICT_MAX, 2), np.float32)
        cmds[:n] = self._wheel_cmds(now + self.latency - latency, n)
        with jax.default_device(self.cpu):
            dx, dy, dyaw, steer, v, r = (float(o) for o in mp.propagate(
                np.float32(max(speed, 0.0)), np.float32(self.steer_est), np.float32(yaw_rate), cmds, n, np.float32(self.mu), self.car))
        x, y, yaw = pose
        c, s = math.cos(yaw), math.sin(yaw)
        return (x + c * dx - s * dy, y + s * dx + c * dy, yaw + dyaw), v, steer, r

    def _update_speed_scale(self, pose, speed, dt):
        xy = np.array(pose[:2])
        if self._scale_acc is None:
            self._scale_acc = (xy, 0.0, 0.0)
            return
        last, d_pf, d_odom = self._scale_acc
        step = float(np.linalg.norm(xy - last))
        if step > 2.0 * speed * dt + SCALE_JUMP:   # the pose filter jumped; start over
            self._scale_acc = (xy, 0.0, 0.0)
            return
        if speed < SCALE_MIN_SPEED:                # slow: pause the window, keep what it has
            self._scale_acc = (xy, d_pf, d_odom)
            return
        d_pf, d_odom = d_pf + step, d_odom + speed * dt
        if d_odom >= SCALE_WINDOW:
            ratio = float(np.clip(d_pf / d_odom, *SCALE_LIMITS))
            self.speed_scale += 0.5 * (ratio - self.speed_scale)
            d_pf = d_odom = 0.0
        self._scale_acc = (xy, d_pf, d_odom)

    def _filter_speed(self, speed, now):
        """Odometry speed with dropouts replaced by the speed loop's prediction."""
        if self.v_est is None or self.last_time is None:
            self.v_est = speed
            return speed
        dt = min(max(now - self.last_time, 0.0), 0.1)
        if speed >= self.v_est - ODOM_BRAKE * dt - ODOM_NOISE:
            self.v_est, self.dropout_since = speed, None
            return speed
        if self.dropout_since is None:
            self.dropout_since = now
            self.dropouts += 1
        if now - self.dropout_since > ODOM_HOLD:     # a real stop (a crash): believe it
            self.v_est, self.dropout_since = speed, None
            return speed
        cmd = self._cmd_at(now - self.latency)[1] / self.speed_scale
        kp = self.model.kp_up if cmd > self.v_est else self.model.kp_down
        self.v_est += kp * (cmd - self.v_est) * dt
        return self.v_est

    def step(self, pose, speed, now, age=0.0, yaw_rate=None):
        """pose (x, y, yaw) in the map frame, speed m/s (odometry), now in seconds,
        age: how long before `now` the pose was measured plus the expected solve
        time. yaw_rate: measured (rad/s); None assumes the steady state for the
        steering angle. Returns the (steer, speed) command."""
        if self.calibrate_speed:
            speed = self._filter_speed(speed, now)
            if self.last_time is not None:
                self._update_speed_scale(pose, speed, now - self.last_time)
            speed *= self.speed_scale               # true speed from here on
        if self.last_time is not None:
            dt = now - self.last_time
            # warm start: advance the previous plan by the elapsed time
            t = np.arange(mp.STEPS) * mp.DT + dt
            src = np.arange(mp.STEPS) * mp.DT
            self.u_nom = np.stack([np.interp(t, src, self.u_nom[:, i]) for i in range(2)], 1).astype(np.float32)
            # the steering servo slews at the rate limit toward the command reaching it
            lim = self.model.steer_rate * mp.H_SUB
            n = max(int(round(dt / mp.H_SUB)), 1)
            for cmd in self._wheel_cmds(self.last_time, n)[:, 0]:
                self.steer_est += float(np.clip(cmd - self.steer_est, -lim, lim))
        self.last_time = now
        pose_now = pose
        if yaw_rate is None:
            yaw_rate = max(speed, 0.0) * math.tan(self.steer_est) / veh.WHEELBASE
        pose, speed, steer_wheel, yaw_rate = self.predict(pose, speed, yaw_rate, now, self.latency + age)
        x, y, yaw = pose
        if self.reverse_until is not None:
            if now < self.reverse_until and self.rear_clearance(*pose_now) > REVERSE_CLEARANCE:
                return self._send(now, self.reverse_steer, REVERSE_SPEED)
            self.reverse_until = None
            self.u_nom[:] = 0.0
        u, traj, cost, samples = self._solve(x, y, yaw, max(speed, 0.0), steer_wheel, yaw_rate)
        self.u_nom = np.array(u)   # writable copy (the solver returns read-only arrays)
        self.plan, self.plan_cost = traj, float(cost)
        self.plan_pose, self.samples = pose, samples

        if abs(speed) < STUCK_SPEED and u[: int(0.5 / mp.DT), 1].max() < STUCK_SPEED:
            self.stuck_since = now if self.stuck_since is None else self.stuck_since
            if now - self.stuck_since > STUCK_TIME and self.rear_clearance(*pose_now) > REVERSE_CLEARANCE:
                self.stuck_since = None
                self.recoveries += 1
                # reversing with the wheels turned away from the line swings the nose toward it
                ahead = self.local_reference(x, y, yaw)[0][mp.REF_BACK + 20]
                self.reverse_steer = -float(np.sign(ahead[1]) or 1.0) * self.model.max_steer
                self.reverse_until = now + REVERSE_TIME
                return self._send(now, self.reverse_steer, REVERSE_SPEED)
        else:
            self.stuck_since = None
        return self._send(now, float(u[0, 0]), self._limit_speed(now, float(u[0, 1])))

    def _limit_speed(self, now, speed):
        """The speed sent may rise at most cmd_accel and fall at most cmd_decel per
        second from the last one sent: the planner may swap between a go and a brake
        plan from one tick to the next, which made the car stutter."""
        if not self.sent or self.sent[-1][2] < 0.0:
            return speed
        t_prev, _, v_prev = self.sent[-1]
        dt = min(max(now - t_prev, 0.0), 0.1)
        speed = min(max(speed, v_prev - self.model.cmd_decel * dt), v_prev + self.model.cmd_accel * dt)
        self.u_nom[0, 1] = speed   # the warm start begins from what was sent
        return speed

    def rear_clearance(self, x, y, yaw):
        """Clearance of the rear of the body to scan returns, swept 0.3 m back."""
        grid = self.clearance_grid(x, y, yaw)
        best = mp.FAR
        for back in (0.0, 0.15, 0.3):
            px = veh.CIRCLE_OFFSETS[0] - back
            ix = int((px - mp.GRID_X0) / mp.GRID_RES); iy = int((0.0 - mp.GRID_Y0) / mp.GRID_RES)
            best = min(best, float(grid[iy, ix]) - veh.CIRCLE_RADIUS)
        return best
