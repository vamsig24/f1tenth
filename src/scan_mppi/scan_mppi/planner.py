"""Sampling MPC (MPPI) in the car's local frame.

Controls are what /drive takes: (steering command, speed command). Each rollout
runs them through the low-level loops the car actually has (steering rate
limit and delay, speed P loop with the motor's acceleration curve) and the
F1TENTH single-track model, the same one f1tenth_gym simulates. Rollouts are
scored on:

  progress   arc length gained along the reference line (the only reward)
  deviation  distance from the reference line
  over speed above the curvature/friction cap of the reference
  grip       total (braking + cornering) acceleration above friction * g,
             and body slip beyond what a real tyre holds
  clearance  footprint distance to LiDAR returns (walls and obstacles alike)
  collision  sticky: every step after first contact is charged

Progress is measured by walking each rollout's reference index forward a
few points per step, never by a global nearest search, so a rollout cannot
earn progress by cutting through a gap to a later part of the lap.

Besides the warm start and smoothed noise, every tick also scores structured
seeds: constant-steer arcs at a few speeds, and closed-loop pure-pursuit
trackers of the reference (and sideways copies of it) at fractions of the
speed cap. The trackers are what let it find fast lines through corners.
The controller predicts the state through the command latency before
calling solve() (see propagate()).
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from . import vehicle as veh

DT = 0.05            # rollout step (s)
STEPS = 30           # 1.5 s: covers braking from 9.5 m/s at A_MAX with room to spare
SAMPLES = 2048       # default; the controller drops to fewer if a solve is too slow (Jetson)
ITERATIONS = 2
NOISE_LAT_ACCEL = 1.5  # m/s^2: steering noise is sized as lateral acceleration, so it shrinks as 1/v^2.
                       # A fixed angle that explores corners at 2 m/s puts every 6 m/s sample into a wall.
NOISE_STEER_MAX = 0.3   # rad, the cap at low speed: getting round something close needs near full lock
NOISE_SPEED = 1.5    # m/s
NOISE_SMOOTH = 0.7   # AR(1) coefficient: correlated noise explores manoeuvres, not jitter
SEED_STEERS = 9
TRACK_SPEED_FRACTIONS = (0.6, 0.75, 0.9, 1.0)
TRACK_OFFSETS = (-0.45, -0.3, -0.15, 0.0, 0.15, 0.3, 0.45)  # m, sideways copies of the line
# Gap path: a line through the gaps between scan returns, found by dynamic
# programming over sideways offsets at stations along the reference, and
# tracked like the shifted copies. Parallel copies cannot weave between
# obstacles on alternate sides; this can.
GAP_SPEED_FRACTIONS = (0.3, 0.45, 0.6, 0.75, 0.9, 1.0)
GAP_STATION = 0.25   # m between stations (less than a box is long)
GAP_STATIONS = 48    # 12 m ahead
GAP_OFFSETS = tuple(round(-1.2 + 0.05 * i, 2) for i in range(49))   # m
GAP_FREE = veh.WIDTH / 2 + 0.07     # m of clearance a station point needs (half the car + the
                                    # planner's contact margin at ~2 m/s)
GAP_MAX_STEP = 0.1   # m of sideways change per station: a ~22 deg heading change, inside the car's turning
SEED_SPEED_STEPS = (0.0, 1.0, 2.5, 5.0)  # m/s above the current speed
REF_BACK = 5         # window points behind the car's projection
REF_POINTS = 200     # window length (20 m at 0.1 m)
SEARCH = 12          # points a rollout's projection may advance per step (1.2 m: 24 m/s)

# Cost weights, in units where 1 = one metre of progress lost.
TEMPERATURE = 0.3    # m: samples more than ~1 m of progress behind the best barely count
W_DEV = 1.0          # per m^2 per s
W_OVERSPEED = 20.0   # per (m/s)^2 per s
W_GRIP = 20.0        # per (m/s^2)^2 per s above the friction circle: grip is a limit, not a trade
W_SLIP = 200.0       # per rad^2 per s of body slip beyond SLIP_MAX
SLIP_MAX = 0.1       # rad. The gym's tyres are linear and never saturate, so without this the
                     # optimiser finds brake-induced oversteer "drifts" that no real car survives.
W_CLEAR = 400.0      # per m^2 per s inside CLEAR_MARGIN
CLEAR_MARGIN = 0.25  # body clearance (m) below which the soft cost starts
W_COLLIDE = 50.0     # per step after first contact
HIT_MARGIN = 0.04    # m of body clearance that already counts as contact, plus
HIT_MARGIN_PER_V = 0.012  # m per m/s: pose, scan-grid and prediction error grow with speed
VIZ_SAMPLES = 64     # rollouts returned for display
W_STEER_JERK = 2.0   # per rad^2 between consecutive steering commands

GRID_RES = 0.05
GRID_X0, GRID_NX = -3.0, 340   # local x in [-3, 14) m
GRID_Y0, GRID_NY = -6.4, 256   # local y in [-6.4, 6.4) m
FAR = 5.0                       # clearance reported off the grid


SUBSTEPS = 5          # RK4 at 0.01 s inside each rollout step, as the gym integrates
H_SUB = DT / SUBSTEPS
DELAY_SUBSTEPS = int(round(veh.STEER_DELAY / H_SUB))


def _inputs(state, steer_cmd, v_cmd, car, mu):
    """The gym's low-level loops (f110_gym pid + constraints): bang-bang steering
    rate toward the command, P control on speed, motor curve and brake limit.
    The tyres cannot pass more than friction * g either way (wheelspin, lock-up)."""
    max_steer, steer_rate, kp_up, kp_down = car[0], car[1], car[2], car[3]
    steer, v = state[2], state[3]
    diff = steer_cmd - steer
    sv = jnp.where(jnp.abs(diff) > 1e-4, jnp.sign(diff) * steer_rate, 0.0)
    sv = jnp.where(((steer <= -max_steer) & (sv <= 0)) | ((steer >= max_steer) & (sv >= 0)), 0.0, sv)
    a = jnp.where(v_cmd > v, kp_up, kp_down) * (v_cmd - v)
    a_motor = veh.A_MAX * jnp.minimum(1.0, veh.V_SWITCH / jnp.maximum(v, 1e-3))
    a_tyre = mu * veh.G
    return sv, jnp.clip(a, -jnp.minimum(veh.A_MAX, a_tyre), jnp.minimum(a_motor, a_tyre))


def _dynamics(s, sv, a, mu):
    """Single-track model of f110_gym (vehicle_dynamics_st), kinematic below 0.5 m/s.
    s = (x, y, steer, v, yaw, yaw_rate, slip)."""
    x, y, d, v, yaw, r, b = s
    lf, lr, L, h, g = veh.LF, veh.LR, veh.WHEELBASE, veh.H_CG, veh.G
    csf, csr, m, I = veh.C_SF, veh.C_SR, veh.MASS, veh.INERTIA
    vs = jnp.where(jnp.abs(v) < 0.5, 0.5, v)  # keep the unused branch finite
    fz_f = g * lr - a * h
    fz_r = g * lf + a * h
    r_dot = (-mu * m / (vs * I * L) * (lf ** 2 * csf * fz_f + lr ** 2 * csr * fz_r) * r
             + mu * m / (I * L) * (lr * csr * fz_r - lf * csf * fz_f) * b
             + mu * m / (I * L) * lf * csf * fz_f * d)
    b_dot = ((mu / (vs ** 2 * L) * (csr * fz_r * lr - csf * fz_f * lf) - 1) * r
             - mu / (vs * L) * (csr * fz_r + csf * fz_f) * b
             + mu / (vs * L) * csf * fz_f * d)
    dyn = (v * jnp.cos(b + yaw), v * jnp.sin(b + yaw), sv, a, r, r_dot, b_dot)
    kin_r_dot = a / L * jnp.tan(d) + v / (L * jnp.cos(d) ** 2) * sv
    kin = (v * jnp.cos(yaw), v * jnp.sin(yaw), sv, a, v / L * jnp.tan(d), kin_r_dot, jnp.zeros_like(v))
    slow = jnp.abs(v) < 0.5
    return tuple(jnp.where(slow, k, dd) for k, dd in zip(kin, dyn))


def _rk4(s, sv, a, mu):
    def add(s, k, c):
        return tuple(si + c * ki for si, ki in zip(s, k))
    k1 = _dynamics(s, sv, a, mu)
    k2 = _dynamics(add(s, k1, H_SUB / 2), sv, a, mu)
    k3 = _dynamics(add(s, k2, H_SUB / 2), sv, a, mu)
    k4 = _dynamics(add(s, k3, H_SUB), sv, a, mu)
    return tuple(si + H_SUB / 6 * (a1 + 2 * a2 + 2 * a3 + a4) for si, a1, a2, a3, a4 in zip(s, k1, k2, k3, k4))


def _clearance(grid, x, y):
    ix = jnp.floor((x - GRID_X0) / GRID_RES).astype(jnp.int32)
    iy = jnp.floor((y - GRID_Y0) / GRID_RES).astype(jnp.int32)
    inside = (ix >= 0) & (ix < GRID_NX) & (iy >= 0) & (iy < GRID_NY)
    val = grid[jnp.clip(iy, 0, GRID_NY - 1), jnp.clip(ix, 0, GRID_NX - 1)]
    return jnp.where(inside, val, FAR)


def rollout_costs(U, v0, steer0, yaw_rate0, steer_prev, ref_xy, ref_v, ref_ds, grid, mu, car):
    """U: (K, T, 2) commands. Returns per-sample cost (K,) and trajectories (K, T, 3)."""
    K = U.shape[0]
    offsets = jnp.arange(-2, SEARCH + 1)
    circle = jnp.asarray(veh.CIRCLE_OFFSETS)

    a_lat = mu * veh.G

    def step(carry, u):
        s, steer_last, idx, hit, cost = carry
        steer_cmd, v_cmd = u[:, 0], u[:, 1]
        v_prev = s[3]
        for j in range(SUBSTEPS):
            # the steering command reaches the wheels STEER_DELAY late
            cmd = steer_last if j < DELAY_SUBSTEPS else steer_cmd
            sv, a = _inputs(s, cmd, v_cmd, car, mu)
            s = _rk4(s, sv, a, mu)
            s = s[:3] + (jnp.maximum(s[3], 0.0),) + s[4:]
        x, y, yaw, v, yaw_rate, slip = s[0], s[1], s[4], s[3], s[5], s[6]
        a_total = jnp.hypot((v - v_prev) / DT, v * yaw_rate)

        cand = jnp.clip(idx[:, None] + offsets[None], 0, REF_POINTS - 1)
        d2 = jnp.sum((ref_xy[cand] - jnp.stack([x, y], -1)[:, None]) ** 2, -1)
        k = jnp.argmin(d2, axis=1)
        idx = jnp.take_along_axis(cand, k[:, None], 1)[:, 0]
        dev2 = jnp.take_along_axis(d2, k[:, None], 1)[:, 0]

        cx = x[:, None] + jnp.cos(yaw)[:, None] * circle[None]
        cy = y[:, None] + jnp.sin(yaw)[:, None] * circle[None]
        clear = jnp.min(_clearance(grid, cx, cy), axis=1) - veh.CIRCLE_RADIUS
        hit = hit | (clear < HIT_MARGIN + HIT_MARGIN_PER_V * v)

        cost = cost + DT * (
            W_DEV * dev2
            + W_OVERSPEED * jnp.maximum(v - ref_v[idx], 0.0) ** 2
            + W_GRIP * jnp.maximum(a_total - a_lat, 0.0) ** 2
            + W_SLIP * jnp.maximum(jnp.abs(slip) - SLIP_MAX, 0.0) ** 2
            + W_CLEAR * jnp.maximum(CLEAR_MARGIN - clear, 0.0) ** 2
        ) + W_COLLIDE * hit
        return (s, steer_cmd, idx, hit, cost), jnp.stack([x, y, v], -1)

    zeros = jnp.zeros(K)
    # Yaw rate is measured (odometry); body slip is not, so it starts at its
    # steady-state value for the steering angle.
    slip0 = jnp.arctan(veh.LR * jnp.tan(steer0) / veh.WHEELBASE)
    s0 = (zeros, zeros, jnp.full(K, steer0), jnp.full(K, v0), zeros,
          jnp.full(K, yaw_rate0), jnp.full(K, slip0))
    init = (s0, jnp.full(K, steer_prev), jnp.full(K, REF_BACK, jnp.int32), jnp.zeros(K, bool), zeros)
    (s, _, idx, hit, cost), traj = jax.lax.scan(step, init, jnp.swapaxes(U, 0, 1))
    x, y = s[0], s[1]
    # fractional progress: project the end point onto the segment after idx
    i1 = jnp.minimum(idx + 1, REF_POINTS - 1)
    tang = ref_xy[i1] - ref_xy[idx]
    tang = tang / jnp.maximum(jnp.linalg.norm(tang, axis=-1, keepdims=True), 1e-6)
    along = jnp.sum((jnp.stack([x, y], -1) - ref_xy[idx]) * tang, -1)
    progress = (idx - REF_BACK) * ref_ds + along
    jerk = jnp.sum(jnp.diff(U[:, :, 0], axis=1) ** 2, axis=1)
    return cost - progress + W_STEER_JERK * jerk, jnp.swapaxes(traj, 0, 1)


def _ref_normals(ref_xy):
    tang = jnp.roll(ref_xy, -1, axis=0) - ref_xy
    tang = tang.at[-1].set(tang[-2])
    tang = tang / jnp.maximum(jnp.linalg.norm(tang, axis=-1, keepdims=True), 1e-6)
    return jnp.stack([-tang[:, 1], tang[:, 0]], -1)


def gap_path(ref_xy, grid):
    """Sideways offset (REF_POINTS,) of a smooth line through the gaps in the
    scan: at each station along the reference, pick one of GAP_OFFSETS so the
    point is at least GAP_FREE from every return, preferring offsets near the
    line, clearance, and small changes between stations (dynamic programming)."""
    normal = _ref_normals(ref_xy)
    offs = jnp.asarray(GAP_OFFSETS)
    step = int(round(GAP_STATION / 0.1))       # reference points are ~0.1 m apart
    idx = jnp.clip(REF_BACK + step * jnp.arange(GAP_STATIONS), 0, REF_POINTS - 1)
    pts = ref_xy[idx][:, None] + offs[None, :, None] * normal[idx][:, None]     # (S, O, 2)
    clr = _clearance(grid, pts[..., 0], pts[..., 1])
    big = 1e4
    node = 0.2 * offs[None] ** 2 + 5.0 * jnp.maximum(0.5 - clr, 0.0) ** 2 + jnp.where(clr < GAP_FREE, big, 0.0)
    # start at the car's own sideways offset from the line
    off0 = -jnp.dot(ref_xy[REF_BACK], normal[REF_BACK])
    d = offs[:, None] - offs[None, :]                                          # (to, from)
    edge = 2.0 * d ** 2 + jnp.where(jnp.abs(d) > GAP_MAX_STEP + 1e-6, big, 0.0)
    first = node[0] + 10.0 * (offs - off0) ** 2 + jnp.where(jnp.abs(offs - off0) > 0.15, big, 0.0)

    def fwd(acc, nd):
        tot = acc[None, :] + edge                                              # (to, from)
        parent = jnp.argmin(tot, axis=1)
        return jnp.min(tot, axis=1) + nd, parent

    last, parents = jax.lax.scan(fwd, first, node[1:])

    def back(k, par):
        return par[k], k

    k_end = jnp.argmin(last)
    _, ks = jax.lax.scan(back, k_end, parents, reverse=True)
    chosen = offs[jnp.concatenate([ks, k_end[None]])]                          # (S,)
    return jnp.interp(jnp.arange(REF_POINTS), idx, chosen)


def _pursuit(offset, frac, v0, steer0, yaw_rate0, steer_prev, ref_xy, ref_v, ref_ds, mu, car, max_speed):
    """Closed-loop pure pursuit of the reference shifted sideways by offset
    (n, REF_POINTS), at frac (n,) of its speed cap, run through the same
    dynamics. Returns the command sequences (n, T, 2)."""
    n = frac.shape[0]
    normal = _ref_normals(ref_xy)
    offsets = jnp.arange(-2, SEARCH + 1)
    rows = jnp.arange(n)

    def step(carry, _):
        s, steer_last, idx = carry
        x, y, v, yaw = s[0], s[1], s[3], s[4]
        cand = jnp.clip(idx[:, None] + offsets[None], 0, REF_POINTS - 1)
        d2 = jnp.sum((ref_xy[cand] - jnp.stack([x, y], -1)[:, None]) ** 2, -1)
        idx = jnp.take_along_axis(cand, jnp.argmin(d2, 1)[:, None], 1)[:, 0]
        look = jnp.maximum(0.6, 0.25 * v)
        j = jnp.clip(idx + (look / ref_ds).astype(jnp.int32), 0, REF_POINTS - 1)
        goal = ref_xy[j] + offset[rows, j][:, None] * normal[j]
        gx, gy = goal[:, 0] - x, goal[:, 1] - y
        lx = jnp.cos(yaw) * gx + jnp.sin(yaw) * gy
        ly = -jnp.sin(yaw) * gx + jnp.cos(yaw) * gy
        steer_cmd = jnp.clip(jnp.arctan2(2 * veh.WHEELBASE * ly, lx * lx + ly * ly), -car[0], car[0])
        jv = jnp.clip(idx + (0.3 * v / ref_ds).astype(jnp.int32), 0, REF_POINTS - 1)
        v_cmd = jnp.minimum(frac * ref_v[jv], max_speed)
        for k in range(SUBSTEPS):
            cmd = steer_last if k < DELAY_SUBSTEPS else steer_cmd
            sv, a = _inputs(s, cmd, v_cmd, car, mu)
            s = _rk4(s, sv, a, mu)
            s = s[:3] + (jnp.maximum(s[3], 0.0),) + s[4:]
        return (s, steer_cmd, idx), jnp.stack([steer_cmd, v_cmd], -1)

    zeros = jnp.zeros(n)
    s0 = (zeros, zeros, jnp.full(n, steer0), jnp.full(n, v0), zeros,
          jnp.full(n, yaw_rate0), jnp.full(n, jnp.arctan(veh.LR * jnp.tan(steer0) / veh.WHEELBASE)))
    _, U = jax.lax.scan(step, (s0, jnp.full(n, steer_prev), jnp.full(n, REF_BACK, jnp.int32)), None, length=STEPS)
    return jnp.swapaxes(U, 0, 1)


def tracking_seeds(v0, steer0, yaw_rate0, steer_prev, ref_xy, ref_v, ref_ds, grid, mu, car, max_speed):
    """Pure-pursuit seeds: sideways copies of the reference at TRACK_SPEED_FRACTIONS
    of its speed cap, and the gap path at GAP_SPEED_FRACTIONS. Returns (n, T, 2);
    they are scored like any other sample."""
    frac, off = jnp.meshgrid(jnp.asarray(TRACK_SPEED_FRACTIONS), jnp.asarray(TRACK_OFFSETS), indexing='ij')
    const = jnp.broadcast_to(off.ravel()[:, None], (off.size, REF_POINTS))
    gap_frac = jnp.asarray(GAP_SPEED_FRACTIONS)
    gap = jnp.broadcast_to(gap_path(ref_xy, grid)[None], (gap_frac.size, REF_POINTS))
    return _pursuit(jnp.concatenate([const, gap]), jnp.concatenate([frac.ravel(), gap_frac]),
                    v0, steer0, yaw_rate0, steer_prev, ref_xy, ref_v, ref_ds, mu, car, max_speed)


@partial(jax.jit, static_argnames=('max_speed', 'samples'))
def solve(key, u_nom, v0, steer0, yaw_rate0, steer_prev, ref_xy, ref_v, ref_ds, grid, mu, car, max_speed, samples=SAMPLES):
    """One control tick: ITERATIONS rounds of MPPI around u_nom (T, 2), `samples` rollouts each.
    car: vehicle.Model.array() (steering limit and rate, speed-loop gains)."""
    v_plan = jnp.maximum(jnp.maximum(u_nom[:, 1], v0), 1.0)
    sigma_steer = jnp.minimum(NOISE_LAT_ACCEL * veh.WHEELBASE / v_plan ** 2, NOISE_STEER_MAX)
    sigma = jnp.stack([sigma_steer, jnp.full(STEPS, NOISE_SPEED)], -1)  # (T, 2)
    lo = jnp.stack([-car[0], 0.0])
    hi = jnp.stack([car[0], max_speed + car[4]])  # the gym's P loop needs headroom to reach max_speed

    # Structured seeds: constant-steer arcs held for 0.5 s then straightened, at a
    # few speeds. Steering and speed noise are independent, so from a slow plan
    # random samples almost never find "faster AND steering differently"; these do.
    steer_lv = jnp.linspace(-1.0, 1.0, SEED_STEERS) * car[0]
    speed_lv = jnp.minimum(v0 + jnp.array(SEED_SPEED_STEPS), max_speed + car[4])
    fade = jnp.clip((20 - jnp.arange(STEPS)) / 10.0, 0.0, 1.0)  # 1 for 0.5 s, 0 after 1 s
    seeds = jnp.stack(jnp.broadcast_arrays(
        steer_lv[:, None, None] * fade[None, None, :],
        speed_lv[None, :, None] + 0 * fade[None, None, :]), -1).reshape(-1, STEPS, 2)
    seeds = jnp.concatenate([seeds, tracking_seeds(v0, steer0, yaw_rate0, steer_prev, ref_xy, ref_v, ref_ds, grid, mu, car, max_speed)])

    def smooth(e):
        def f(prev, n):
            out = NOISE_SMOOTH * prev + np.sqrt(1 - NOISE_SMOOTH ** 2) * n
            return out, out
        _, out = jax.lax.scan(f, e[:, 0], jnp.swapaxes(e, 0, 1))
        return jnp.swapaxes(out, 0, 1)

    for _ in range(ITERATIONS):
        key, sub = jax.random.split(key)
        noise = smooth(jax.random.normal(sub, (samples, STEPS, 2))) * sigma
        U = jnp.clip(u_nom[None] + noise, lo, hi)
        U = U.at[0].set(u_nom)                                   # keep the warm start
        U = U.at[1].set(u_nom.at[:, 1].set(0.0))                 # and a hard stop on the same steering
        U = jnp.concatenate([U, seeds])                          # seeds on top of the random samples
        costs, trajs = rollout_costs(U, v0, steer0, yaw_rate0, steer_prev, ref_xy, ref_v, ref_ds, grid, mu, car)
        best = jnp.min(costs)
        w = jnp.exp(-(costs - best) / TEMPERATURE)
        w = w / jnp.sum(w)
        u_nom = jnp.einsum('k,ktc->tc', w, U)
    # Score the averaged plan itself for the visualisation and the stats.
    c_opt, traj_opt = rollout_costs(u_nom[None], v0, steer0, yaw_rate0, steer_prev, ref_xy, ref_v, ref_ds, grid, mu, car)
    # A spread of the last iteration's rollouts, for display only.
    return u_nom, traj_opt[0], c_opt[0], trajs[::samples // VIZ_SAMPLES]


PREDICT_MAX = 30      # substeps (0.3 s): the longest latency propagate() covers


@jax.jit
def propagate(v0, steer0, yaw_rate0, cmds, n, mu, car):
    """Run the model n substeps (H_SUB each) from the car frame under cmds
    (PREDICT_MAX, 2), the (steering, speed) commands in effect at the wheels
    for each substep. Returns (x, y, yaw, steer, v, yaw_rate) at the end."""
    slip0 = jnp.arctan(veh.LR * jnp.tan(steer0) / veh.WHEELBASE)
    s0 = (0.0, 0.0, steer0, v0, 0.0, yaw_rate0, slip0)
    s0 = tuple(jnp.asarray(v, jnp.float32) for v in s0)

    def step(s, x):
        k, u = x
        sv, a = _inputs(s, u[0], u[1], car, mu)
        nxt = _rk4(s, sv, a, mu)
        nxt = nxt[:3] + (jnp.maximum(nxt[3], 0.0),) + nxt[4:]
        return tuple(jnp.where(k < n, b, a_) for b, a_ in zip(nxt, s)), None

    s, _ = jax.lax.scan(step, s0, (jnp.arange(PREDICT_MAX), cmds))
    return s[0], s[1], s[4], s[2], s[3], s[5]
