#!/usr/bin/env python3
"""How well does the planner's vehicle model predict the real car? Run on car bags.

    python3 model_check.py <bag dir> [<bag dir> ...] [--fit] [--horizon=1.0]

For instants while the car drives (every 0.1 s), the model is started from the
particle-filter pose, the measured speed and yaw rate and the steering the servo
should be at, then driven by the /drive commands the car really executed
(delayed by the latency) for up to 1 s. Its position is compared with the
particle filter's at +0.25, +0.5 and +1.0 s. Errors are split into along-track
(+: the model is ahead of the car, i.e. too fast or too little latency) and
cross-track (+: the model is left of the car: it turns too much left / too little
right) parts, plus heading. Because the real commands are replayed, this tests
the model alone, not the planner.

--fit searches latency, steering rate, the speed loop's kp_up / kp_down and the
tyres' cornering stiffness (tyre_mu) for the values that predict best (coordinate descent over a grid). Fit
on several bags and only adopt values that help on all of them (vehicle.CAR).

A bag recorded with the node running (launch record:=true) also has
/mppi/prediction; its predictions are scored the same way the node's status
line does.
"""
import glob
import math
import os
import sqlite3
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import jax
import jax.numpy as jnp
import numpy as np
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from scan_mppi import planner as mp
from scan_mppi import vehicle as veh

H = mp.H_SUB                 # 0.01 s integration step
HORIZONS = (0.25, 0.5, 1.0)  # s
SAMPLE_EVERY = 0.1           # s between start instants
MIN_SPEED = 0.8              # m/s: start instants slower than this are skipped

TOPICS = {'/pf/viz/inferred_pose': 'pose', '/odom': 'odom', '/drive': 'drive', '/mppi/prediction': 'pred'}


def stamp(h):
    return h.stamp.sec + 1e-9 * h.stamp.nanosec


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def read_bag(path):
    """Poses (header time), odom speed and /drive (receive time), predictions."""
    db = sqlite3.connect('file:%s?immutable=1' % sorted(glob.glob(os.path.join(path, '*.db3')))[0], uri=True)
    topics = {i: (name, typ) for i, name, typ in db.execute('select id, name, type from topics')}
    out = {k: [] for k in TOPICS.values()}
    for tid, ts, raw in db.execute('select topic_id, timestamp, data from messages order by timestamp'):
        name, typ = topics[tid]
        if name not in TOPICS:
            continue
        m = deserialize_message(raw, get_message(typ))
        kind, t = TOPICS[name], ts * 1e-9
        if kind == 'pose':
            out[kind].append((stamp(m.header), m.pose.position.x, m.pose.position.y, yaw_of(m.pose.orientation)))
        elif kind == 'odom':
            out[kind].append((t, m.twist.twist.linear.x))
        elif kind == 'drive':
            out[kind].append((t, m.drive.steering_angle, m.drive.speed))
        else:
            out[kind].extend((stamp(p.header), k, p.pose.position.x, p.pose.position.y, yaw_of(p.pose.orientation))
                             for k, p in enumerate(m.poses))
    return {k: np.array(v) for k, v in out.items()}


class Track:
    """Particle-filter poses as continuous functions of time."""

    def __init__(self, pose):
        self.t, self.x, self.y = pose[:, 0], pose[:, 1], pose[:, 2]
        self.yaw = np.unwrap(pose[:, 3])

    def at(self, t):
        return np.interp(t, self.t, self.x), np.interp(t, self.t, self.y), np.interp(t, self.t, self.yaw)

    def speed(self, t, w=0.1):
        (x0, y0, _), (x1, y1, _) = self.at(t - w / 2), self.at(t + w / 2)
        return np.hypot(x1 - x0, y1 - y0) / w

    def yaw_rate(self, t, w=0.1):
        return (self.at(t + w / 2)[2] - self.at(t - w / 2)[2]) / w


def errors(pred_x, pred_y, pred_yaw, act_x, act_y, act_yaw):
    dx, dy = pred_x - act_x, pred_y - act_y
    along = dx * np.cos(act_yaw) + dy * np.sin(act_yaw)
    cross = -dx * np.sin(act_yaw) + dy * np.cos(act_yaw)
    dyaw = np.degrees(np.angle(np.exp(1j * (pred_yaw - act_yaw))))
    return np.hypot(dx, dy), along, cross, dyaw


def summary(horizon, e, along, cross, dyaw):
    status, hints = veh.grade(horizon, 100 * np.median(e), 100 * along.mean(), 100 * cross.mean())
    return (f'{100 * np.median(e):4.0f} cm (p90 {100 * np.percentile(e, 90):3.0f}) {status:4s}  along {100 * along.mean():+4.0f}  '
            f'cross {100 * cross.mean():+4.0f} cm  yaw {np.median(np.abs(dyaw)):4.1f} deg' + ''.join(f'  [{h}]' for h in hints))


@jax.jit
def simulate(v0, steer0, yaw_rate0, cmds, mu, car):
    """Run the planner's model (planner._inputs, planner._rk4) from the car frame
    under the wheel commands cmds (N, n, 2) at H steps. Returns x, y, yaw (N, n)."""
    def one(v0, steer0, r0, cmds):
        slip0 = jnp.arctan(veh.LR * jnp.tan(steer0) / veh.WHEELBASE)
        s0 = (0.0, 0.0, steer0, v0, 0.0, r0, slip0)
        s0 = tuple(jnp.asarray(v, jnp.float32) for v in s0)

        def step(s, u):
            sv, a = mp._inputs(s, u[0], u[1], car, mu)
            s = mp._rk4(s, sv, a, car[8])
            s = s[:3] + (jnp.maximum(s[3], 0.0),) + s[4:]
            return s, jnp.stack([s[0], s[1], s[4]])

        _, out = jax.lax.scan(step, s0, cmds)
        return out
    return jax.vmap(one)(v0, steer0, yaw_rate0, cmds)


class Bag:
    def __init__(self, path):
        self.name = os.path.basename(path.rstrip('/'))
        d = read_bag(path)
        if len(d['pose']) < 50 or len(d['drive']) < 20 or len(d['odom']) < 20:
            raise ValueError(f'{self.name}: needs /pf/viz/inferred_pose, /odom and /drive')
        self.track, self.odom, self.drive, self.pred = Track(d['pose']), d['odom'], d['drive'], d['pred']
        tr = self.track
        # odometry scale over the moving parts (as the controller estimates it)
        tg = np.arange(tr.t[0] + 0.5, tr.t[-1] - 0.5, 0.05)
        v_pf, v_od = tr.speed(tg), np.interp(tg, self.odom[:, 0], self.odom[:, 1])
        m = (v_od > 1.0) & (np.abs(v_pf / np.maximum(v_od, 1e-3) - 0.85) < 0.35)
        self.scale = float(np.sum(v_pf[m]) / np.sum(v_od[m])) if m.sum() > 20 else 0.85
        # start instants: driving, odometry agrees with the PF (no dropout), PF continuous
        t_end = min(tr.t[-1], self.drive[-1, 0]) - max(HORIZONS) - 0.05
        ts = np.arange(max(tr.t[0], self.drive[0, 0]) + 1.0, t_end, SAMPLE_EVERY)
        v = np.interp(ts, self.odom[:, 0], self.odom[:, 1]) * self.scale
        gap = np.array([np.diff(tr.t[(tr.t > t - 0.1) & (tr.t < t + max(HORIZONS) + 0.1)]).max(initial=0.0) for t in ts])
        ok = (v > MIN_SPEED) & (np.abs(v - tr.speed(ts)) < 0.5) & (gap < 0.1)
        self.ts, self.v0 = ts[ok], v[ok]
        if len(self.ts) < 10:
            e = ValueError(f'{self.name}: only {len(self.ts)} instants of driving above {MIN_SPEED} m/s')
            e.bag = self
            raise e
        self.r0 = tr.yaw_rate(self.ts)
        self.pose0 = tr.at(self.ts)
        self.n = int(round(max(HORIZONS) / H))
        self.actual = [tr.at(self.ts + h) for h in HORIZONS]

    def _cmd(self, t):
        """/drive command in force at times t (array): last one sent at or before t."""
        i = np.clip(np.searchsorted(self.drive[:, 0], t, 'right') - 1, 0, len(self.drive) - 1)
        return self.drive[i, 1], self.drive[i, 2] * self.scale

    def evaluate(self, model, mu):
        lat = model.latency
        k = np.arange(self.n) * H
        steer_cmd, _ = self._cmd(self.ts[:, None] + k[None] - lat - veh.STEER_DELAY)
        _, speed_cmd = self._cmd(self.ts[:, None] + k[None] - lat)
        # steering at the start: the servo slewing toward its commands over the second before
        back = np.arange(-1.0, 0.0, H)
        hist, _ = self._cmd(self.ts[:, None] + back[None] - lat - veh.STEER_DELAY)
        steer = hist[:, 0].copy()
        lim = model.steer_rate * H
        for j in range(1, hist.shape[1]):
            steer += np.clip(hist[:, j] - steer, -lim, lim)
        cmds = np.stack([steer_cmd, speed_cmd], -1).astype(np.float32)
        out = np.asarray(simulate(jnp.asarray(self.v0, jnp.float32), jnp.asarray(steer, jnp.float32),
                                  jnp.asarray(self.r0, jnp.float32), jnp.asarray(cmds),
                                  jnp.float32(mu), jnp.asarray(model.array())))
        x0, y0, yaw0 = self.pose0
        c, s = np.cos(yaw0), np.sin(yaw0)
        res = []
        for h, (ax, ay, ayaw) in zip(HORIZONS, self.actual):
            j = int(round(h / H)) - 1
            lx, ly, lyaw = out[:, j, 0], out[:, j, 1], out[:, j, 2]
            res.append(errors(x0 + c * lx - s * ly, y0 + s * lx + c * ly, yaw0 + lyaw, ax, ay, ayaw))
        return res

    def prediction_report(self):
        """Score /mppi/prediction (the node's own predictions) against the poses."""
        if len(self.pred) == 0:
            return []
        p, tr = self.pred, self.track
        p = p[(p[:, 0] > tr.t[0]) & (p[:, 0] < tr.t[-1])]
        lines = []
        for k, h, label in ((0, veh.CAR.latency, 'start (latency)'), (int(round(0.5 / mp.DT)), 0.5, 'plan +0.5 s'),
                            (int(round(1.0 / mp.DT)), 1.0, 'plan +1.0 s')):
            q = p[(p[:, 1] == k) & (tr.speed(p[:, 0]) > MIN_SPEED)]
            if len(q) < 5:
                continue
            lines.append(f'  {label:16s} ' + summary(h, *errors(q[:, 2], q[:, 3], q[:, 4], *tr.at(q[:, 0]))))
        return lines


def score(bags, model, mu):
    """Fit objective: median position error at 0.5 and 1.0 s, averaged over bags."""
    return float(np.mean([np.mean([np.median(r[0]) for h, r in zip(HORIZONS, b.evaluate(model, mu)) if h >= 0.5])
                          for b in bags]))


def report(bags, model, mu, title):
    print(f'{title}: latency {model.latency:.2f} s, steer_rate {model.steer_rate:.1f} rad/s, '
          f'kp_up {model.kp_up:.1f}, kp_down {model.kp_down:.1f}, tyre_mu {model.tyre_mu:.2f} (friction {mu:.2f})')
    for b in bags:
        print(f'  {b.name} ({len(b.ts)} starts, odom scale {b.scale:.3f})')
        for h, r in zip(HORIZONS, b.evaluate(model, mu)):
            print(f'    +{h:.2f} s  ' + summary(h, *r))


GRID = {'latency': [0.06, 0.08, 0.10, 0.12, 0.14, 0.15, 0.16, 0.18, 0.20, 0.22, 0.25],
        'steer_rate': [1.5, 2.0, 2.5, 3.0, 3.5, 4.5, 6.0],
        'kp_up': [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0],
        'kp_down': [2.0, 3.0, 4.0, 5.0, 6.0, 8.0],
        'tyre_mu': [0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.3]}


def fit(bags, model, mu):
    best = score(bags, model, mu)
    for _ in range(3):
        changed = False
        for name, values in GRID.items():
            for val in values:
                m2 = model._replace(**{name: val})
                s = score(bags, m2, mu)
                if s < best - 1e-4:
                    best, model, changed = s, m2, True
        if not changed:
            break
    return model


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    opts = [a for a in sys.argv[1:] if a.startswith('--')]
    if not args:
        sys.exit(__doc__)
    mu = next((float(o.split('=')[1]) for o in opts if o.startswith('--friction=')), 0.4)   # grip limit, as car.yaml
    bags = []
    for p in args:
        try:
            bags.append(Bag(p))
        except ValueError as e:
            print(f'skipped {e}')
            if getattr(e, 'bag', None) is not None:   # too little driving to fit, but the node's predictions still count
                lines = e.bag.prediction_report()
                if lines:
                    print(f'{e.bag.name}: the node\'s own predictions (/mppi/prediction)')
                    print('\n'.join(lines))
    if not bags:
        sys.exit('no usable bags')
    model = veh.CAR
    report(bags, model, mu, 'current model (vehicle.CAR)')
    for b in bags:
        lines = b.prediction_report()
        if lines:
            print(f'{b.name}: the node\'s own predictions (/mppi/prediction)')
            print('\n'.join(lines))
    if '--fit' in opts:
        m2 = fit(bags, model, mu)
        print(f'\nobjective (median error at 0.5 and 1.0 s): {100 * score(bags, model, mu):.1f} cm now, '
              f'{100 * score(bags, m2, mu):.1f} cm fitted')
        report(bags, m2, mu, 'fitted')
        if len(bags) > 1:
            print('\nper bag, current -> fitted:')
            for b in bags:
                print(f'  {b.name}: {100 * score([b], model, mu):.1f} -> {100 * score([b], m2, mu):.1f} cm')


if __name__ == '__main__':
    main()
