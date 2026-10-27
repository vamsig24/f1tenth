"""Reference line: plain (x, y) waypoints -> resampled line with a speed cap.

The CSV only has to carry x and y in its first two columns (any header lines,
',' or ';' separated, so pure pursuit and Levine raceline files both load).
The speed cap comes from the line's curvature and the friction coefficient,
then forward/backward passes respect the motor and braking limits. It only
caps the planner: the rollouts themselves enforce lateral grip, so a line
that cuts inside the reference can go faster than the cap suggests locally.
"""
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.ndimage import uniform_filter1d

from . import vehicle as veh

DS = 0.1  # resampled spacing (m)
CORNER_FRACTION = 0.9  # of friction * g used for steady cornering; the rest absorbs tracking error
BRAKE_FRACTION = 0.6   # of friction * g used to brake for corners. Braking at the limit unloads the
                       # rear axle and the car snaps sideways (oversteer), so plan to arrive slow
                       # enough on firm braking; the planner keeps the full limit for emergencies.


def load_xy(path):
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.replace(';', ',').split(',')
            try:
                rows.append((float(parts[0]), float(parts[1])))
            except (ValueError, IndexError):
                continue  # header or comment
    xy = np.asarray(rows, dtype=float)
    if len(xy) < 4:
        raise ValueError(f'{path}: need at least 4 waypoints, found {len(xy)}')
    return xy


class Reference:
    def __init__(self, path, friction, max_speed):
        xy = load_xy(path)
        keep = np.concatenate([[True], np.linalg.norm(np.diff(xy, axis=0), axis=1) > 1e-3])
        xy = xy[keep]
        spacing = np.median(np.linalg.norm(np.diff(xy, axis=0), axis=1))
        close = max(1.0, 5 * spacing)
        # Recorded laps often run on past the start; cut at the point in the
        # second half that comes back closest to the start.
        half = len(xy) // 2
        back = np.linalg.norm(xy[half:] - xy[0], axis=1)
        j = half + int(np.argmin(back))
        if back.min() < close and j < len(xy) - 1:
            xy = xy[:j + 1]
        gap = np.linalg.norm(xy[0] - xy[-1])
        self.closed = gap < close
        if self.closed and gap < 1e-3:
            xy = xy[:-1]

        # Light smoothing: hand-drawn or recorded lines carry centimetre noise
        # that would otherwise show up as curvature spikes.
        tck, _ = splprep([xy[:, 0], xy[:, 1]], s=len(xy) * 1e-4, per=self.closed)
        u = np.linspace(0, 1, 20 * len(xy), endpoint=not self.closed)
        dense = np.array(splev(u, tck)).T
        seg = np.linalg.norm(np.diff(dense, axis=0), axis=1)
        s_dense = np.concatenate([[0.0], np.cumsum(seg)])
        if self.closed:
            s_dense = np.append(s_dense, s_dense[-1] + np.linalg.norm(dense[0] - dense[-1]))
            u = np.append(u, 1.0)
        self.length = s_dense[-1]
        n = int(self.length / DS)
        u_new = np.interp(np.arange(n) * self.length / n, s_dense, u)
        self.xy = np.array(splev(u_new, tck)).T
        d1 = np.array(splev(u_new, tck, der=1)).T
        d2 = np.array(splev(u_new, tck, der=2)).T
        kappa = (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / np.linalg.norm(d1, axis=1) ** 3
        self.kappa = uniform_filter1d(kappa, size=5, mode='wrap' if self.closed else 'nearest')
        self.ds = self.length / n
        self.n = n
        self.v = self._speed_cap(friction, max_speed)

    def _speed_cap(self, friction, max_speed):
        """Fastest speed along the line inside the same friction circle the
        planner enforces: corner at CORNER_FRACTION of the grip, and brake or
        accelerate only with what the corner leaves (plus the motor curve)."""
        a_grip = friction * veh.G
        k = np.maximum(np.abs(self.kappa), 1e-4)
        v = np.minimum(max_speed, np.sqrt(CORNER_FRACTION * a_grip / k))
        n, ds = self.n, self.ds
        if not self.closed:
            v[-1] = 0.0

        def long_limit(vi, ki):
            lat = min(vi * vi * ki, a_grip)
            return np.sqrt(a_grip ** 2 - lat ** 2)

        passes = 3 if self.closed else 1
        for _ in range(passes):
            for i in range(n - 1 if not self.closed else n):
                j = (i + 1) % n
                acc = min(long_limit(v[i], k[i]), veh.A_MAX * min(1.0, veh.V_SWITCH / max(v[i], 1e-3)))
                v[j] = min(v[j], np.sqrt(v[i] ** 2 + 2 * acc * ds))
            for i in range(n - 2 if not self.closed else n - 1, -1, -1):
                j = (i + 1) % n
                dec = min(long_limit(v[j], k[j]), veh.A_MAX, BRAKE_FRACTION * a_grip)
                v[i] = min(v[i], np.sqrt(v[j] ** 2 + 2 * dec * ds))
        return v

    def nearest(self, p, hint=None, window=3.0):
        """Index of the closest point; searches near `hint` so the projection
        cannot jump to a parallel stretch of the line on the far side of a wall."""
        if hint is None:
            return int(np.argmin(np.linalg.norm(self.xy - p, axis=1)))
        w = int(window / self.ds)
        idx = np.arange(hint - w, hint + w + 1)
        idx = idx % self.n if self.closed else np.clip(idx, 0, self.n - 1)
        return int(idx[np.argmin(np.linalg.norm(self.xy[idx] - p, axis=1))])

    def window(self, start, count):
        """`count` consecutive points from `start` (wrapping on a closed line)."""
        idx = np.arange(start, start + count)
        idx = idx % self.n if self.closed else np.clip(idx, 0, self.n - 1)
        return self.xy[idx], self.v[idx]
