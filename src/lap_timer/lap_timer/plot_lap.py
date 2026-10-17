#!/usr/bin/env python3
"""Plot lap logs written by lap_timer.

    plot_lap.py LOG.csv [LOG2.csv ...] [--labels vanilla improved] [--out laps.png]
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from PIL import Image

MAX_TRACKING_ERROR = 0.5


def share_dir():
    try:
        from ament_index_python.packages import get_package_share_directory
        return get_package_share_directory('lap_timer')
    except Exception:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


def load_map(name='levine'):
    d = os.path.join(share_dir(), 'maps')
    meta = {}
    with open(os.path.join(d, name + '.yaml')) as f:
        for line in f:
            if ':' in line:
                k, v = line.split(':', 1)
                meta[k.strip()] = v.strip()
    res = float(meta['resolution'])
    ox, oy = [float(x) for x in meta['origin'].strip('[]').split(',')[:2]]
    img = np.array(Image.open(os.path.join(d, meta['image'])).convert('L'))
    h, w = img.shape
    return img, [ox, ox + w * res, oy, oy + h * res]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('logs', nargs='+')
    ap.add_argument('--labels', nargs='+')
    ap.add_argument('--out', default='laps.png')
    args = ap.parse_args()
    labels = args.labels or [os.path.splitext(os.path.basename(p))[0] for p in args.logs]
    runs = [np.genfromtxt(p, delimiter=',', names=True) for p in args.logs]

    img, extent = load_map()
    wp = np.loadtxt(os.path.join(share_dir(), 'waypoints', 'levine.csv'), delimiter=',', skiprows=1)
    wp = np.vstack([wp, wp[:1]])
    n = len(runs)
    fig = plt.figure(figsize=(7 * max(n, 2), 10))
    gs = GridSpec(2, 2 * n, height_ratios=[1.2, 1], figure=fig)

    for i, (r, label) in enumerate(zip(runs, labels)):
        ax = fig.add_subplot(gs[0, 2 * i:2 * i + 2])
        ax.imshow(img, cmap='gray', extent=extent)
        ax.plot(wp[:, 0], wp[:, 1], 'k--', lw=1, label='waypoints')
        ax.plot(r['x'], r['y'], color=f'C{i}', lw=2, label='driven path')
        ax.legend(loc='center', fontsize=9)
        ax.set_xlim(r['x'].min() - 2, r['x'].max() + 2)
        ax.set_ylim(r['y'].min() - 2, r['y'].max() + 2)
        ax.set_aspect('equal')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_title(f'{label}: {r["t"][-1]:.2f} s, max error {r["tracking_error"].max():.2f} m')

    ax_e = fig.add_subplot(gs[1, :])
    for i, (r, label) in enumerate(zip(runs, labels)):
        ax_e.plot(r['t'], r['tracking_error'], color=f'C{i}', label=label)
    ax_e.axhline(MAX_TRACKING_ERROR, color='r', ls='--', label='limit')
    ax_e.set(xlabel='time [s]', ylabel='tracking error [m]', title='Tracking error')
    ax_e.grid(True)
    ax_e.legend()

    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f'Saved {args.out}')


if __name__ == '__main__':
    main()
