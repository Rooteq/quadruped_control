#!/usr/bin/env python3
"""
Plot the stand-and-hold scenario from a rosbag2 directory.

Three figures:
  1. roll & pitch over time
  2. body z position (should hold steady at nominal)
  3. desired GRF z-component per leg (should converge to ≈ m·g / 4 per stance leg)

Usage:
    python3 plot_stand.py path/to/bag_directory
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

from bag_reader import (
    read_bag, common_t0,
    odom_extract, float_array_extract,
)

ODOM_TOPIC = '/odom'
GRF_TOPIC  = '/mpc/grfs'

LEG_NAMES = ['FL', 'FR', 'BL', 'BR']
NOMINAL_Z = 0.27


def main():
    if len(sys.argv) < 2:
        print(f'usage: {sys.argv[0]} <bag_dir>')
        sys.exit(1)
    bag_path = sys.argv[1]
    print(f'Reading {bag_path} …')

    data = read_bag(bag_path, topics=[ODOM_TOPIC, GRF_TOPIC])
    print(f'  {ODOM_TOPIC}: {len(data.get(ODOM_TOPIC, []))} msgs')
    print(f'  {GRF_TOPIC}:  {len(data.get(GRF_TOPIC, []))} msgs')

    t0 = common_t0(data)
    if t0 is None:
        print('Empty bag.'); sys.exit(1)

    odom = odom_extract(data.get(ODOM_TOPIC, []), t0)
    grfs = float_array_extract(data.get(GRF_TOPIC, []), t0)

    figs = []

    # ── Figure 1: roll, pitch ───────────────────────────────────────
    fig1, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    fig1.suptitle('Stand-and-hold: orientation')
    if odom is not None:
        axes[0].plot(odom['t'], np.degrees(odom['rpy'][:, 0]), 'b-')
        axes[0].set_ylabel('roll [deg]');  axes[0].grid(True)
        axes[1].plot(odom['t'], np.degrees(odom['rpy'][:, 1]), 'b-')
        axes[1].set_ylabel('pitch [deg]'); axes[1].set_xlabel('time [s]')
        axes[1].grid(True)
    figs.append(fig1)

    # ── Figure 2: body z position ───────────────────────────────────
    fig2, ax = plt.subplots(figsize=(10, 4))
    fig2.suptitle('Stand-and-hold: body height')
    if odom is not None:
        ax.plot(odom['t'], odom['pos'][:, 2], 'b-', label='actual z')
        ax.axhline(NOMINAL_Z, color='r', linestyle='--', label=f'nominal z = {NOMINAL_Z} m')
    ax.set_ylabel('z [m]'); ax.set_xlabel('time [s]')
    ax.legend(); ax.grid(True)
    figs.append(fig2)

    # ── Figure 3: GRFs_z per leg ────────────────────────────────────
    if grfs is not None and grfs['data'].shape[1] >= 12:
        fig3, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
        fig3.suptitle('Stand-and-hold: desired GRF z-component per leg')
        for i, ax in enumerate(axes):
            ax.plot(grfs['t'], grfs['data'][:, 3 * i + 2], 'b-')
            ax.set_ylabel(f'{LEG_NAMES[i]} fz [N]'); ax.grid(True)
        axes[-1].set_xlabel('time [s]')
        figs.append(fig3)
    else:
        print(f'  {GRF_TOPIC} not in bag (or wrong shape) — skipping GRF figure.')

    for i, fig in enumerate(figs, 1):
        fig.tight_layout()
        fig.savefig(f'stand_fig{i}.png', dpi=120)
    plt.show()


if __name__ == '__main__':
    main()
