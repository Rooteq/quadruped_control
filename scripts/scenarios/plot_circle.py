#!/usr/bin/env python3
"""
Plot the full-circle scenario from a rosbag2 directory.

Four figures:
  1. xy trajectory — desired (integrated from cmd_vel) vs actual CoM
  2. yaw rate vs time (commanded vs actual)
  3. roll and pitch vs time
  4. body-frame vx — commanded vs actual

Usage:
    python3 plot_circle.py path/to/bag_directory
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

from bag_reader import read_bag, common_t0, odom_extract, cmd_vel_extract

ODOM_TOPIC = '/odom'
CMD_TOPIC  = '/cmd_vel'


def main():
    if len(sys.argv) < 2:
        print(f'usage: {sys.argv[0]} <bag_dir>')
        sys.exit(1)
    bag_path = sys.argv[1]
    print(f'Reading {bag_path} …')

    data = read_bag(bag_path, topics=[ODOM_TOPIC, CMD_TOPIC])
    print(f'  {ODOM_TOPIC}: {len(data.get(ODOM_TOPIC, []))} msgs')
    print(f'  {CMD_TOPIC}:  {len(data.get(CMD_TOPIC, []))} msgs')

    t0 = common_t0(data)
    if t0 is None:
        print('Empty bag.'); sys.exit(1)

    odom = odom_extract(data.get(ODOM_TOPIC, []), t0)
    cmd  = cmd_vel_extract(data.get(CMD_TOPIC, []), t0)
    if odom is None:
        print(f'No {ODOM_TOPIC} in bag.'); sys.exit(1)

    # ── Integrate desired trajectory from cmd_vel (body-frame vx, vy + wz) ────
    x0   = odom['pos'][0, 0]
    y0   = odom['pos'][0, 1]
    yaw0 = odom['rpy'][0, 2]
    x_des = y_des = yaw_des = None
    if cmd is not None:
        dt      = np.diff(cmd['t'], prepend=cmd['t'][0])
        yaw_des = yaw0 + np.cumsum(cmd['wz'] * dt)
        # Body-frame (vx, vy) → world frame via current yaw.
        vx_w = cmd['vx'] * np.cos(yaw_des) - cmd['vy'] * np.sin(yaw_des)
        vy_w = cmd['vx'] * np.sin(yaw_des) + cmd['vy'] * np.cos(yaw_des)
        x_des = x0 + np.cumsum(vx_w * dt)
        y_des = y0 + np.cumsum(vy_w * dt)

    # ── Figure 1: xy trajectory ──────────────────────────────────────
    fig1, ax = plt.subplots(figsize=(7, 7))
    fig1.suptitle('Circle scenario: xy trajectory')
    if x_des is not None:
        ax.plot(x_des, y_des, 'r--', label='desired')
    ax.plot(odom['pos'][:, 0], odom['pos'][:, 1], 'b-', label='actual')
    ax.plot(odom['pos'][0, 0],  odom['pos'][0, 1],  'go', markersize=9, label='start')
    ax.plot(odom['pos'][-1, 0], odom['pos'][-1, 1], 'ro', markersize=9, label='end')
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    ax.legend(); ax.grid(True); ax.set_aspect('equal', adjustable='datalim')

    # Figures 2–4 use double-size text (Figure 1 keeps default font).
    big_font = {
        'font.size'        : 20,
        'axes.titlesize'   : 20,
        'axes.labelsize'   : 20,
        'xtick.labelsize'  : 18,
        'ytick.labelsize'  : 18,
        'legend.fontsize'  : 18,
        'figure.titlesize' : 24,
    }
    with plt.rc_context(big_font):
        # ── Figure 2: yaw rate ───────────────────────────────────────
        fig2, ax = plt.subplots(figsize=(10, 4))
        fig2.suptitle('Circle scenario: yaw-rate tracking')
        if cmd is not None:
            ax.plot(cmd['t'], np.degrees(cmd['wz']), 'r--', label='yaw-rate commanded')
        ax.plot(odom['t'], np.degrees(odom['ang_vel'][:, 2]), 'b-', label='yaw-rate actual')
        ax.set_ylabel('yaw rate [deg/s]'); ax.set_xlabel('time [s]')
        ax.legend(); ax.grid(True)

        # ── Figure 3: roll, pitch ────────────────────────────────────
        fig3, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
        fig3.suptitle('Circle scenario: roll & pitch')
        ax1.plot(odom['t'], np.degrees(odom['rpy'][:, 0]), 'b-')
        ax1.set_ylabel('roll [deg]'); ax1.grid(True)
        ax2.plot(odom['t'], np.degrees(odom['rpy'][:, 1]), 'b-')
        ax2.set_ylabel('pitch [deg]'); ax2.set_xlabel('time [s]'); ax2.grid(True)

        # ── Figure 4: body-frame vx ──────────────────────────────────
        yaw_actual = odom['rpy'][:, 2]
        cos_y      = np.cos(yaw_actual)
        sin_y      = np.sin(yaw_actual)
        vx_body    =  cos_y * odom['vel'][:, 0] + sin_y * odom['vel'][:, 1]

        fig4, ax = plt.subplots(figsize=(10, 4))
        fig4.suptitle('Circle scenario: body-frame vx')
        if cmd is not None:
            ax.plot(cmd['t'], cmd['vx'], 'r--', label='vx body cmd')
        ax.plot(odom['t'], vx_body, 'b-', label='vx body actual')
        ax.set_ylabel('vx body [m/s]'); ax.set_xlabel('time [s]')
        ax.legend(); ax.grid(True)

    for i, fig in enumerate([fig1, fig2, fig3, fig4], 1):
        fig.tight_layout()
        fig.savefig(f'circle_fig{i}.png', dpi=120)
    plt.show()


if __name__ == '__main__':
    main()
