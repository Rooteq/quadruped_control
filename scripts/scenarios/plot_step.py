#!/usr/bin/env python3
"""
Plot the velocity-step scenario from a rosbag2 directory.

Five figures:
  1. velocity tracking (vx reference vs actual)
  2. orientation (roll, pitch, yaw)
  3. body z position
  4. desired GRF z-component for all 4 legs (4 subplots stacked)
  5. BL leg actuator torques (3 joints)

Usage:
    python3 plot_step.py path/to/bag_directory
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

from bag_reader import (
    read_bag, common_t0,
    odom_extract, cmd_vel_extract, float_array_extract,
)

ODOM_TOPIC   = '/odom'                                    # ground truth
CMD_TOPIC    = '/cmd_vel'
TAU_TOPIC    = '/forward_effort_controller/commands'      # commanded torques
GRF_TOPIC    = '/mpc/grfs'                                # optional, see README

LEG_NAMES = ['FL', 'FR', 'BL', 'BR']
NOMINAL_Z = 0.27


def main():
    if len(sys.argv) < 2:
        print(f'usage: {sys.argv[0]} <bag_dir>')
        sys.exit(1)
    bag_path = sys.argv[1]
    print(f'Reading {bag_path} …')

    topics = [ODOM_TOPIC, CMD_TOPIC, TAU_TOPIC, GRF_TOPIC]
    data = read_bag(bag_path, topics=topics)
    for t in topics:
        print(f'  {t}: {len(data.get(t, []))} msgs')

    t0 = common_t0(data)
    if t0 is None:
        print('Empty bag.'); sys.exit(1)

    odom  = odom_extract(data.get(ODOM_TOPIC, []),  t0)
    cmd   = cmd_vel_extract(data.get(CMD_TOPIC, []), t0)
    taus  = float_array_extract(data.get(TAU_TOPIC, []), t0)
    grfs  = float_array_extract(data.get(GRF_TOPIC, []), t0)

    figs = []

    # ── Figure 1: velocity tracking ──────────────────────────────────
    fig1, ax_v = plt.subplots(figsize=(10, 4))
    fig1.suptitle('Step response: forward velocity tracking')
    if cmd is not None:
        ax_v.plot(cmd['t'], cmd['vx'], 'r--', label='vx reference')
    if odom is not None:
        ax_v.plot(odom['t'], odom['vel'][:, 0], 'b-', label='vx actual')
    ax_v.set_ylabel('vx [m/s]'); ax_v.set_xlabel('time [s]')
    ax_v.legend(); ax_v.grid(True)
    figs.append(fig1)

    # Figures 2–5 use double-size text (Figure 1 keeps default font).
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
        # ── Figure 2: orientation ───────────────────────────────────────
        fig2, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
        fig2.suptitle('Step response: orientation')
        labels = ['roll', 'pitch', 'yaw']
        if odom is not None:
            for i, ax in enumerate(axes):
                ax.plot(odom['t'], np.degrees(odom['rpy'][:, i]), 'b-')
                ax.set_ylabel(f'{labels[i]} [deg]'); ax.grid(True)
            axes[-1].set_xlabel('time [s]')
        figs.append(fig2)

        # ── Figure 3: body z position ────────────────────────────────────
        fig3, ax = plt.subplots(figsize=(10, 4))
        fig3.suptitle('Step response: body height')
        if odom is not None:
            ax.plot(odom['t'], odom['pos'][:, 2], 'b')
            # ax.axhline(NOMINAL_Z, color='r', linestyle='--', label=f'nominal z = {NOMINAL_Z} m')
        ax.set_ylabel('z [m]'); ax.set_xlabel('time [s]')
        ax.legend(); ax.grid(True)
        figs.append(fig3)

        # ── Figure 4: GRFs_z per leg (stacked) ───────────────────────────
        if grfs is not None and grfs['data'].shape[1] >= 12:
            fig4, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
            fig4.suptitle('Step response: desired GRF z-component per leg')
            for i, ax in enumerate(axes):
                ax.plot(grfs['t'], grfs['data'][:, 3 * i + 2], 'b-')
                ax.set_ylabel(f'{LEG_NAMES[i]} fz [N]'); ax.grid(True)
            axes[-1].set_xlabel('time [s]')
            figs.append(fig4)
        else:
            print(f'  {GRF_TOPIC} not in bag (or wrong shape) — skipping GRF figure.')

        # ── Figure 5: BL leg torques ─────────────────────────────────────
        if taus is not None and taus['data'].shape[1] >= 12:
            fig5, ax = plt.subplots(figsize=(10, 5))
            fig5.suptitle('Step response: BL leg actuator torques')
            # Canonical joint order FL[0:3], FR[3:6], BL[6:9], BR[9:12]
            ax.plot(taus['t'], taus['data'][:, 6], 'r-', linewidth=0.75, label='BL Hip Roll joint')
            ax.plot(taus['t'], taus['data'][:, 7], 'g-', linewidth=0.75, label='BL Hip Pitch joint')
            ax.plot(taus['t'], taus['data'][:, 8], 'b-', linewidth=0.75, label='BL Knee Pitch joint')
            ax.set_xlabel('time [s]'); ax.set_ylabel('torque [Nm]')
            ax.legend(); ax.grid(True)
            figs.append(fig5)
        else:
            print(f'  {TAU_TOPIC} not in bag — skipping BL torques figure.')

    for i, fig in enumerate(figs, 1):
        fig.tight_layout()
        fig.savefig(f'step_fig{i}.png', dpi=120)
    plt.show()


if __name__ == '__main__':
    main()
