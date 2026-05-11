#!/usr/bin/env python3
"""
Standalone logger: run with  python3 logger.py
Subscribes to /odom and plots live:
  - 2D world XY position (origin-zeroed)
  - World-Z rotation (yaw)
  - Body-X rotation (roll)
"""

import math
import threading

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry


def quat_to_euler(x, y, z, w):
    """Returns (roll, pitch, yaw) from quaternion."""
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr, cosr)

    sinp = 2.0 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)

    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (z * z + y * y)
    yaw = math.atan2(siny, cosy)

    return roll, pitch, yaw


class OdomLogger(Node):
    def __init__(self, data):
        super().__init__("odom_logger")
        self._data = data
        self._origin = None
        self.create_subscription(Odometry, "/odom", self._cb, 10)

    def _cb(self, msg: Odometry):
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        roll, _, yaw = quat_to_euler(q.x, q.y, q.z, q.w)

        if self._origin is None:
            self._origin = (px, py)

        ox, oy = self._origin
        self._data["x"].append(px - ox)
        self._data["y"].append(py - oy)
        MAX = 2_500
        self._data["yaw"].append(math.degrees(yaw))
        self._data["roll"].append(math.degrees(roll))
        if len(self._data["yaw"]) > MAX:
            self._data["yaw"].pop(0)
            self._data["roll"].pop(0)


def spin_ros(node):
    rclpy.spin(node)


def main():
    rclpy.init()

    data = {"x": [], "y": [], "yaw": [], "roll": []}

    node = OdomLogger(data)
    thread = threading.Thread(target=spin_ros, args=(node,), daemon=True)
    thread.start()

    fig = plt.figure(figsize=(12, 5))
    fig.suptitle("Quadruped Odometry Logger", fontsize=13)

    ax_xy  = fig.add_subplot(1, 3, 1, aspect="equal")
    ax_yaw = fig.add_subplot(1, 3, 2)
    ax_roll = fig.add_subplot(1, 3, 3)

    ax_xy.set_title("XY Position (world)")
    ax_xy.set_xlabel("x [m]")
    ax_xy.set_ylabel("y [m]")
    ax_xy.grid(True)

    ax_yaw.set_title("Yaw (world Z rotation)")
    ax_yaw.set_xlabel("sample")
    ax_yaw.set_ylabel("yaw [deg]")
    ax_yaw.grid(True)

    ax_roll.set_title("Roll (body X rotation)")
    ax_roll.set_xlabel("sample")
    ax_roll.set_ylabel("roll [deg]")
    ax_roll.grid(True)

    line_xy,  = ax_xy.plot([], [], "b-", lw=1)
    dot_xy,   = ax_xy.plot([], [], "ro", ms=5)
    line_yaw, = ax_yaw.plot([], [], "g-", lw=1)
    line_roll,= ax_roll.plot([], [], "m-", lw=1)

    def update(_):
        xs = data["x"]
        ys = data["y"]
        yaws = data["yaw"]
        rolls = data["roll"]

        if not xs:
            return line_xy, dot_xy, line_yaw, line_roll

        line_xy.set_data(xs, ys)
        dot_xy.set_data([xs[-1]], [ys[-1]])

        margin = 0.5
        ax_xy.set_xlim(min(xs) - margin, max(xs) + margin)
        ax_xy.set_ylim(min(ys) - margin, max(ys) + margin)

        idx = range(len(yaws))
        line_yaw.set_data(idx, yaws)
        ax_yaw.set_xlim(0, max(1, len(yaws)))
        ax_yaw.set_ylim(min(yaws) - 5, max(yaws) + 5)

        line_roll.set_data(range(len(rolls)), rolls)
        ax_roll.set_xlim(0, max(1, len(rolls)))
        ax_roll.set_ylim(min(rolls) - 5, max(rolls) + 5)

        return line_xy, dot_xy, line_yaw, line_roll

    ani = animation.FuncAnimation(fig, update, interval=100, blit=False, cache_frame_data=False)
    _ = ani  # keep reference so GC doesn't collect the animation

    plt.tight_layout()
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
