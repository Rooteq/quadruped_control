#!/usr/bin/env python3
"""
Publish cmd_vel to walk a quarter circle of radius 4 m at 0.5 m/s.

  R = 4 m,  v = 0.5 m/s  →  ω = v/R = 0.125 rad/s
  arc = π·R/2 ≈ 12.57 m  →  duration ≈ 25.13 s

The robot's heading stays tangent to the circle throughout.
"""

import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


LINEAR_VEL   = 0.5          # m/s forward
RADIUS       = 4.0          # m
ANGULAR_VEL  = LINEAR_VEL / RADIUS   # = 0.125 rad/s
DURATION     = (math.pi / 2 * RADIUS) / LINEAR_VEL   # quarter-circle time ≈ 12.57 s
PUBLISH_HZ   = 20


class QuarterCircleWalker(Node):
    def __init__(self):
        super().__init__('quarter_circle_walker')
        self.pub_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timer_ = self.create_timer(1.0 / PUBLISH_HZ, self.tick)
        self.elapsed_ = 0.0
        self.done_ = False
        self.get_logger().info(
            f'Walking quarter circle: R={RADIUS} m, v={LINEAR_VEL} m/s, '
            f'ω={ANGULAR_VEL:.4f} rad/s, duration={DURATION:.2f} s'
        )

    def tick(self):
        if self.done_:
            return

        self.elapsed_ += 1.0 / PUBLISH_HZ

        if self.elapsed_ >= DURATION:
            self._stop()
            self.done_ = True
            self.get_logger().info('Quarter circle complete — stopping.')
            return

        msg = Twist()
        msg.linear.x  = LINEAR_VEL
        msg.angular.z = ANGULAR_VEL
        self.pub_.publish(msg)

    def _stop(self):
        self.pub_.publish(Twist())   # zero twist


def main():
    rclpy.init()
    node = QuarterCircleWalker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
