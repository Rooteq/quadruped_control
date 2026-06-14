#!/usr/bin/env python3
"""
Stand-and-hold scenario: publishes zero cmd_vel for TOTAL seconds.

Used as a baseline: verifies the controller is stable at rest, GRFs
converge to gravity compensation (≈ m·g / 4 per stance leg), and there
is no orientation/position drift.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


TOTAL      = 10.0
PUBLISH_HZ = 50


class StandAndHold(Node):
    def __init__(self):
        super().__init__('stand_and_hold')
        self.pub_   = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timer_ = self.create_timer(1.0 / PUBLISH_HZ, self.tick)
        self.elapsed_ = 0.0
        self.done_    = False
        self.get_logger().info(f'Stand-and-hold for {TOTAL} s.')

    def tick(self):
        if self.done_:
            return
        self.elapsed_ += 1.0 / PUBLISH_HZ
        self.pub_.publish(Twist())   # zero command
        if self.elapsed_ >= TOTAL:
            self.done_ = True
            self.get_logger().info('Stand-and-hold complete.')


def main():
    rclpy.init()
    node = StandAndHold()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
