#!/usr/bin/env python3
"""
Full-circle scenario.

  Body-frame command:  vx = V_X,  wz = W_Z
  Geometric radius  :  R  = V_X / W_Z   (defaults to 1 m)
  Period            :  T  = 2π / W_Z    (≈ 12.57 s with W_Z = 0.5 rad/s)
"""

import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


V_X            = 0.35                # m/s forward (body frame)
W_Z            = 0.5                 # rad/s yaw rate
RADIUS         = V_X / W_Z           # 0.70 m
# A full revolution depends only on W_Z (T = 2π / W_Z), so changing V_X
# changes the radius but the circle still closes in CIRCLE_PERIOD seconds.
CIRCLE_PERIOD  = 2.0 * math.pi / W_Z # 12.566 s
HOLD_BEFORE    = 2.0                 # s
HOLD_AFTER     = 2.0                 # s
TOTAL          = HOLD_BEFORE + CIRCLE_PERIOD + HOLD_AFTER
PUBLISH_HZ     = 50


class CircleWalker(Node):
    def __init__(self):
        super().__init__('circle_walker')
        self.pub_   = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timer_ = self.create_timer(1.0 / PUBLISH_HZ, self.tick)
        self.elapsed_ = 0.0
        self.done_    = False
        self.get_logger().info(
            f'Circle: R={RADIUS:.2f} m, vx={V_X} m/s, wz={W_Z} rad/s, '
            f'period={CIRCLE_PERIOD:.2f} s (total {TOTAL:.1f}s)'
        )

    def tick(self):
        if self.done_:
            return
        self.elapsed_ += 1.0 / PUBLISH_HZ
        msg = Twist()
        if HOLD_BEFORE <= self.elapsed_ < HOLD_BEFORE + CIRCLE_PERIOD:
            msg.linear.x  = V_X
            msg.angular.z = W_Z
        self.pub_.publish(msg)
        if self.elapsed_ >= TOTAL:
            self.done_ = True
            self.get_logger().info('Circle scenario complete.')


def main():
    rclpy.init()
    node = CircleWalker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.pub_.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
