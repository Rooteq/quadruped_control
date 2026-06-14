#!/usr/bin/env python3
"""
Velocity-step scenario.

  0–2 s   : hold zero  (baseline)
  2–6 s   : step up to vx = V_STEP m/s forward
  6–10 s  : step down back to zero
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


V_STEP        = 0.5      # m/s forward
HOLD_BEFORE   = 2.0      # s
STEP_DURATION = 4.0      # s
HOLD_AFTER    = 4.0      # s
TOTAL         = HOLD_BEFORE + STEP_DURATION + HOLD_AFTER
PUBLISH_HZ    = 50


class StepVx(Node):
    def __init__(self):
        super().__init__('step_vx')
        self.pub_   = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timer_ = self.create_timer(1.0 / PUBLISH_HZ, self.tick)
        self.elapsed_ = 0.0
        self.done_    = False
        self.get_logger().info(
            f'Step vx: {HOLD_BEFORE}s zero, {STEP_DURATION}s @ {V_STEP} m/s, '
            f'{HOLD_AFTER}s zero (total {TOTAL}s)'
        )

    def tick(self):
        if self.done_:
            return
        self.elapsed_ += 1.0 / PUBLISH_HZ
        msg = Twist()
        if HOLD_BEFORE <= self.elapsed_ < HOLD_BEFORE + STEP_DURATION:
            msg.linear.x = V_STEP
        self.pub_.publish(msg)
        if self.elapsed_ >= TOTAL:
            self.done_ = True
            self.get_logger().info('Step scenario complete.')


def main():
    rclpy.init()
    node = StepVx()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.pub_.publish(Twist())   # final zero command
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
