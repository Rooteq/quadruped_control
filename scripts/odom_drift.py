#!/usr/bin/env python3
"""
Subscribe to /odom, add simulated drift + noise, republish as /odom_error.

  Position    — accumulates a random-walk drift (slowly growing offset).
  Orientation — accumulates a random-walk yaw bias (slowly growing yaw offset).
  Twist       — white Gaussian noise added per message (non-accumulating).

Tune the *_STD constants to dial drift severity up or down.
"""

import math
import random
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry


# Random-walk standard deviations per second (cumulative effect).
POS_DRIFT_STD_PER_S = 0.02   # m / sqrt(s)   — slow position drift
YAW_DRIFT_STD_PER_S = 0.02   # rad / sqrt(s) — slow yaw drift

# White-noise standard deviations per sample (instantaneous error on the twist).
LIN_VEL_NOISE_STD = 0.01     # m/s
ANG_VEL_NOISE_STD = 0.01     # rad/s


def quat_to_yaw(qw, qx, qy, qz):
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def quat_from_yaw_delta(qw, qx, qy, qz, dyaw):
    # Post-multiply by Rz(dyaw): adds dyaw to the yaw component.
    c, s = math.cos(dyaw * 0.5), math.sin(dyaw * 0.5)
    return (
        qw * c - qz * s,   # w
        qx * c + qy * s,   # x
        qy * c - qx * s,   # y
        qz * c + qw * s,   # z
    )


class OdomDriftSimulator(Node):
    def __init__(self):
        super().__init__('odom_drift_simulator')

        self.sub_ = self.create_subscription(
            Odometry, '/odom', self.odom_cb, 10)
        self.pub_ = self.create_publisher(Odometry, '/odom_error', 10)

        self.pos_offset_ = [0.0, 0.0, 0.0]   # accumulated position drift
        self.yaw_offset_ = 0.0               # accumulated yaw drift
        self.last_stamp_ = None

        self.get_logger().info(
            f'Republishing /odom → /odom_error with drift: '
            f'pos {POS_DRIFT_STD_PER_S} m/√s, yaw {YAW_DRIFT_STD_PER_S} rad/√s, '
            f'twist noise {LIN_VEL_NOISE_STD} m/s, {ANG_VEL_NOISE_STD} rad/s'
        )

    def odom_cb(self, msg: Odometry):
        # Time step for random-walk scaling.
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.last_stamp_ is None:
            dt = 0.0
        else:
            dt = max(0.0, stamp - self.last_stamp_)
        self.last_stamp_ = stamp

        # Random-walk increment: σ_step = σ_per_s · √dt.
        sqrt_dt = math.sqrt(dt)
        # for i in range(3):
        self.pos_offset_[0] += POS_DRIFT_STD_PER_S * sqrt_dt
        self.pos_offset_[1] += POS_DRIFT_STD_PER_S * sqrt_dt
        # self.yaw_offset_ += random.gauss(0.0, YAW_DRIFT_STD_PER_S * sqrt_dt)
        self.yaw_offset_ += YAW_DRIFT_STD_PER_S * sqrt_dt
            # self.pos_offset_[i] += random.gauss(0.0, POS_DRIFT_STD_PER_S * sqrt_dt)
        # self.yaw_offset_ += random.gauss(0.0, YAW_DRIFT_STD_PER_S * sqrt_dt)

        # Drifted pose.
        out = Odometry()
        out.header = msg.header
        out.child_frame_id = msg.child_frame_id

        out.pose.pose.position.x = msg.pose.pose.position.x + self.pos_offset_[0]
        out.pose.pose.position.y = msg.pose.pose.position.y + self.pos_offset_[1]
        out.pose.pose.position.z = msg.pose.pose.position.z + self.pos_offset_[2]

        qw, qx, qy, qz = (
            msg.pose.pose.orientation.w, msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y, msg.pose.pose.orientation.z)
        nw, nx, ny, nz = quat_from_yaw_delta(qw, qx, qy, qz, self.yaw_offset_)
        out.pose.pose.orientation.w = nw
        out.pose.pose.orientation.x = nx
        out.pose.pose.orientation.y = ny
        out.pose.pose.orientation.z = nz

        # Per-sample white noise on the twist (does not accumulate).
        out.twist.twist.linear.x  = msg.twist.twist.linear.x  + random.gauss(0.0, LIN_VEL_NOISE_STD)
        out.twist.twist.linear.y  = msg.twist.twist.linear.y  + random.gauss(0.0, LIN_VEL_NOISE_STD)
        out.twist.twist.linear.z  = msg.twist.twist.linear.z  + random.gauss(0.0, LIN_VEL_NOISE_STD)
        out.twist.twist.angular.x = msg.twist.twist.angular.x + random.gauss(0.0, ANG_VEL_NOISE_STD)
        out.twist.twist.angular.y = msg.twist.twist.angular.y + random.gauss(0.0, ANG_VEL_NOISE_STD)
        out.twist.twist.angular.z = msg.twist.twist.angular.z + random.gauss(0.0, ANG_VEL_NOISE_STD)

        self.pub_.publish(out)


def main():
    rclpy.init()
    node = OdomDriftSimulator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
