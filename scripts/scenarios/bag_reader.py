"""
Helpers for reading rosbag2 sqlite3 files and extracting common quantities.

Imported by the plot_*.py scripts. Not meant to run standalone.
"""

import glob
import math
import os

import numpy as np

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def _detect_storage_id(bag_path):
    """Pick the rosbag2 storage plugin by sniffing the bag directory contents."""
    if glob.glob(os.path.join(bag_path, '*.mcap')):
        return 'mcap'
    if glob.glob(os.path.join(bag_path, '*.db3')):
        return 'sqlite3'
    # Default to mcap (ROS 2 Jazzy+ default); rosbag2_py will raise if wrong.
    return 'mcap'


def read_bag(bag_path, topics=None, storage_id=None):
    """
    Read a rosbag2 directory and return a dict of topic name → list of
    (timestamp_ns, deserialised_msg) tuples. If `topics` is given, only those
    topics are kept; otherwise everything in the bag is loaded.

    `storage_id` is auto-detected from file extensions (.mcap → 'mcap',
    .db3 → 'sqlite3'); pass explicitly to override.
    """
    if storage_id is None:
        storage_id = _detect_storage_id(bag_path)
    storage_options   = rosbag2_py.StorageOptions(uri=bag_path, storage_id=storage_id)
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr')

    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    type_map   = {t.name: t.type for t in reader.get_all_topics_and_types()}
    filter_set = set(topics) if topics is not None else None
    data       = {t: [] for t in (topics if topics is not None else type_map.keys())}

    while reader.has_next():
        topic, raw, t = reader.read_next()
        if filter_set is not None and topic not in filter_set:
            continue
        if topic not in type_map:
            continue
        msg = deserialize_message(raw, get_message(type_map[topic]))
        data.setdefault(topic, []).append((t, msg))
    return data


def quat_to_rpy(qw, qx, qy, qz):
    """ZYX Euler angles (roll, pitch, yaw) from quaternion."""
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def common_t0(data):
    """Earliest receive timestamp across all topics that have at least one message."""
    earliest = [v[0][0] for v in data.values() if v]
    if not earliest:
        return None
    return min(earliest)


def odom_extract(odom_msgs, t0):
    """Pull t (s, relative to t0), pos, world-frame vel, ang_vel, rpy from /odom-style messages."""
    if not odom_msgs:
        return None
    t = (np.array([m[0] for m in odom_msgs]) - t0) * 1e-9
    pos = np.array([[m[1].pose.pose.position.x,
                     m[1].pose.pose.position.y,
                     m[1].pose.pose.position.z] for m in odom_msgs])
    vel = np.array([[m[1].twist.twist.linear.x,
                     m[1].twist.twist.linear.y,
                     m[1].twist.twist.linear.z] for m in odom_msgs])
    ang = np.array([[m[1].twist.twist.angular.x,
                     m[1].twist.twist.angular.y,
                     m[1].twist.twist.angular.z] for m in odom_msgs])
    rpy = np.array([quat_to_rpy(m[1].pose.pose.orientation.w,
                                m[1].pose.pose.orientation.x,
                                m[1].pose.pose.orientation.y,
                                m[1].pose.pose.orientation.z) for m in odom_msgs])
    return {'t': t, 'pos': pos, 'vel': vel, 'ang_vel': ang, 'rpy': rpy}


def cmd_vel_extract(cmd_msgs, t0):
    """Pull t, vx, vy, wz from /cmd_vel."""
    if not cmd_msgs:
        return None
    t  = (np.array([m[0] for m in cmd_msgs]) - t0) * 1e-9
    vx = np.array([m[1].linear.x  for m in cmd_msgs])
    vy = np.array([m[1].linear.y  for m in cmd_msgs])
    wz = np.array([m[1].angular.z for m in cmd_msgs])
    return {'t': t, 'vx': vx, 'vy': vy, 'wz': wz}


def float_array_extract(float_msgs, t0):
    """Pull (t, data) from Float64MultiArray messages, where data is shape (N, K)."""
    if not float_msgs:
        return None
    t = (np.array([m[0] for m in float_msgs]) - t0) * 1e-9
    data = np.array([list(m[1].data) for m in float_msgs])
    return {'t': t, 'data': data}
