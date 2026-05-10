#!/bin/bash
set -e

# Source core ROS2
source "/opt/ros/jazzy/setup.bash"

# Source workspace if built
if [ -f "/home/rosuser/ros2_ws/install/setup.bash" ]; then
    source "/home/rosuser/ros2_ws/install/setup.bash"
fi

exec "$@"