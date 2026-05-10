#!/bin/bash

set -e

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Get real user UID and GID even if run with sudo
USER_UID=${SUDO_UID:-$(id -u)}
USER_GID=${SUDO_GID:-$(id -g)}

# Build the docker image
echo "Building the Docker image..."
docker build -t quadruped_env:jazzy \
    --build-arg USER_UID=$USER_UID \
    --build-arg USER_GID=$USER_GID \
    -f "$SCRIPT_DIR/Dockerfile" "$SCRIPT_DIR"

# Allow local X11 connections for the GUI (MuJoCo/RViz)
xhost +local:docker

# Define the absolute path to your ROS2 workspace
# This assumes run.sh is executed from anywhere as long as we know the ws root
WORKSPACE_PATH="/home/rooteq/ros2_ws"

echo "Running the Docker container..."
docker run -it --rm \
    --name quadruped_sim \
    --net=host \
    --ipc=host \
    --env="DISPLAY=$DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --volume="$WORKSPACE_PATH/src/quadruped_control:/home/rosuser/ros2_ws/src/quadruped_control" \
    quadruped_env:jazzy