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
WORKSPACE_PATH="/home/rooteq/ros2_ws"

# --- GPU auto-detection ---
GPU_FLAGS=""
GPU_ENV=""

detect_gpu() {
    # Check for NVIDIA GPU
    if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
        echo "Detected NVIDIA GPU"
        GPU_FLAGS="--gpus all --device /dev/dri"
        GPU_ENV="-e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all -e MUJOCO_GL=egl"
        return
    fi

    # Check /dev/dri for AMD or Intel GPUs
    if [ -d /dev/dri ]; then
        local gpu_vendor=""
        for card in /sys/class/drm/card*/device/vendor; do
            if [ -f "$card" ]; then
                vendor_id=$(cat "$card")
                case "$vendor_id" in
                    0x1002) gpu_vendor="AMD" ;;
                    0x8086) gpu_vendor="Intel" ;;
                esac
            fi
        done

        if [ -n "$gpu_vendor" ]; then
            echo "Detected $gpu_vendor GPU"
            GPU_FLAGS="--device /dev/dri"
            GPU_ENV="-e MUJOCO_GL=egl"

            # Add video/render group access
            if [ -e /dev/dri/renderD128 ]; then
                local render_gid=$(stat -c '%g' /dev/dri/renderD128)
                GPU_FLAGS="$GPU_FLAGS --group-add $render_gid"
            fi
            return
        fi
    fi

    echo "WARNING: No GPU detected, MuJoCo will use software rendering"
    GPU_ENV="-e MUJOCO_GL=osmesa"
}

detect_gpu

echo "Running the Docker container..."
docker run -it --rm \
    --name quadruped_sim \
    --net=host \
    --ipc=host \
    --env="DISPLAY=$DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --volume="$WORKSPACE_PATH/src/quadruped_control:/home/rosuser/ros2_ws/src/quadruped_control" \
    $GPU_FLAGS \
    $GPU_ENV \
    quadruped_env:jazzy
