FROM osrf/ros:jazzy-desktop

# Set environment variables for non-interactive installations
ENV DEBIAN_FRONTEND=noninteractive

# Install common build tools and dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    build-essential \
    python3-colcon-common-extensions \
    python3-rosdep \
    ros-jazzy-ament-cmake \
    ros-jazzy-pinocchio \
    ros-jazzy-xacro \
    && rm -rf /var/lib/apt/lists/*

# Install GPU drivers and rendering libraries (Mesa covers Intel/AMD, NVIDIA injected at runtime)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-dri \
    libglx-mesa0 \
    libegl-mesa0 \
    libegl1 \
    libgles2 \
    mesa-utils \
    mesa-vulkan-drivers \
    vulkan-tools \
    libvulkan1 \
    && rm -rf /var/lib/apt/lists/*

# Default to EGL for MuJoCo GPU rendering (works headless and with display)
ENV MUJOCO_GL=egl

# Download and install MuJoCo (Update the version if needed)
ARG MUJOCO_VERSION=3.8.0
RUN mkdir -p /opt/mujoco && \
    wget https://github.com/google-deepmind/mujoco/releases/download/${MUJOCO_VERSION}/mujoco-${MUJOCO_VERSION}-linux-x86_64.tar.gz -O mujoco.tar.gz && \
    tar -xzf mujoco.tar.gz -C /opt/mujoco --strip-components=1 && \
    rm mujoco.tar.gz

# Set the MUJOCO_DIR environment variable
ENV MUJOCO_DIR=/opt/mujoco

# Add MuJoCo library to the library path
ENV LD_LIBRARY_PATH=/opt/mujoco/lib:${LD_LIBRARY_PATH:-}

# Install qpOASES
RUN git clone https://github.com/coin-or/qpOASES.git /opt/qpOASES && \
    cd /opt/qpOASES && mkdir build && cd build && \
    cmake .. -DCMAKE_POSITION_INDEPENDENT_CODE=ON && \
    make -j$(nproc) && make install

# Install OSQP
RUN git clone --recursive https://github.com/osqp/osqp.git -b v0.6.3 /opt/osqp && \
    cd /opt/osqp && mkdir build && cd build && \
    cmake .. -DBUILD_SHARED_LIBS=ON -DCMAKE_POSITION_INDEPENDENT_CODE=ON && \
    make -j$(nproc) && make install

# Install CasADi with IPOPT and OSQP support
RUN apt-get update && apt-get install -y coinor-libipopt-dev && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/casadi/casadi.git -b 3.6.5 /opt/casadi && \
    cd /opt/casadi && mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DWITH_IPOPT=ON -DWITH_OSQP=ON -DCMAKE_POSITION_INDEPENDENT_CODE=ON && \
    make -j$(nproc) && make install

# Setup a non-root user that matches the host user
ARG USERNAME=rosuser
ARG USER_UID=1000
ARG USER_GID=1000

RUN if getent passwd $USER_UID > /dev/null; then userdel -r $(getent passwd $USER_UID | cut -d: -f1); fi \
    && if getent group $USER_GID > /dev/null; then groupdel $(getent group $USER_GID | cut -d: -f1); fi \
    && groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo "$USERNAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && mkdir -p /home/$USERNAME/ros2_ws/src \
    && chown -R $USER_UID:$USER_GID /home/$USERNAME

WORKDIR /home/${USERNAME}/ros2_ws

# Clone mujoco_ros2_control into the container's workspace
USER $USERNAME
RUN cd /home/${USERNAME}/ros2_ws/src && \
    git clone https://github.com/ros-controls/mujoco_ros2_control.git

# Install dependencies using rosdep
USER root
RUN rosdep init || true \
    && rosdep update \
    && apt-get update \
    && rosdep install --from-paths /home/${USERNAME}/ros2_ws/src --ignore-src -r -y \
    && rm -rf /var/lib/apt/lists/*

# Build mujoco_ros2_control
USER $USERNAME
RUN bash -c "source /opt/ros/jazzy/setup.bash && colcon build --symlink-install"

# Add setup instructions to bashrc
RUN echo "source /opt/ros/jazzy/setup.bash" >> /home/${USERNAME}/.bashrc

# Copy entrypoint script into the container
USER root
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

USER $USERNAME

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]