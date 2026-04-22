import sys

# Patch mpc.hpp
with open('src/controller/inc/mpc.hpp', 'r') as f:
    content = f.read()

content = content.replace(
    "                const GaitScheduler& gait_scheduler);",
    "                const GaitScheduler& gait_scheduler,\n                const std::array<Eigen::Vector3d, NUM_LEGS>& foot_positions);"
)
with open('src/controller/inc/mpc.hpp', 'w') as f:
    f.write(content)

# Patch mpc.cpp
with open('src/controller/src/mpc.cpp', 'r') as f:
    content = f.read()

content = content.replace(
    "                 const GaitScheduler& gait_scheduler)\n{\n    x0_           = model.stateVector();",
    "                 const GaitScheduler& gait_scheduler,\n                 const std::array<Eigen::Vector3d, NUM_LEGS>& foot_positions)\n{\n    x0_           = model.stateVector();"
)

content = content.replace(
    """    // Since Pinocchio is using a FreeFlyer model, footPosition already
    // returns coordinates in the absolute world frame.
    for (int i = 0; i < static_cast<int>(NUM_LEGS); ++i)
        foot_positions_[i] = model.footPosition(i);""",
    """    for (int i = 0; i < static_cast<int>(NUM_LEGS); ++i)
        foot_positions_[i] = foot_positions[i];"""
)

with open('src/controller/src/mpc.cpp', 'w') as f:
    f.write(content)
