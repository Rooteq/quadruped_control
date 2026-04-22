import sys

fname = 'src/controller/inc/quadro_controller.hpp'
with open(fname, 'r') as f:
    content = f.read()

content = content.replace(
    "leg_targets_ = trajectory_generator_.generate(\n            quadro_model_, gait_scheduler_, desired_linear_vel_, desired_angular_vel_);",
    "leg_targets_ = trajectory_generator_.generate(\n            quadro_model_, gait_scheduler_, desired_linear_vel_, desired_angular_vel_, quadro_model_.stateVector().segment<3>(9));"
)

with open(fname, 'w') as f:
    f.write(content)
