import sys

fname = 'src/controller/inc/quadro_controller.hpp'
with open(fname, 'r') as f:
    content = f.read()

old_call = "        mpc_.update(quadro_model_, desired_angular_vel_, desired_linear_vel_, x_ref_, gait_scheduler_);"

new_call = """        std::array<Eigen::Vector3d, NUM_LEGS> footprints;
        for (int i = 0; i < 4; ++i) {
            if (gait_scheduler_.inStance(i)) {
                footprints[i] = quadro_model_.footPosition(i);
            } else {
                footprints[i] = trajectory_generator_.getLandingPos(i);
            }
        }
        mpc_.update(quadro_model_, desired_angular_vel_, desired_linear_vel_, x_ref_, gait_scheduler_, footprints);"""

content = content.replace(old_call, new_call)

with open(fname, 'w') as f:
    f.write(content)
