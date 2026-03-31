#include "dynamic_controller.hpp"

namespace quadro
{

std::array<double, NUM_JOINTS> DynamicController::computeTorques(
    const QuadroModel& model,
    const GaitScheduler& /*gait*/,
    const std::array<double, NUM_JOINTS>& desired_positions)
{
    std::array<double, NUM_JOINTS> torques{};

    const auto& q = model.jointPositions();
    const auto& dq = model.jointVelocities();
    const auto& g = model.gravityCompensation();

    // PD + gravity compensation for all joints
    // Later: stance legs will use J^T * mpc_forces instead of PD
    for (size_t i = 0; i < NUM_JOINTS; ++i)
    {
        double pos_error = desired_positions[i] - q[i];
        double vel_error = 0.0 - dq[i];  // desired velocity = 0

        torques[i] = kp_[i] * pos_error + kd_[i] * vel_error + g[i];
    }

    return torques;
}

} // namespace quadro
