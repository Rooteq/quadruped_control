#pragma once

#include <array>
#include <eigen3/Eigen/Dense>

#include "model.hpp"
#include "gait_scheduler.hpp"

namespace quadro
{

class DynamicController
{
public:
    DynamicController() = default;

    /// Compute joint torques: PD + gravity compensation for all legs.
    /// Later: stance legs will use MPC forces instead of PD.
    std::array<double, NUM_JOINTS> computeTorques(
        const QuadroModel& model,
        const GaitScheduler& gait,
        const std::array<double, NUM_JOINTS>& desired_positions);

private:
    // Per-joint PD gains, indexed by JointIdx
    //                              hip    knee   ankle
    std::array<double, NUM_JOINTS> kp_ = {
        40.0,  40.0,  30.0,   // FL
        40.0,  40.0,  30.0,   // FR
        40.0,  40.0,  30.0,   // BL
        40.0,  40.0,  30.0,   // BR
    };

    std::array<double, NUM_JOINTS> kd_ = {
        1.0,   1.0,   0.8,    // FL
        1.0,   1.0,   0.8,    // FR
        1.0,   1.0,   0.8,    // BL
        1.0,   1.0,   0.8,    // BR
    };
};

} // namespace quadro
