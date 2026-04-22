#pragma once

#include <array>
#include <eigen3/Eigen/Dense>

#include "model.hpp"
#include "gait_scheduler.hpp"
#include "trajectory_generator.hpp"

namespace quadro
{

class DynamicController
{
public:
    DynamicController() = default;

    /// Standing controller — Cartesian PD via J^T to reach nominal pose.
    /// leg_targets hold desired foot positions in world frame.
    std::array<double, NUM_JOINTS> computeStand(
        const QuadroModel& model,
        const std::array<LegTarget, NUM_LEGS>& leg_targets);

    /// Swing/walk controller — Cartesian PD swing + J^T GRF stance + gravity.
    std::array<double, NUM_JOINTS> computeTorques(
        const QuadroModel& model,
        const GaitScheduler& gait,
        const std::array<LegTarget, NUM_LEGS>& leg_targets,
        const std::array<Eigen::Vector3d, NUM_LEGS>& grfs);

private:
    // Stand: Cartesian PD gains — needs Kp*err > mg/4 per leg (~17 N for 7 kg robot)
    Eigen::Matrix3d Kp_stand_ = Eigen::DiagonalMatrix<double,3>(300.0, 300.0, 300.0);
    Eigen::Matrix3d Kd_stand_ = Eigen::DiagonalMatrix<double,3>(10.0,  10.0,  10.0);

    // Walk: higher exact PD matching python
    Eigen::Matrix3d Kp_ = Eigen::DiagonalMatrix<double,3>(80.0, 80.0, 80.0);
    Eigen::Matrix3d Kd_ = Eigen::DiagonalMatrix<double,3>(17.0,  17.0,  17.0);
};

} // namespace quadro
