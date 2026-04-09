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

    /// Standing controller — high-gain Cartesian PD to lift and hold the robot.
    /// Used before handing off to the swing/walk controller.
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
    // Stand: high gains needed to lift the robot against gravity
    Eigen::Matrix3d Kp_stand_ = Eigen::DiagonalMatrix<double,3>(300.0, 300.0, 300.0);
    Eigen::Matrix3d Kd_stand_ = Eigen::DiagonalMatrix<double,3>(7.0,  7.0,  7.0);

    // Walk: lower gains for compliant swing tracking
    Eigen::Matrix3d Kp_ = Eigen::DiagonalMatrix<double,3>(40.0, 40.0, 40.0);
    Eigen::Matrix3d Kd_ = Eigen::DiagonalMatrix<double,3>(7.5,  7.5,  7.5);
};

} // namespace quadro
