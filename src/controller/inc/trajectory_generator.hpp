#pragma once

#include <array>
#include <eigen3/Eigen/Dense>

#include "inverse_kinematics.hpp"

namespace quadro
{

// Forward declaration — full definition in model.hpp, included by .cpp
class QuadroModel;

class TrajectoryGenerator
{
public:
    TrajectoryGenerator() = default;
    explicit TrajectoryGenerator(double dt) : dt_(dt) {}

    /// Generate desired joint positions in canonical (JointIdx) order.
    std::array<double, 12> generate(
        const QuadroModel& model,
        const Eigen::Vector3d& desired_linear_vel,
        const Eigen::Vector3d& desired_angular_vel);

private:
    double dt_ = 0.033;  // default 30Hz
    InverseKinematics ik;

    Eigen::Vector3d legs_origin[sizeof(LegIdx)] = {{0.185, 0.0628, 0.0},
                                     {0.185, -0.0628, 0.0},
                                     {-0.185, 0.0628, 0.0},
                                     {-0.185, -0.0628, 0.0}};
};

} // namespace quadro
