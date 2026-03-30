#pragma once

#include <array>
#include <cmath>
#include <eigen3/Eigen/Dense>

#include "inverse_kinematics.hpp"
#include "gait_scheduler.hpp"

namespace quadro
{

// Forward declaration — full definition in model.hpp, included by .cpp
class QuadroModel;

struct SwingState {
    Eigen::Vector3d liftoff_pos = Eigen::Vector3d::Zero();  // body frame
    Eigen::Vector3d landing_pos = Eigen::Vector3d::Zero();  // body frame
    double apex_height = 0.05;  // meters above liftoff/landing z
    bool active = false;
};

class TrajectoryGenerator
{
public:
    TrajectoryGenerator() = default;
    explicit TrajectoryGenerator(double dt) : dt_(dt) {}

    /// Generate desired joint positions in canonical (JointIdx) order.
    /// Only computes swing leg targets; stance legs hold current position.
    std::array<double, 12> generate(
        const QuadroModel& model,
        const GaitScheduler& gait,
        const Eigen::Vector3d& desired_linear_vel,
        const Eigen::Vector3d& desired_angular_vel);

private:
    /// Raibert foot placement heuristic (body frame)
    Eigen::Vector3d computeLandingPos(
        const QuadroModel& model,
        const GaitScheduler& gait,
        int leg_idx,
        const Eigen::Vector3d& body_velocity) const;

    /// Half-sine swing arc: XY linear interp, Z sine bump
    Eigen::Vector3d evaluateSwing(const SwingState& state, double phase) const;

    static constexpr double NOMINAL_HEIGHT = -0.27;  // body-frame z of feet when standing

    double dt_ = 0.033;
    InverseKinematics ik;
    std::array<SwingState, NUM_LEGS> swing_states_;

    Eigen::Vector3d hipPos[NUM_LEGS] = {
                    {0.112, -0.188, 0.0},
                    {-0.112, -0.188, 0.0},
                    {0.112, 0.188, 0.0},
                    {-0.112, 0.188, 0.0}
                };

    // Eigen::Vector3d legs_origin[NUM_LEGS] = {{0.185, 0.0628, 0.0},
    //                 {0.185, -0.0628, 0.0},
    //                 {-0.185, 0.0628, 0.0},
    //                 {-0.185, -0.0628, 0.0}};

};

} // namespace quadro
