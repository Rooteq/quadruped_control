#pragma once

#include <array>
#include <cmath>
#include <eigen3/Eigen/Dense>

#include "gait_scheduler.hpp"

namespace quadro
{

// Forward declaration — full definition in model.hpp, included by .cpp
class QuadroModel;

struct LegTarget {
    Eigen::Vector3d foot_pos = Eigen::Vector3d::Zero();  // world frame
    Eigen::Vector3d foot_vel = Eigen::Vector3d::Zero();  // world frame
    Eigen::Vector3d foot_acc = Eigen::Vector3d::Zero();  // world frame
};

struct SwingState {
    Eigen::Vector3d liftoff_pos = Eigen::Vector3d::Zero();  // body frame
    Eigen::Vector3d landing_pos = Eigen::Vector3d::Zero();  // body frame
    double apex_height = 0.10;  // meters above liftoff/landing z
    bool active = false;
    bool stance_initialized = false;
    Eigen::Vector3d stance_foot_pos = Eigen::Vector3d::Zero();  // foot target held during stance
};

class TrajectoryGenerator
{
public:
    TrajectoryGenerator() = default;
    explicit TrajectoryGenerator(double dt);

    /// Generate desired foot Cartesian targets for all legs.
    /// Swing legs track Bezier arc; stance legs hold last landing position.
    std::array<LegTarget, NUM_LEGS> generate(
        const QuadroModel& model,
        const GaitScheduler& gait,
        const Eigen::Vector3d& desired_linear_vel,
        const Eigen::Vector3d& desired_angular_vel,
        const Eigen::Vector3d& current_vel);

    static constexpr double NOMINAL_HEIGHT = -0.27;  // body-frame z of feet when standing

    Eigen::Vector3d nominalFootPosition(int leg) const
    {
        return {hipPos[leg].x(), hipPos[leg].y(), NOMINAL_HEIGHT};
    }

    Eigen::Vector3d getLandingPos(int leg) const
    {
        return swing_states_[leg].landing_pos;
    }

private:
    /// Raibert foot placement heuristic (body frame)
    Eigen::Vector3d computeLandingPos(
        const QuadroModel& model,
        const GaitScheduler& gait,
        int leg_idx,
        const Eigen::Vector3d& current_vel,
        const Eigen::Vector3d& desired_vel,
        double yaw_rate_des) const;

    /// Bezier swing arc: smooth-step XY, cubic Bezier Z
    Eigen::Vector3d evaluateSwing(const SwingState& state, double phase) const;

    /// Analytical time-derivative of evaluateSwing
    Eigen::Vector3d evaluateSwingVelocity(const SwingState& state, double phase,
                                          double phase_rate) const;

    /// Analytical second time-derivative of evaluateSwing (world frame)
    Eigen::Vector3d evaluateSwingAcceleration(const SwingState& state, double phase,
                                              double phase_rate) const;

    std::array<SwingState, NUM_LEGS> swing_states_;


    double dt_ = 0.033;

    Eigen::Vector3d hipPos[NUM_LEGS] = {
                    {0.12, -0.188, 0.0},
                    {-0.12, -0.188, 0.0},
                    {0.12, 0.188, 0.0},
                    {-0.12, 0.188, 0.0}
                };

    // Eigen::Vector3d legs_origin[NUM_LEGS] = {{0.185, 0.0628, 0.0},
    //                 {0.185, -0.0628, 0.0},
    //                 {-0.185, 0.0628, 0.0},
    //                 {-0.185, -0.0628, 0.0}};

};

} // namespace quadro
