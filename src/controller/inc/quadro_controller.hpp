#pragma once
#include <string>
#include <array>
#include <memory>
#include <iostream>

#include "model.hpp"
#include "trajectory_generator.hpp"
#include "gait_scheduler.hpp"
#include "dynamic_controller.hpp"

namespace quadro
{

class Controller
{
public:
    Controller() = default;
    explicit Controller(const std::string& urdf_path, double planning_dt)
        : quadro_model_(urdf_path), planning_dt_(planning_dt), trajectory_generator_(planning_dt)
    {
    }

    void updateState(const Eigen::VectorXd& q, const Eigen::VectorXd& dq, const Eigen::VectorXd& effort)
    {
        quadro_model_.updateState(q, dq, effort);
    }

    void setDesiredVelocity(const Eigen::Vector3d& linear, const Eigen::Vector3d& angular)
    {
        desired_linear_vel_ = linear;
        desired_angular_vel_ = angular;
    }

    void runPlanning()
    {
        gait_scheduler_.advance(planning_dt_);
        leg_targets_ = trajectory_generator_.generate(
            quadro_model_, gait_scheduler_, desired_linear_vel_, desired_angular_vel_);
    }

    bool isStandingComplete() const { return standing_complete_; }

    std::array<double, NUM_JOINTS> calculateStand(double t)
    {
        // Latch start time on first call
        if (stand_start_time_ < 0.0)
            stand_start_time_ = t;

        const double elapsed = t - stand_start_time_;

        // Gradually lower feet: z goes from 0 → NOMINAL_HEIGHT over STAND_LOWER_DURATION seconds
        const double alpha = std::min(elapsed / STAND_LOWER_DURATION, 1.0);
        const double z_target = alpha * TrajectoryGenerator::NOMINAL_HEIGHT;

        for (int leg = 0; leg < static_cast<int>(NUM_LEGS); ++leg)
        {
            Eigen::Vector3d nominal = trajectory_generator_.nominalFootPosition(leg);
            leg_targets_for_stand_[leg].foot_pos = {nominal.x(), nominal.y(), z_target};
            leg_targets_for_stand_[leg].foot_vel = Eigen::Vector3d::Zero();
        }

        // Hold for STAND_HOLD_DURATION after legs are fully lowered, then hand off
        if (!standing_complete_ && alpha >= 1.0)
        {
            if (settle_time_ < 0.0) settle_time_ = t;
            if (t - settle_time_ >= STAND_HOLD_DURATION)
                standing_complete_ = true;
        }

        return dynamic_controller_.computeStand(quadro_model_, leg_targets_for_stand_);
    }

    /// Compute joint torques in canonical (JointIdx) order.
    std::array<double, NUM_JOINTS> calculateControl()
    {
        return dynamic_controller_.computeTorques(
            quadro_model_, gait_scheduler_, leg_targets_);
    }

    const QuadroModel& model() const { return quadro_model_; }
    const Eigen::Vector3d& desiredLinearVelocity() const { return desired_linear_vel_; }
    const Eigen::Vector3d& desiredAngularVelocity() const { return desired_angular_vel_; }

private:
    QuadroModel quadro_model_;

    Eigen::Vector3d desired_linear_vel_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d desired_angular_vel_ = Eigen::Vector3d::Zero();

    double planning_dt_ = 0.033;
    TrajectoryGenerator trajectory_generator_;
    GaitScheduler gait_scheduler_;
    DynamicController dynamic_controller_;

    std::array<LegTarget, NUM_LEGS> leg_targets_{};
    std::array<LegTarget, NUM_LEGS> leg_targets_for_stand_{};

    bool   standing_complete_    = false;
    double stand_start_time_     = -1.0;
    double settle_time_          = -1.0;

    static constexpr double STAND_LOWER_DURATION = 2.0;  // seconds to lower legs
    static constexpr double STAND_HOLD_DURATION  = 1.0;  // seconds to hold before walking
};

} // namespace quadro
