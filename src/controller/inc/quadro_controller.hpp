#pragma once
#include <string>
#include <array>
#include <memory>
#include <iostream>



#include "model.hpp"
#include "trajectory_generator.hpp"
#include "gait_scheduler.hpp"

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
        desired_joint_positions_ = trajectory_generator_.generate(
            quadro_model_, gait_scheduler_, desired_linear_vel_, desired_angular_vel_);
    }

    /// Return desired joint positions in canonical (JointIdx) order.
    std::array<double, NUM_JOINTS> calculateControl()
    {
        return desired_joint_positions_;
    }

    const QuadroModel& model() const { return quadro_model_; }
    const Eigen::Vector3d& desiredLinearVelocity() const { return desired_linear_vel_; }
    const Eigen::Vector3d& desiredAngularVelocity() const { return desired_angular_vel_; }

public:
    QuadroModel quadro_model_;

private:

    Eigen::Vector3d desired_linear_vel_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d desired_angular_vel_ = Eigen::Vector3d::Zero();

    double planning_dt_ = 0.033;
    TrajectoryGenerator trajectory_generator_;
    GaitScheduler gait_scheduler_;

    std::array<double, NUM_JOINTS> desired_joint_positions_{};
};

} // namespace quadro

