#pragma once
#include <string>
#include <array>
#include <memory>
#include <iostream>
#include <cmath>



#include "model.hpp"
#include "trajectory_generator.hpp"
#include "gait_scheduler.hpp"

namespace quadro
{




// ── MPC horizon ──────────────────────────────────────────────────────────────
// Both the reference trajectory and the dynamics matrices array are sized by
// this constant — they must always match.
static constexpr int    HORIZON_STEPS = 10;     // number of MPC timesteps
static constexpr double MPC_DT        = 0.033;  // seconds per step (~30 Hz)
// Total prediction window: HORIZON_STEPS × MPC_DT = 0.33 s

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

    void calculateDynamicsMatrices() // calculate matrices etc
    {
        
    }

    void calculateDesiredBodyTrajectory()
    {
        const auto& x0 = quadro_model_.stateVector();
        double yaw     = x0[2];
        double yaw_rate = desired_angular_vel_[2];

        // Rotate body-frame cmd_vel into world frame using current yaw
        double cy = std::cos(yaw), sy = std::sin(yaw);
        double vx_w = cy * desired_linear_vel_[0] - sy * desired_linear_vel_[1];
        double vy_w = sy * desired_linear_vel_[0] + cy * desired_linear_vel_[1];

        for (int i = 0; i < HORIZON_STEPS; ++i)
        {
            double t = i * MPC_DT;
            x_ref_[i].setZero();
            // roll, pitch = 0 (desired level)
            x_ref_[i][2]  = yaw + yaw_rate * t;      // yaw integrated forward
            x_ref_[i][3]  = x0[3] + vx_w * t;        // px
            x_ref_[i][4]  = x0[4] + vy_w * t;        // py
            x_ref_[i][5]  = x0[5];                    // pz — track current height
            x_ref_[i][8]  = yaw_rate;                 // wz
            x_ref_[i][9]  = vx_w;                     // vx
            x_ref_[i][10] = vy_w;                     // vy
            x_ref_[i][12] = x0[12];                   // -g (copied from state)
        }
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

    // Reference trajectory — HORIZON_STEPS × 13-state vectors
    // x_ref_[n] = desired state at prediction step n
    std::array<Eigen::Matrix<double, 13, 1>, HORIZON_STEPS> x_ref_{};
};

} // namespace quadro

