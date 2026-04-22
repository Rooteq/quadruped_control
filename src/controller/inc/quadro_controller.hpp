#pragma once
#include <string>
#include <array>
#include <memory>
#include <iostream>
#include <cmath>



#include "model.hpp"
#include "trajectory_generator.hpp"
#include "gait_scheduler.hpp"
#include "mpc.hpp"
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
            quadro_model_, gait_scheduler_, desired_linear_vel_, desired_angular_vel_, quadro_model_.stateVector().segment<3>(9));
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

        const auto& x0 = quadro_model_.stateVector();
        Eigen::Vector3d base_pos(x0[3], x0[4], x0[5]);
        Eigen::Matrix3d R_z = quadro_model_.bodyYawRotation();

        for (int leg = 0; leg < static_cast<int>(NUM_LEGS); ++leg)
        {
            Eigen::Vector3d nominal = trajectory_generator_.nominalFootPosition(leg);
            Eigen::Vector3d target_world = base_pos + R_z * nominal;
            leg_targets_for_stand_[leg].foot_pos = {target_world.x(), target_world.y(), z_target};
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
        return dynamic_controller_.computeTorques(quadro_model_, gait_scheduler_, leg_targets_, grfs_);
    }

    void updateMPC()
    {
        std::array<Eigen::Vector3d, NUM_LEGS> footprints;
        for (int i = 0; i < 4; ++i) {
            if (gait_scheduler_.inStance(i)) {
                footprints[i] = quadro_model_.footPosition(i);
            } else {
                footprints[i] = trajectory_generator_.getLandingPos(i);
            }
        }
        mpc_.update(quadro_model_, desired_angular_vel_, desired_linear_vel_, x_ref_, gait_scheduler_, footprints);
    }

    /// Run MPC solve and copy resulting GRFs into grfs_.
    /// Caller must hold grf_mutex_ around this call.
    void runMPC()
    {
        mpc_.run();
        grfs_ = mpc_.groundReactionForces();
    }

    const std::array<Eigen::Vector3d, NUM_LEGS>& groundReactionForces() const { return grfs_; }

    void calculateDynamicsMatrices() // calculate matrices etc
    {
        mpc_.calculateDynamicsMatrices();
    }

    void calculateDesiredBodyTrajectory()
    {
        const auto& x0 = quadro_model_.stateVector();
        double yaw      = x0[2];
        double yaw_rate = desired_angular_vel_[2];

        // Desired Z height from nominal standing height
        double z_pos_des_body = -TrajectoryGenerator::NOMINAL_HEIGHT;

        // Initialize anchor on first pass
        if (!pos_des_initialized_)
        {
            pos_des_world_ << x0[3], x0[4], z_pos_des_body;
            pos_des_initialized_ = true;
        }

        // Clamp desired XY to within max_pos_error of current XY to prevent windup (spring-like limit)
        const double max_pos_error = 0.1;
        if (pos_des_world_[0] - x0[3] > max_pos_error) pos_des_world_[0] = x0[3] + max_pos_error;
        if (x0[3] - pos_des_world_[0] > max_pos_error) pos_des_world_[0] = x0[3] - max_pos_error;

        if (pos_des_world_[1] - x0[4] > max_pos_error) pos_des_world_[1] = x0[4] + max_pos_error;
        if (x0[4] - pos_des_world_[1] > max_pos_error) pos_des_world_[1] = x0[4] - max_pos_error;

        // Lock desired Z absolute
        pos_des_world_[2] = z_pos_des_body;

        // Rotate body-frame cmd_vel into world frame using R_z (matches go2.R_z in Python ref)
        const Eigen::Matrix3d& R_z = quadro_model_.bodyYawRotation();
        Eigen::Vector3d vel_world  = R_z * desired_linear_vel_;
        double vx_w = vel_world[0];
        double vy_w = vel_world[1];

        // Advance anchor forward so the "spring" naturally pulls the robot
        pos_des_world_[0] += vx_w * planning_dt_;
        pos_des_world_[1] += vy_w * planning_dt_;

        for (int i = 0; i < HORIZON_STEPS; ++i)
        {
            // Python MPC looks at t = (i+1)*dt since it optimizes the END of the interval
            double t = (i + 1) * MPC_DT;
            
            x_ref_[i].setZero();
            // roll, pitch = 0 (desired level)
            x_ref_[i][2]  = yaw + yaw_rate * t;                // yaw integrated forward
            x_ref_[i][3]  = pos_des_world_[0] + vx_w * t;      // px anchored to clamped pos_des_world
            x_ref_[i][4]  = pos_des_world_[1] + vy_w * t;      // py anchored to clamped pos_des_world
            x_ref_[i][5]  = pos_des_world_[2];                 // pz — absolute constant target height
            x_ref_[i][8]  = yaw_rate;                          // wz
            x_ref_[i][9]  = vx_w;                              // vx
            x_ref_[i][10] = vy_w;                              // vy
            x_ref_[i][12] = x0[12];                            // -g (copied from state)
        }
    }

    const QuadroModel& model() const { return quadro_model_; }
    const Eigen::Vector3d& desiredLinearVelocity() const { return desired_linear_vel_; }
    const Eigen::Vector3d& desiredAngularVelocity() const { return desired_angular_vel_; }

public:
    QuadroModel quadro_model_;
    MPC mpc_;
    DynamicController dynamic_controller_;

private:

    Eigen::Vector3d desired_linear_vel_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d desired_angular_vel_ = Eigen::Vector3d::Zero();

    // Track integrated reference trajectory anchor (spring-like limit)
    Eigen::Vector3d pos_des_world_{};
    bool pos_des_initialized_ = false;

    double planning_dt_ = 0.033;
    TrajectoryGenerator trajectory_generator_;
    GaitScheduler gait_scheduler_;

    std::array<LegTarget, NUM_LEGS> leg_targets_{};
    std::array<LegTarget, NUM_LEGS> leg_targets_for_stand_{};

    bool   standing_complete_    = false;
    double stand_start_time_     = -1.0;
    double settle_time_          = -1.0;

    static constexpr double STAND_LOWER_DURATION = 2.0;  // seconds to lower legs
    static constexpr double STAND_HOLD_DURATION  = 0.2;  // seconds to hold before walking

    // Reference trajectory — HORIZON_STEPS × 13-state vectors
    // x_ref_[n] = desired state at prediction step n
    std::array<Eigen::Matrix<double, 13, 1>, HORIZON_STEPS> x_ref_{};

    // Ground reaction forces
    std::array<Eigen::Vector3d, NUM_LEGS> grfs_{};
};

} // namespace quadro
