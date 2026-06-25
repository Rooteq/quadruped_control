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

    void runPlanning();

    bool isStandingComplete() const { return standing_complete_; }

    std::array<double, NUM_JOINTS> calculateStand(double t);

    /// Compute joint torques in canonical (JointIdx) order.
    std::array<double, NUM_JOINTS> calculateControl()
    {
        return dynamic_controller_.computeTorques(quadro_model_, gait_scheduler_, leg_targets_, grfs_);
    }

    void updateMPC();

    /// Run MPC solve and copy resulting GRFs into grfs_.
    /// Caller must hold grf_mutex_ around this call.
    void runMPC()
    {
        mpc_.run_casadi();
        grfs_ = mpc_.groundReactionForces();
    }

    const std::array<Eigen::Vector3d, NUM_LEGS>& groundReactionForces() const { return grfs_; }

    void calculateDynamicsMatrices()
    {
        mpc_.calculateDynamicsMatrices();
    }

    void calculateDesiredBodyTrajectory();

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

    double planning_dt_ = 0.033;
    TrajectoryGenerator trajectory_generator_;
    GaitScheduler gait_scheduler_;

    std::array<LegTarget, NUM_LEGS> leg_targets_{};
    std::array<LegTarget, NUM_LEGS> leg_targets_for_stand_{};
    std::array<Eigen::Vector3d, NUM_LEGS> foot_start_pos_{};

    bool   standing_complete_    = false;
    double stand_start_time_     = -1.0;
    double settle_time_          = -1.0;

    static constexpr double STAND_LOWER_DURATION = 3.0;  // seconds to reach nominal height
    static constexpr double STAND_HOLD_DURATION  = 1.5;  // seconds to hold before walking

    // Reference-trajectory anchor: initialised to the current CoM on the first
    // MPC tick, then clamped to within ±MAX_REF_POS_ERROR of the CoM each tick.
    // MPC integrates px/py from this anchor to get a position setpoint to track
    // instead of letting the reference drift along with the body.
    Eigen::Vector3d ref_pos_des_world_      = Eigen::Vector3d::Zero();
    bool            ref_pos_des_initialized_ = false;
    static constexpr double MAX_REF_POS_ERROR = 0.1;   // metres

    // Desired CoM/base height used as the pz reference.
    double desired_z_height_ = -TrajectoryGenerator::NOMINAL_HEIGHT;

    // Reference trajectory: x_ref_[n] = desired 13-state at prediction step n
    std::array<Eigen::Matrix<double, 13, 1>, HORIZON_STEPS> x_ref_{};

    std::array<Eigen::Vector3d, NUM_LEGS> grfs_{};
};

} // namespace quadro
