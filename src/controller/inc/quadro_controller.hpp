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
        if (stand_start_time_ < 0.0)
        {
            stand_start_time_ = t;
            for (int leg = 0; leg < static_cast<int>(NUM_LEGS); ++leg)
                foot_start_pos_[leg] = quadro_model_.footPosition(leg);
        }

        const double elapsed   = t - stand_start_time_;
        const double alpha_raw = std::min(elapsed / STAND_LOWER_DURATION, 1.0);
        const double alpha     = alpha_raw * alpha_raw * (3.0 - 2.0 * alpha_raw);

        // Use base_link (NOT the CoM stored in stateVector()[3:5]) as the kinematic
        // anchor: foot_world = base_link + R_bw · body_frame_offset. Mixing the CoM
        // anchor with body-frame offsets bakes the (CoM − base_link) offset into
        // every leg target, which causes asymmetric stance and inward drift.
        const Eigen::Vector3d& base_pos = quadro_model_.bodyPosition();
        const Eigen::Matrix3d& R_bw = quadro_model_.bodyToWorldRotation();

        for (int leg = 0; leg < static_cast<int>(NUM_LEGS); ++leg)
        {
            // Desired foot in body frame: hipPos xy + nominal stand height
            Eigen::Vector3d p_des_body = trajectory_generator_.nominalFootPosition(leg);
            Eigen::Vector3d p_des_world = base_pos + R_bw * p_des_body;

            leg_targets_for_stand_[leg].foot_pos =
                (1.0 - alpha) * foot_start_pos_[leg] + alpha * p_des_world;
            leg_targets_for_stand_[leg].foot_vel = Eigen::Vector3d::Zero();
        }

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
        // Per-horizon-step lever arms (matches Python's r_*_traj_world): for each
        // step, each leg gets either zero (swing) or the lever planned at its
        // last takeoff (stance). MPC's B[n] uses these directly.
        std::array<std::array<Eigen::Vector3d, NUM_LEGS>, HORIZON_STEPS> levers;
        trajectory_generator_.computeHorizonLevers<HORIZON_STEPS>(
            quadro_model_, gait_scheduler_,
            desired_linear_vel_, desired_angular_vel_,
            MPC_DT, levers);

        mpc_.update(quadro_model_, desired_angular_vel_, desired_linear_vel_,
                    x_ref_, gait_scheduler_, levers);
    }

    /// Run MPC solve and copy resulting GRFs into grfs_.
    /// Caller must hold grf_mutex_ around this call.
    void runMPC()
    {
        mpc_.run_casadi(); // mpc_.run();
        // mpc_.run();
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
        const double yaw      = x0[2];
        const double yaw_rate = desired_angular_vel_[2];

        // Rotate body-frame cmd_vel into world frame (matches Python's
        // ComTraj.generate_traj: vel_desired_world = R_z @ [vx_body, vy_body, 0]).
        const Eigen::Matrix3d& R_z = quadro_model_.bodyYawRotation();
        const Eigen::Vector3d vel_world = R_z * desired_linear_vel_;
        const double vx_w = vel_world[0];
        const double vy_w = vel_world[1];

        // ── pos_des_world: slowly-tracked anchor (Python's ComTraj.pos_des_world) ──
        // Init to current CoM on first call, then each tick clamp to within
        // ±MAX_REF_POS_ERROR of the actual CoM. This is what gives MPC a position
        // setpoint to spring back to instead of integrating reference from wherever
        // the body has drifted to. Z is overwritten with the explicit height target.
        if (!ref_pos_des_initialized_)
        {
            ref_pos_des_world_ = Eigen::Vector3d(x0[3], x0[4], desired_z_height_);
            ref_pos_des_initialized_ = true;
        }
        for (int axis = 0; axis < 2; ++axis)   // x, y only
        {
            const double d = ref_pos_des_world_[axis] - x0[3 + axis];
            if (d >  MAX_REF_POS_ERROR) ref_pos_des_world_[axis] = x0[3 + axis] + MAX_REF_POS_ERROR;
            if (d < -MAX_REF_POS_ERROR) ref_pos_des_world_[axis] = x0[3 + axis] - MAX_REF_POS_ERROR;
        }
        ref_pos_des_world_[2] = desired_z_height_;

        // Integrate reference forward from the ANCHOR (not the current CoM). With
        // vel_des = 0 this gives a constant pz/px/py reference that MPC can spring
        // back to; with vel_des > 0 the reference walks forward at the desired rate
        // from the anchor — matching Python's pos_traj_world = pos_des + v_w · t.
        for (int i = 0; i < HORIZON_STEPS; ++i)
        {
            const double t = (i + 1) * MPC_DT;

            x_ref_[i].setZero();
            x_ref_[i][2]  = yaw + yaw_rate * t;                       // yaw
            x_ref_[i][3]  = ref_pos_des_world_[0] + vx_w * t;         // px from anchor
            x_ref_[i][4]  = ref_pos_des_world_[1] + vy_w * t;         // py from anchor
            x_ref_[i][5]  = ref_pos_des_world_[2];                    // pz held at desired
            x_ref_[i][8]  = yaw_rate;                                 // wz
            x_ref_[i][9]  = vx_w;                                     // vx
            x_ref_[i][10] = vy_w;                                     // vy
            x_ref_[i][12] = x0[12];                                   // -g
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

    // Reference-trajectory anchor (Python's ComTraj.pos_des_world). Initialised
    // to the current CoM on the first MPC tick, then clamped to within
    // ±MAX_REF_POS_ERROR of the CoM each tick. MPC integrates px/py from this
    // anchor — gives an actual position setpoint to track instead of letting
    // the reference drift along with the body.
    Eigen::Vector3d ref_pos_des_world_      = Eigen::Vector3d::Zero();
    bool            ref_pos_des_initialized_ = false;
    static constexpr double MAX_REF_POS_ERROR = 0.1;   // metres (matches Python)

    // Desired CoM/base height used as the pz reference. Python sets this from
    // the user command (z_pos_des_body=0.27 in the trot example); we use the
    // same nominal value. Make this a setter target if cmd_vel.linear.z is
    // ever repurposed as a height setpoint.
    double desired_z_height_ = -TrajectoryGenerator::NOMINAL_HEIGHT;  // 0.27 m

    // Reference trajectory — HORIZON_STEPS × 13-state vectors
    // x_ref_[n] = desired state at prediction step n
    std::array<Eigen::Matrix<double, 13, 1>, HORIZON_STEPS> x_ref_{};

    // Ground reaction forces
    std::array<Eigen::Vector3d, NUM_LEGS> grfs_{};
};

} // namespace quadro
