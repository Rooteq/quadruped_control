#include "quadro_controller.hpp"

namespace quadro
{

void Controller::runPlanning()
{
    gait_scheduler_.advance(planning_dt_);
    leg_targets_ = trajectory_generator_.generate(
        quadro_model_, gait_scheduler_,
        quadro_model_.stateVector().segment<3>(9),
        desired_linear_vel_, desired_angular_vel_);
}

std::array<double, NUM_JOINTS> Controller::calculateStand(double t)
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

    // base_link (NOT the CoM in stateVector()[3:5]) is the kinematic anchor:
    // foot_world = base_link + R_bw · body_frame_offset. Mixing the CoM anchor
    // with body-frame offsets bakes the (CoM − base_link) offset into every leg
    // target, which causes asymmetric stance and inward drift.
    const Eigen::Vector3d& base_pos = quadro_model_.bodyPosition();
    const Eigen::Matrix3d& R_bw = quadro_model_.bodyToWorldRotation();

    for (int leg = 0; leg < static_cast<int>(NUM_LEGS); ++leg)
    {
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

void Controller::updateMPC()
{
    // Per-horizon-step lever arms: for each step, each leg gets either zero
    // (swing) or the lever planned at its last takeoff (stance). MPC's B[n]
    // uses these directly.
    std::array<std::array<Eigen::Vector3d, NUM_LEGS>, HORIZON_STEPS> levers;
    trajectory_generator_.computeHorizonLevers<HORIZON_STEPS>(
        quadro_model_, gait_scheduler_,
        desired_linear_vel_, desired_angular_vel_,
        MPC_DT, levers);

    mpc_.update(quadro_model_, desired_angular_vel_, desired_linear_vel_,
                x_ref_, gait_scheduler_, levers);
}

void Controller::calculateDesiredBodyTrajectory()
{
    const auto& x0 = quadro_model_.stateVector();
    const double yaw      = x0[2];
    const double yaw_rate = desired_angular_vel_[2];

    const Eigen::Matrix3d& R_z = quadro_model_.bodyYawRotation();
    const Eigen::Vector3d vel_world = R_z * desired_linear_vel_;
    const double vx_w = vel_world[0];
    const double vy_w = vel_world[1];

    // pos_des_world: slowly-tracked anchor. Init to current CoM on the first
    // call, then each tick clamp to within ±MAX_REF_POS_ERROR of the actual
    // CoM. Z is overwritten with the explicit height target.
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

    // Integrate the reference forward from the anchor (not the current CoM), so
    // with vel_des = 0 the px/py reference is constant and MPC can spring back
    // to it; with vel_des > 0 it walks forward at the desired rate.
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

} // namespace quadro
