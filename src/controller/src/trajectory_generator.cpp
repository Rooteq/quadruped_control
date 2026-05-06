#include "trajectory_generator.hpp"
#include "model.hpp"

namespace quadro
{

TrajectoryGenerator::TrajectoryGenerator(double dt) : dt_(dt)
{
    // Initialize stance foot positions to nominal standing pose so that
    // legs starting in stance on the first planning tick get a valid target.
    for (int leg = 0; leg < static_cast<int>(NUM_LEGS); ++leg)
    {
        swing_states_[leg].stance_foot_pos = hipPos[leg];
        swing_states_[leg].stance_foot_pos.z() = NOMINAL_HEIGHT;
        swing_states_[leg].landing_pos = swing_states_[leg].stance_foot_pos;
        swing_states_[leg].stance_initialized = true;
    }
}

std::array<LegTarget, NUM_LEGS> TrajectoryGenerator::generate(
    const QuadroModel& model,
    const GaitScheduler& gait,
    const Eigen::Vector3d& desired_linear_vel,
    const Eigen::Vector3d& desired_angular_vel,
    const Eigen::Vector3d& current_vel)
{
    std::array<LegTarget, NUM_LEGS> targets{};

    const Eigen::VectorXd& state = model.stateVector();
    Eigen::Vector3d base_pos_w = model.bodyPosition();          // base_link, not CoM
    Eigen::Vector3d base_vel_w(state[9], state[10], state[11]); // CoM linear velocity (world)
    Eigen::Vector3d com_world(state[3], state[4], state[5]);    // CoM (state[3:5] is CoM)
    const Eigen::Matrix3d& R_b_w = model.bodyToWorldRotation();

    // Maintain pos_des_world_: init to current CoM on first call, then clamp
    // to within ±MAX_POS_ERROR of the actual CoM each tick. Mirrors Python's
    // ComTraj.pos_des_world — slowly tracked anchor, never integrated by vel_des.
    if (!pos_des_initialized_)
    {
        pos_des_world_       = com_world;
        pos_des_initialized_ = true;
    }
    for (int axis = 0; axis < 2; ++axis)   // x, y only
    {
        const double d = pos_des_world_[axis] - com_world[axis];
        if (d >  MAX_POS_ERROR) pos_des_world_[axis] = com_world[axis] + MAX_POS_ERROR;
        if (d < -MAX_POS_ERROR) pos_des_world_[axis] = com_world[axis] - MAX_POS_ERROR;
    }

    // Phase rate: how fast swing phase [0,1] advances per second
    const double swing_duration = (1.0 - gait.gait().duty_cycle) * gait.gait().period;
    const double phase_rate = (swing_duration > 1e-6) ? 1.0 / swing_duration : 0.0;

    for (int leg = 0; leg < static_cast<int>(NUM_LEGS); ++leg)
    {
        // With a FreeFlyer model, pinocchio outputs already give world absolute positions
        Eigen::Vector3d p_world = model.footPosition(leg);
        Eigen::Vector3d v_world = model.footVelocity(leg);

        if (gait.inStance(leg))
        {
            // On swing→stance transition, lock the touchdown position.
            // Holding this anchor means the Cartesian PD (used as fallback when
            // a leg is mis-routed to swing) pulls back to the correct foothold
            // rather than following wherever the foot drifted.
            if (!swing_states_[leg].stance_initialized)
            {
                swing_states_[leg].stance_foot_pos   = p_world;
                swing_states_[leg].stance_initialized = true;
            }

            swing_states_[leg].active = false;

            targets[leg].foot_pos = swing_states_[leg].stance_foot_pos;
            targets[leg].foot_vel = Eigen::Vector3d::Zero();
            targets[leg].foot_acc = Eigen::Vector3d::Zero();
        }
        else
        {
            swing_states_[leg].stance_initialized = false;

            // On stance→swing transition, initialize arc
            if (!swing_states_[leg].active)
            {
                swing_states_[leg].liftoff_pos = p_world;
                // Use ACTUAL body velocity for Raibert heuristic
                swing_states_[leg].landing_pos = computeLandingPos(
                    model, gait, leg, current_vel, desired_linear_vel,
                    desired_angular_vel[2], pos_des_world_);
                swing_states_[leg].active = true;
            }

            double phase = gait.swingPhase(leg);  // [0, 1]
            targets[leg].foot_pos = evaluateSwing(swing_states_[leg], phase);
            targets[leg].foot_vel = evaluateSwingVelocity(swing_states_[leg], phase, phase_rate);
            targets[leg].foot_acc = evaluateSwingAcceleration(swing_states_[leg], phase, phase_rate);
        }
    }

    return targets;
}

Eigen::Vector3d TrajectoryGenerator::computeLandingPos(
    const QuadroModel& model,
    const GaitScheduler& gait,
    int leg_idx,
    const Eigen::Vector3d& current_vel,
    const Eigen::Vector3d& desired_vel,
    double yaw_rate_des,
    const Eigen::Vector3d& pos_des_world) const
{
    // Use base_link (not CoM) as kinematic anchor — see calculateStand for rationale.
    Eigen::Vector3d base_pos = model.bodyPosition();
    base_pos.z() = 0.0;                                  // project onto ground plane
    Eigen::Matrix3d R_z = model.bodyYawRotation();
    Eigen::Vector3d hip_pos_world = base_pos + R_z * hipPos[leg_idx];

    double t_swing   = (1.0 - gait.gait().duty_cycle) * gait.gait().period;
    double t_stance  = gait.gait().duty_cycle * gait.gait().period;
    double T         = t_swing + 0.5 * t_stance;
    double pred_time = T / 2.0;

    // Body-frame x = forward → larger gain on forward axis (matches Python).
    const double k_v_x = 0.4 * T;          // forward
    const double k_v_y = 0.2 * T;          // lateral
    const double k_p_x = 0.10;             // forward
    const double k_p_y = 0.05;             // lateral

    // Desired velocity in world frame (rotate body-frame cmd through R_z).
    const Eigen::Vector3d des_vel_world = R_z * desired_vel;

    // Nominal: hip projected onto ground plane
    Eigen::Vector3d pos_nominal(hip_pos_world.x(), hip_pos_world.y(), 0.02);

    // Drift: integrate the DESIRED velocity over pred_time (matches Python's
    // pos_drift_term — uses x_vel_des, NOT the current measured velocity).
    // Using current_vel here is positive feedback on body drift; desired_vel
    // makes the foot land where the body SHOULD be at mid-stance.
    Eigen::Vector3d drift(des_vel_world.x() * pred_time,
                          des_vel_world.y() * pred_time, 0.0);

    // Velocity correction: damps body velocity error (foot lands in direction
    // of velocity error, body decelerates as it passes over the foot).
    Eigen::Vector3d vel_correction(k_v_x * (current_vel.x() - des_vel_world.x()),
                                   k_v_y * (current_vel.y() - des_vel_world.y()),
                                   0.0);

    // Position correction: spring back toward the slowly-tracked anchor
    // pos_des_world. Without this term the body has no equilibrium position —
    // it just damps velocity and stays wherever it drifted.
    const Eigen::VectorXd& state = model.stateVector();
    Eigen::Vector3d pos_correction(k_p_x * (state[3] - pos_des_world.x()),
                                   k_p_y * (state[4] - pos_des_world.y()),
                                   0.0);

    // Rotation correction: accounts for where the hip will be after yawing by dtheta
    // r_xy = hip offset from body center in world frame
    double dtheta = yaw_rate_des * pred_time;
    double r_x = hip_pos_world.x() - base_pos.x();
    double r_y = hip_pos_world.y() - base_pos.y();
    Eigen::Vector3d rot_correction(-dtheta * r_y, dtheta * r_x, 0.0);

    return pos_nominal + drift + vel_correction + pos_correction + rot_correction;
}

Eigen::Vector3d TrajectoryGenerator::evaluateSwing(
    const SwingState& state, double phase) const
{
    // Smooth-step (Hermite): zero velocity at phase=0 and phase=1
    double s = phase * phase * (3.0 - 2.0 * phase);

    double x = state.liftoff_pos.x() + s * (state.landing_pos.x() - state.liftoff_pos.x());
    double y = state.liftoff_pos.y() + s * (state.landing_pos.y() - state.liftoff_pos.y());

    // Cubic Bezier Z: P0=liftoff, P1=liftoff+h, P2=landing+h, P3=landing
    double t = phase;
    double t2 = t * t, t3 = t2 * t;
    double mt = 1.0 - t, mt2 = mt * mt, mt3 = mt2 * mt;

    double p0 = state.liftoff_pos.z();
    double p1 = state.liftoff_pos.z() + state.apex_height;
    double p2 = state.landing_pos.z() + state.apex_height;
    double p3 = state.landing_pos.z();

    double z = mt3 * p0 + 3.0 * mt2 * t * p1 + 3.0 * mt * t2 * p2 + t3 * p3;

    return {x, y, z};
}

Eigen::Vector3d TrajectoryGenerator::evaluateSwingVelocity(
    const SwingState& state, double phase, double phase_rate) const
{
    // d/dt = d/dphase * phase_rate

    // XY: smooth-step derivative — ds/dphase = 6*phase*(1-phase)
    double ds_dphase = 6.0 * phase * (1.0 - phase);
    double vx = (state.landing_pos.x() - state.liftoff_pos.x()) * ds_dphase * phase_rate;
    double vy = (state.landing_pos.y() - state.liftoff_pos.y()) * ds_dphase * phase_rate;

    // Z: cubic Bezier derivative — dz/dphase = 3*(mt2*(p1-p0) + 2*mt*t*(p2-p1) + t2*(p3-p2))
    double t = phase;
    double t2 = t * t;
    double mt = 1.0 - t, mt2 = mt * mt;

    double p0 = state.liftoff_pos.z();
    double p1 = state.liftoff_pos.z() + state.apex_height;
    double p2 = state.landing_pos.z() + state.apex_height;
    double p3 = state.landing_pos.z();

    double dz_dphase = 3.0 * (mt2 * (p1 - p0) + 2.0 * mt * t * (p2 - p1) + t2 * (p3 - p2));
    double vz = dz_dphase * phase_rate;

    return {vx, vy, vz};
}

Eigen::Vector3d TrajectoryGenerator::evaluateSwingAcceleration(
    const SwingState& state, double phase, double phase_rate) const
{
    // d²/dt² = d²/dphase² · phase_rate²  (constant phase_rate over a swing)
    const double pr2 = phase_rate * phase_rate;

    // XY: smooth-step second derivative — d²s/dphase² = 6·(1 − 2·phase)
    double d2s_dphase2 = 6.0 * (1.0 - 2.0 * phase);
    double ax = (state.landing_pos.x() - state.liftoff_pos.x()) * d2s_dphase2 * pr2;
    double ay = (state.landing_pos.y() - state.liftoff_pos.y()) * d2s_dphase2 * pr2;

    // Z: cubic Bezier second derivative
    // d²B/dt² = 6·(1−t)·(p2 − 2·p1 + p0) + 6·t·(p3 − 2·p2 + p1)
    double t  = phase;
    double mt = 1.0 - t;

    double p0 = state.liftoff_pos.z();
    double p1 = state.liftoff_pos.z() + state.apex_height;
    double p2 = state.landing_pos.z() + state.apex_height;
    double p3 = state.landing_pos.z();

    double d2z_dphase2 = 6.0 * mt * (p2 - 2.0 * p1 + p0)
                       + 6.0 * t  * (p3 - 2.0 * p2 + p1);
    double az = d2z_dphase2 * pr2;

    return {ax, ay, az};
}

} // namespace quadro
