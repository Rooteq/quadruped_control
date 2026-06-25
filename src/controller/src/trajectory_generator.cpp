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
    const Eigen::Vector3d& current_vel,
    const Eigen::Vector3d& desired_linear_vel,
    const Eigen::Vector3d& desired_angular_vel)
{
    std::array<LegTarget, NUM_LEGS> targets{};

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
            // On swing→stance transition, lock the touchdown position so the
            // Cartesian PD fallback pulls back to the correct foothold.
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
                swing_states_[leg].landing_pos = computeLandingPos(
                    model, gait, leg, current_vel,
                    desired_linear_vel, desired_angular_vel[2]);
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
    double desired_yaw_rate) const
{
    // Current hip projected onto the ground plane. The 0.02 m z lift is
    // foot-landing clearance.
    Eigen::Vector3d base_pos = model.bodyPosition();
    base_pos.z() = 0.0;
    const Eigen::Matrix3d& R_z = model.bodyYawRotation();
    const Eigen::Vector3d hip_pos_world = base_pos + R_z * hipPos[leg_idx];
    const Eigen::Vector3d p_h(hip_pos_world.x(), hip_pos_world.y(), 0.02);

    // Half of scheduled stance time.
    const double t_stance    = gait.gait().duty_cycle * gait.gait().period;
    const double half_stance = 0.5 * t_stance;

    // Foot-placement gains. >1 makes the foot land further in the direction of
    // motion (Raibert) or applies more aggressive velocity-error correction
    // (capture).
    constexpr double K_RAIBERT = 1.0;
    constexpr double K_CAPTURE = 1.0;

    // Desired velocity rotated to world frame (cmd is body-frame).
    const Eigen::Vector3d v_des_world = R_z * desired_vel;

    // Raibert heuristic: K_RAIBERT · (T_stance/2) · v_des
    const Eigen::Vector3d raibert(K_RAIBERT * half_stance * v_des_world.x(),
                                  K_RAIBERT * half_stance * v_des_world.y(),
                                  0.0);

    // Capture point: K_CAPTURE · sqrt(z0/|g|) · (v − v_des). z0 is the CoM
    // height above the feet (|NOMINAL_HEIGHT|).
    constexpr double g_mag = 9.81;
    const double z0    = std::abs(NOMINAL_HEIGHT);
    const double k_cap = std::sqrt(z0 / g_mag);
    const Eigen::Vector3d capture(K_CAPTURE * k_cap * (current_vel.x() - v_des_world.x()),
                                  K_CAPTURE * k_cap * (current_vel.y() - v_des_world.y()),
                                  0.0);

    // Rotation correction: hip displacement due to commanded yaw rate over
    // T_stance/2.
    const double dtheta = desired_yaw_rate * half_stance;
    const double r_x = hip_pos_world.x() - base_pos.x();
    const double r_y = hip_pos_world.y() - base_pos.y();
    const Eigen::Vector3d rot_correction(-dtheta * r_y, dtheta * r_x, 0.0);

    return p_h + raibert + capture + rot_correction;
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
