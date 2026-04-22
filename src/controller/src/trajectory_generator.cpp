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
    const Eigen::Vector3d& /*desired_angular_vel*/,
    const Eigen::Vector3d& current_vel)
{
    std::array<LegTarget, NUM_LEGS> targets{};

    const Eigen::VectorXd& state = model.stateVector();
    Eigen::Vector3d base_pos_w(state[3], state[4], state[5]);
    Eigen::Vector3d base_vel_w(state[9], state[10], state[11]);
    const Eigen::Matrix3d& R_b_w = model.bodyToWorldRotation();

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
                    model, gait, leg, current_vel, desired_linear_vel);
                swing_states_[leg].active = true;
            }

            double phase = gait.swingPhase(leg);  // [0, 1]
            targets[leg].foot_pos = evaluateSwing(swing_states_[leg], phase);
            targets[leg].foot_vel = evaluateSwingVelocity(swing_states_[leg], phase, phase_rate);
        }
    }

    return targets;
}

Eigen::Vector3d TrajectoryGenerator::computeLandingPos(
    const QuadroModel& model,
    const GaitScheduler& gait,
    int leg_idx,
    const Eigen::Vector3d& current_vel,
    const Eigen::Vector3d& desired_vel) const
{
    // Hip position in world frame: body-frame hardcoded offset rotated by yaw-only R_z.
    // These hipPos values are the validated Raibert reference points for this robot.
    const Eigen::VectorXd& state = model.stateVector();
    Eigen::Vector3d base_pos(state[3], state[4], 0.0);  // z=0: project onto ground plane
    Eigen::Matrix3d R_z = model.bodyYawRotation();
    Eigen::Vector3d hip_pos_world = base_pos + R_z * hipPos[leg_idx];

    double t_swing  = (1.0 - gait.gait().duty_cycle) * gait.gait().period;
    double t_stance = gait.gait().duty_cycle * gait.gait().period;
    double T        = t_swing + 0.5 * t_stance;
    double pred_time = T / 2.0;

    double k_v_x = 0.4 * T;
    double k_v_y = 0.2 * T;

    // Nominal: directly below hip on the ground plane
    Eigen::Vector3d pos_nominal(hip_pos_world.x(), hip_pos_world.y(), 0.02);
    Eigen::Vector3d drift(desired_vel.x() * pred_time, desired_vel.y() * pred_time, 0.0);
    Eigen::Vector3d vel_correction(k_v_x * (current_vel.x() - desired_vel.x()),
                                   k_v_y * (current_vel.y() - desired_vel.y()),
                                   0.0);

    return pos_nominal + drift + vel_correction;
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

} // namespace quadro
