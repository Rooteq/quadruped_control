#include "trajectory_generator.hpp"
#include "model.hpp"

namespace quadro
{

std::array<double, 12> TrajectoryGenerator::generate(
    const QuadroModel& model,
    const GaitScheduler& gait,
    const Eigen::Vector3d& desired_linear_vel,
    const Eigen::Vector3d& /*desired_angular_vel*/)
{
    std::array<double, 12> desired_positions{};

    for (int leg = 0; leg < static_cast<int>(NUM_LEGS); ++leg)
    {
        size_t base = leg * JOINTS_PER_LEG;

        if (gait.inStance(leg))
        {
            // On swing→stance transition, initialize stance foot position
            if (!swing_states_[leg].stance_initialized)
            {
                swing_states_[leg].stance_foot_pos = swing_states_[leg].landing_pos;
                swing_states_[leg].stance_initialized = true;
            }

            // [TEMPORARY] Drift foot opposite to velocity to fake walking
            fakeStanceWalk(leg, desired_linear_vel);

            // Recompute IK from drifting foot position
            ik.calcJointPositions(static_cast<LegIdx>(leg),
                                  swing_states_[leg].stance_foot_pos);
            swing_states_[leg].stance_joints = {
                ik.legs[leg].q1, ik.legs[leg].q2, ik.legs[leg].q3};

            swing_states_[leg].active = false;
            desired_positions[base + 0] = swing_states_[leg].stance_joints[0];
            desired_positions[base + 1] = swing_states_[leg].stance_joints[1];
            desired_positions[base + 2] = swing_states_[leg].stance_joints[2];
        }
        else
        {
            /* HERE, THE VELOCITY SHOULD BE THE V_COM - IT SHOULD BE THE RESULT OF MPC WORKING, NOT DESIRED VEL*/
            /* So desired velocity causes mpc to generate forces that propel the body in this direction, legs react*/

            // Detect swing entry — initialize trajectory once
            swing_states_[leg].stance_initialized = false;
            if (!swing_states_[leg].active)
            {
                swing_states_[leg].liftoff_pos = model.footPosition(leg);
                swing_states_[leg].landing_pos = computeLandingPos(
                    model, gait, leg, desired_linear_vel);
                swing_states_[leg].active = true;
            }

            double phase = gait.swingPhase(leg);  // [0, 1]
            Eigen::Vector3d foot_target = evaluateSwing(swing_states_[leg], phase);

            // Convert body-frame foot position to hip-local for IK
            Eigen::Vector3d hip_pos = hipPos[leg];


            Eigen::Vector3d foot_pose = foot_target;

            // ik.calcJointPositions(static_cast<LegIdx>(leg),
            //                       foot_local.x(), foot_local.y(), foot_local.z());
            ik.calcJointPositions(static_cast<LegIdx>(leg),
                                  foot_pose);

            desired_positions[base + 0] = ik.legs[leg].q1;
            desired_positions[base + 1] = ik.legs[leg].q2;
            desired_positions[base + 2] = ik.legs[leg].q3;
        }
    }

    return desired_positions;
}

Eigen::Vector3d TrajectoryGenerator::computeLandingPos(
    const QuadroModel& model,
    const GaitScheduler& gait,
    int leg_idx,
    const Eigen::Vector3d& body_velocity) const
{
    // Raibert heuristic: land under hip + velocity compensation
    // Eigen::Vector3d hip_pos = model.hipPosition(leg_idx);
    Eigen::Vector3d hip_pos = hipPos[leg_idx];

    // Project hip onto nominal ground plane in body frame
    Eigen::Vector3d hip_ground = hip_pos;
    hip_ground.z() = NOMINAL_HEIGHT;

    double stance_duration = gait.gait().duty_cycle * gait.gait().period;
    Eigen::Vector3d landing = hip_ground + body_velocity * (stance_duration / 2.0);
    landing.z() = NOMINAL_HEIGHT;

    return landing;
}

Eigen::Vector3d TrajectoryGenerator::evaluateSwing(
    const SwingState& state, double phase) const
{
    // Smooth-step (Hermite basis): zero velocity at phase=0 and phase=1
    double s = phase * phase * (3.0 - 2.0 * phase);

    // XY: smooth interpolation from liftoff to landing
    double x = state.liftoff_pos.x() + s * (state.landing_pos.x() - state.liftoff_pos.x());
    double y = state.liftoff_pos.y() + s * (state.landing_pos.y() - state.liftoff_pos.y());

    // Z: cubic Bezier with 4 control points for smooth lift-and-place
    //   P0 = liftoff_z       (start on ground, vel=0)
    //   P1 = liftoff_z + h   (tangent pulls up → zero horizontal vel at start)
    //   P2 = landing_z + h   (tangent pulls up → zero horizontal vel at end)
    //   P3 = landing_z       (end on ground, vel=0)
    double t = phase;
    double t2 = t * t;
    double t3 = t2 * t;
    double mt = 1.0 - t;
    double mt2 = mt * mt;
    double mt3 = mt2 * mt;

    double p0 = state.liftoff_pos.z();
    double p1 = state.liftoff_pos.z() + state.apex_height;
    double p2 = state.landing_pos.z() + state.apex_height;
    double p3 = state.landing_pos.z();

    double z = mt3 * p0 + 3.0 * mt2 * t * p1 + 3.0 * mt * t2 * p2 + t3 * p3;

    return {x, y, z};
}

void TrajectoryGenerator::fakeStanceWalk(int leg_idx, const Eigen::Vector3d& desired_linear_vel)
{
    // Move foot opposite to desired velocity (body moves forward → foot slides back)
    swing_states_[leg_idx].stance_foot_pos.x() -= desired_linear_vel.x() * dt_;
    swing_states_[leg_idx].stance_foot_pos.y() -= desired_linear_vel.y() * dt_;
    // Z stays at NOMINAL_HEIGHT — foot stays on ground
}

} // namespace quadro
