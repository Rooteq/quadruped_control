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
            // Stance: hold current joint positions (later replaced by MPC torques)
            swing_states_[leg].active = false;
            desired_positions[base + 0] = model.jointPositions()[base + 0];
            desired_positions[base + 1] = model.jointPositions()[base + 1];
            desired_positions[base + 2] = model.jointPositions()[base + 2];
        }
        else
        {

            /* HERE, THE VELOCITY SHOULD BE THE V_COM - IT SHOULD BE THE RESULT OF MPC WORKING, NOT DESIRED VEL*/
            /* So desired velocity causes mpc to generate forces that propel the body in this direction, legs react*/

            // Detect swing entry — initialize trajectory once
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


            // Eigen::Vector3d foot_local = foot_target - hip_pos;
            // Eigen::Vector3d foot_local = foot_target - hip_pos;
            // Eigen::Vector3d foot_target = Eigen::Vector3d(0.112, -0.188, -0.27);
            // Eigen::Vector3d foot_pose = Eigen::Vector3d(0.0, 0.0, -0.27) + hip_pos;
            // Eigen::Vector3d foot_pose = foot_target + hip_pos;
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
    Eigen::Vector3d pos;

    // XY: linear interpolation from liftoff to landing
    pos.x() = state.liftoff_pos.x() + phase * (state.landing_pos.x() - state.liftoff_pos.x());
    pos.y() = state.liftoff_pos.y() + phase * (state.landing_pos.y() - state.liftoff_pos.y());

    // Z: linear interp ground level + half-sine arc for clearance
    double ground_z = state.liftoff_pos.z()
                      + phase * (state.landing_pos.z() - state.liftoff_pos.z());
    pos.z() = ground_z + state.apex_height * std::sin(M_PI * phase);

    return pos;
}

} // namespace quadro
