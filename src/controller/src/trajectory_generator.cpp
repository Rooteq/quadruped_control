#include "trajectory_generator.hpp"
#include "model.hpp"

namespace quadro
{

std::array<double, 12> TrajectoryGenerator::generate(
    const QuadroModel& model,
    const GaitScheduler& gait,
    const Eigen::Vector3d& desired_linear_vel,
    const Eigen::Vector3d& desired_angular_vel)
{
    std::array<double, 12> desired_positions{};

    for (int leg = 0; leg < static_cast<int>(NUM_LEGS); ++leg)
    {
        size_t base = leg * JOINTS_PER_LEG;

        if (gait.inStance(leg))
        {
            // Stance: hold current joint positions (later replaced by MPC torques)
            desired_positions[base + 0] = model.jointPositions()[base + 0];
            desired_positions[base + 1] = model.jointPositions()[base + 1];
            desired_positions[base + 2] = model.jointPositions()[base + 2];
        }
        else
        {
            // Swing: compute foot target via IK
            // TODO: use swingPhase to interpolate along a swing arc (liftoff → apex → landing)
            // For now: move foot to default standing position below hip
            double swing_phase = gait.swingPhase(leg); // -1 if leg in stance
            (void)swing_phase;  // will be used for trajectory interpolation

            ik.calcJointPositions(static_cast<LegIdx>(leg), 0.0, 0.0, -0.27);

            desired_positions[base + 0] = ik.legs[leg].q1;
            desired_positions[base + 1] = ik.legs[leg].q2;
            desired_positions[base + 2] = ik.legs[leg].q3;
        }
    }

    return desired_positions;
}

} // namespace quadro
