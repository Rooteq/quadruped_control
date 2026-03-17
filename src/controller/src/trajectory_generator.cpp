#include "trajectory_generator.hpp"
#include "model.hpp"

namespace quadro
{

std::array<double, 12> TrajectoryGenerator::generate(
    const QuadroModel& model,
    const Eigen::Vector3d& desired_linear_vel,
    const Eigen::Vector3d& desired_angular_vel)
{
    std::array<double, 12> desired_positions{};  // zeros for now

    ik.calcJointPositions(LegIdx::FL, 0.0, 0.0, -0.27);

    desired_positions[FL_HIP] = ik.legs[FL].q1;
    desired_positions[FL_KNEE] = ik.legs[FL].q2;
    desired_positions[FL_ANKLE] = ik.legs[FL].q3;
    // ik.legs[FL];

    return desired_positions;
}

} // namespace quadro
