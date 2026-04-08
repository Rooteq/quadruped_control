#include "dynamic_controller.hpp"

namespace quadro
{

std::array<double, NUM_JOINTS> DynamicController::computeTorques(
    const QuadroModel& model,
    const GaitScheduler& gait,
    const std::array<double, NUM_JOINTS>& desired_positions,
    const std::array<Eigen::Vector3d, NUM_LEGS>& grfs)
{
    std::array<double, NUM_JOINTS> torques{};

    const auto& q  = model.jointPositions();
    const auto& dq = model.jointVelocities();
    const auto& g  = model.gravityCompensation();

    // Rotation from world frame to base frame: R = Rz*Ry*Rx, R^T maps world→base.
    // GRFs from MPC are in world frame; Jacobian is in base frame.
    const auto& x = model.stateVector();
    const Eigen::Matrix3d R =
        (Eigen::AngleAxisd(x[2], Eigen::Vector3d::UnitZ()) *
         Eigen::AngleAxisd(x[1], Eigen::Vector3d::UnitY()) *
         Eigen::AngleAxisd(x[0], Eigen::Vector3d::UnitX())).toRotationMatrix();

    for (int leg = 0; leg < static_cast<int>(NUM_LEGS); ++leg)
    {
        const int base = leg * static_cast<int>(JOINTS_PER_LEG);

        if (gait.inStance(leg))
        {
            // ── Stance: τ = Jᵀ Rᵀ f ──────────────────────────────────────
            // The MPC GRF already accounts for gravity, so no extra gravity
            // compensation is needed for stance joints.
            const Eigen::Matrix<double, 3, 3> J_leg =
                model.footJacobianLinear(leg).block<3, 3>(0, base);

            const Eigen::Vector3d f_base = R.transpose() * grfs[leg];
            const Eigen::Vector3d tau_leg = J_leg.transpose() * f_base;

            for (int j = 0; j < static_cast<int>(JOINTS_PER_LEG); ++j)
                torques[base + j] = tau_leg[j];
        }
        else
        {
            // ── Swing: PD + gravity compensation ─────────────────────────
            for (int j = 0; j < static_cast<int>(JOINTS_PER_LEG); ++j)
            {
                const int idx = base + j;
                torques[idx] = kp_[idx] * (desired_positions[idx] - q[idx])
                             + kd_[idx] * (0.0 - dq[idx])
                             + g[idx];
            }
        }
    }

    return torques;
}

} // namespace quadro
