#include "dynamic_controller.hpp"

namespace quadro
{

std::array<double, NUM_JOINTS> DynamicController::computeStand(
    const QuadroModel& model,
    const std::array<LegTarget, NUM_LEGS>& leg_targets)
{
    std::array<double, NUM_JOINTS> torques{};
    const auto& g = model.gravityCompensation();

    for (int leg = 0; leg < static_cast<int>(NUM_LEGS); ++leg)
    {
        Eigen::Matrix3d J   = model.footJacobian(leg);
        Eigen::Vector3d p   = model.footPosition(leg);
        Eigen::Vector3d v   = model.footVelocity(leg);

        Eigen::Vector3d pos_err = leg_targets[leg].foot_pos - p;
        Eigen::Vector3d vel_err = leg_targets[leg].foot_vel - v;

        Eigen::Vector3d force   = Kp_stand_ * pos_err + Kd_stand_ * vel_err;
        Eigen::Vector3d tau_leg = J.transpose() * force;

        size_t base = leg * JOINTS_PER_LEG;
        for (size_t j = 0; j < JOINTS_PER_LEG; ++j)
            torques[base + j] = tau_leg[j] + g[base + j];
    }

    return torques;
}

std::array<double, NUM_JOINTS> DynamicController::computeTorques(
    const QuadroModel& model,
    const GaitScheduler& gait,
    const std::array<LegTarget, NUM_LEGS>& leg_targets,
    const std::array<Eigen::Vector3d, NUM_LEGS>& grfs)
{
    std::array<double, NUM_JOINTS> torques{};

    const auto& g  = model.gravityCompensation();

    // GRFs from MPC are in world frame; Jacobian is in base frame.
    const auto& x = model.stateVector();
    const Eigen::Matrix3d R =
        (Eigen::AngleAxisd(x[2], Eigen::Vector3d::UnitZ()) *
         Eigen::AngleAxisd(x[1], Eigen::Vector3d::UnitY()) *
         Eigen::AngleAxisd(x[0], Eigen::Vector3d::UnitX())).toRotationMatrix();

    for (int leg = 0; leg < static_cast<int>(NUM_LEGS); ++leg)
    {
        Eigen::Matrix3d J   = model.footJacobian(leg);
        size_t base = leg * JOINTS_PER_LEG;

        if (gait.inStance(leg))
        {
            // ── Stance: τ = Jᵀ Rᵀ f ──────────────────────────────────────
            // The MPC GRF already accounts for gravity, so no extra gravity
            // compensation is needed for stance joints.
            const Eigen::Vector3d f_base = R.transpose() * grfs[leg];
            const Eigen::Vector3d tau_leg = J.transpose() * f_base;

            for (size_t j = 0; j < JOINTS_PER_LEG; ++j)
                torques[base + j] = tau_leg[j];
        }
        else
        {
            // ── Swing: Cartesian PD + gravity compensation ────────────────
            Eigen::Vector3d p   = model.footPosition(leg);
            Eigen::Vector3d v   = model.footVelocity(leg);

            Eigen::Vector3d pos_err = leg_targets[leg].foot_pos - p;
            Eigen::Vector3d vel_err = leg_targets[leg].foot_vel - v;

            Eigen::Vector3d force   = Kp_ * pos_err + Kd_ * vel_err;
            Eigen::Vector3d tau_leg = J.transpose() * force;

            for (size_t j = 0; j < JOINTS_PER_LEG; ++j)
                torques[base + j] = tau_leg[j] + g[base + j];
        }
    }

    return torques;
}

} // namespace quadro
