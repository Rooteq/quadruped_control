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
    
    // Convert to world frame since our model generates local coordinates
    const Eigen::Matrix3d& R_b_w = model.bodyToWorldRotation();
    const Eigen::VectorXd& state = model.stateVector();
    Eigen::Vector3d base_pos_w(state[3], state[4], state[5]);
    Eigen::Vector3d base_vel_w(state[9], state[10], state[11]);

    for (int leg = 0; leg < static_cast<int>(NUM_LEGS); ++leg)
    {
        Eigen::Matrix3d J   = model.footJacobian(leg);
        size_t base = leg * JOINTS_PER_LEG;

        if (gait.inStance(leg))
        {
            // ── Stance: τ = Jᵀ f  (Newton-Euler quasi-static) ─────
            // J is in LOCAL_WORLD_ALIGNED frame (world axes), GRF is in world frame
            const Eigen::Vector3d tau_leg = J.transpose() * -grfs[leg];

            for (size_t j = 0; j < JOINTS_PER_LEG; ++j)
                torques[base + j] = tau_leg[j];
        }
        else
        {
            // ── Swing: Cartesian PD + gravity compensation ────────────────
            // With a FreeFlyer model, pinocchio outputs foot kinematics in the world frame
            Eigen::Vector3d p_world = model.footPosition(leg);
            Eigen::Vector3d v_world = model.footVelocity(leg);

            Eigen::Vector3d pos_err_w = leg_targets[leg].foot_pos - p_world;
            Eigen::Vector3d vel_err_w = leg_targets[leg].foot_vel - v_world;

            Eigen::Vector3d force_w   = Kp_ * pos_err_w + Kd_ * vel_err_w;
            Eigen::Vector3d tau_leg   = J.transpose() * force_w;

            for (size_t j = 0; j < JOINTS_PER_LEG; ++j)
                torques[base + j] = tau_leg[j] + g[base + j];
        }
    }

    return torques;
}

} // namespace quadro
