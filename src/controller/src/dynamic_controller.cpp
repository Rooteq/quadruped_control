#include "dynamic_controller.hpp"
#include <algorithm>

namespace quadro
{

static constexpr double STAND_TAU_MAX = 50.0;  // Nm per joint — clamp against impulses at startup

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
            torques[base + j] = std::clamp(tau_leg[j] + g[base + j], -STAND_TAU_MAX, STAND_TAU_MAX);
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

    const auto& nle = model.nonlinearEffects();   // C·v + g, canonical order

    // Cache the M factorization once — reused for every swing leg this tick.
    // Only compute it if we actually have a swing leg; otherwise skip.
    bool any_swing = false;
    for (int leg = 0; leg < static_cast<int>(NUM_LEGS); ++leg)
        if (!gait.inStance(leg)) { any_swing = true; break; }

    Eigen::LDLT<Eigen::MatrixXd> M_ldlt;
    if (any_swing)
        M_ldlt.compute(model.massMatrix());

    for (int leg = 0; leg < static_cast<int>(NUM_LEGS); ++leg)
    {
        Eigen::Matrix3d J   = model.footJacobian(leg);
        size_t base = leg * JOINTS_PER_LEG;

        if (gait.inStance(leg))
        {
            // ── Stance: τ = Jᵀ·(−f_grf)  (virtual work; GRF acts FROM ground ON robot)
            const Eigen::Vector3d tau_leg = J.transpose() * -grfs[leg];

            for (size_t j = 0; j < JOINTS_PER_LEG; ++j)
                torques[base + j] = tau_leg[j];
        }
        else
        {
            // ── Swing: operational-space inertia + Jdot·v feedforward + Coriolis ──
            // Mirrors Python:
            //   Λ      = (J·M⁻¹·Jᵀ)⁻¹               (3×3)
            //   f_ff   = Λ·(a_des − Jdot·v)
            //   force  = Kp·e_p + Kd·e_v + f_ff
            //   τ_leg  = J_footᵀ·force + (C·v + g)[leg slice]
            Eigen::Vector3d p_world = model.footPosition(leg);
            Eigen::Vector3d v_world = model.footVelocity(leg);

            Eigen::Vector3d pos_err = leg_targets[leg].foot_pos - p_world;
            Eigen::Vector3d vel_err = leg_targets[leg].foot_vel - v_world;

            Eigen::Matrix<double, 3, Eigen::Dynamic> J_full = model.footFullJacobianLinear(leg);
            Eigen::Vector3d Jdot_v = model.footJdotV(leg);

            Eigen::MatrixXd MinvJT = M_ldlt.solve(J_full.transpose());     // nv × 3
            Eigen::Matrix3d Lambda = (J_full * MinvJT).inverse();           // 3 × 3

            Eigen::Vector3d f_ff   = Lambda * (leg_targets[leg].foot_acc - Jdot_v);
            Eigen::Vector3d force  = Kp_ * pos_err + Kd_ * vel_err + f_ff;
            Eigen::Vector3d tau_leg = J.transpose() * force;

            for (size_t j = 0; j < JOINTS_PER_LEG; ++j)
                torques[base + j] = tau_leg[j] + nle[base + j];
        }
    }

    return torques;
}

} // namespace quadro
