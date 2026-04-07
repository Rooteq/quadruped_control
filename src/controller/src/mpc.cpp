#include "mpc.hpp"

namespace quadro
{

void MPC::update(const QuadroModel& model,
                 const Eigen::Vector3d& angular_vel_cmd,
                 const Eigen::Vector3d& linear_vel_cmd,
                 const std::array<Eigen::Matrix<double, 13, 1>, HORIZON_STEPS>& x_ref,
                 const GaitScheduler& gait_scheduler)
{
    x0_           = model.stateVector();
    mass_         = model.mass();
    body_inertia_ = model.bodyInertia();

    for (int i = 0; i < static_cast<int>(NUM_LEGS); ++i)
        foot_positions_[i] = model.footPosition(i);

    angular_vel_cmd_  = angular_vel_cmd;
    linear_vel_cmd_   = linear_vel_cmd;
    x_ref_            = x_ref;
    contact_schedule_ = gait_scheduler.contactTable<HORIZON_STEPS>(MPC_DT);
}

void MPC::calculateDynamicsMatrices()
{
    // ── Ac ────────────────────────────────────────────────────────
    // From paper eq. (16), using average yaw over the horizon (Section IV-C).
    // Ac depends only on ψ — averaging avoids recomputing per step.
    //
    // State layout: x = [φ θ ψ | px py pz | ωx ωy ωz | vx vy vz | -g]
    //               idx  0 1 2    3  4  5    6  7  8    9 10 11   12
    //
    // Non-zero blocks:
    //   Θ̇   = Rz(ψ)ᵀ · ω       →  Ac[0:3, 6:9]  = Rz(ψ)ᵀ
    //   ṗ   = v                  →  Ac[3:6, 9:12] = I₃
    //   p̈_z += x[12]            →  Ac[11, 12]    = 1   (gravity state: x[12] = -g)

    double avg_yaw = 0.0;
    for (int i = 0; i < HORIZON_STEPS; ++i)
        avg_yaw += x_ref_[i][2];
    avg_yaw /= HORIZON_STEPS;

    const double cy = std::cos(avg_yaw);
    const double sy = std::sin(avg_yaw);

    // Rz(ψ)ᵀ — transpose of yaw rotation matrix (eq. 12)
    Eigen::Matrix3d Rz_T;
    Rz_T <<  cy, sy, 0.0,
            -sy, cy, 0.0,
             0.0, 0.0, 1.0;

    Ac_.setZero();
    Ac_.block<3, 3>(0, 6) = Rz_T;                        // Θ̇ = Rz^T · ω
    Ac_.block<3, 3>(3, 9) = Eigen::Matrix3d::Identity(); // ṗ = v
    Ac_(11, 12)           = 1.0;                          // p̈_z += gravity state

    // ── Ad (first-order ZOH: Ad ≈ I + Ac·dt) ──────────────────────
    // Accurate enough at 30 Hz; replace with matrix exponential if needed.
    Ad_ = Eigen::Matrix<double, 13, 13>::Identity() + Ac_ * MPC_DT;

    // ── Bc[n] and Bd[n] — per horizon step ───────────────────────
    // Bc depends on per-step yaw ψ[n] (rotates inertia) and contact schedule
    // (zeros out swing-foot columns). From eq. (16):
    //
    //   Bc rows 6-8  (ω):  Î[n]⁻¹ · [rᵢ]×   for each stance foot i
    //   Bc rows 9-11 (ṗ):  I₃ / m             for each stance foot i
    //   All other rows:    0

    const Eigen::Matrix3d I3_over_m = Eigen::Matrix3d::Identity() / mass_;

    for (int n = 0; n < HORIZON_STEPS; ++n)
    {
        Bc_[n].setZero();

        // World-frame inertia at step n: Î = Rz(ψ[n]) · _BI · Rz(ψ[n])ᵀ  (eq. 15)
        const double psi = x_ref_[n][2];
        const double cpsi = std::cos(psi), spsi = std::sin(psi);

        Eigen::Matrix3d Rz;
        Rz << cpsi, -spsi, 0.0,
              spsi,  cpsi, 0.0,
               0.0,   0.0, 1.0;

        const Eigen::Matrix3d I_hat     = Rz * body_inertia_ * Rz.transpose();
        const Eigen::Matrix3d I_hat_inv = I_hat.inverse();

        // Predicted COM position at step n from reference trajectory (x_ref_[n][3:6]).
        // Stance feet are fixed on the ground, so ri evolves as the COM moves forward.
        // Swing feet: their columns are zeroed below — ri is unused for them.
        const Eigen::Vector3d com_pos_n(x_ref_[n][3], x_ref_[n][4], x_ref_[n][5]);

        for (int i = 0; i < static_cast<int>(NUM_LEGS); ++i)
        {
            if (!contact_schedule_[n][i]) continue;  // swing: columns stay zero

            // Vector from predicted COM to foot contact point (world frame)
            const Eigen::Vector3d r = foot_positions_[i] - com_pos_n;

            // Angular rows (6-8): Î⁻¹ · [r]×
            Bc_[n].block<3, 3>(6, 3 * i) = I_hat_inv * skewSymmetric(r);

            // Linear rows (9-11): I₃ / m
            Bc_[n].block<3, 3>(9, 3 * i) = I3_over_m;
        }

        // Bd[n] = Bc[n] · dt  (first-order ZOH, consistent with Ad)
        Bd_[n] = Bc_[n] * MPC_DT;
    }
}

Eigen::Matrix3d MPC::skewSymmetric(const Eigen::Vector3d& v)
{
    Eigen::Matrix3d S;
    S <<   0.0, -v[2],  v[1],
          v[2],   0.0, -v[0],
         -v[1],  v[0],   0.0;
    return S;
}

void MPC::run()
{
    // Zero all GRFs — QP solve will populate these once implemented.
    // Swing legs stay zero; stance legs will receive forces from QP solution U*.
    for (auto& f : grfs_)
        f.setZero();
}

} // namespace quadro
