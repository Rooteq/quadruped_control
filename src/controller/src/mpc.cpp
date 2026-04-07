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
    const int k  = HORIZON_STEPS;
    const int nf = static_cast<int>(NUM_LEGS);

    // ── 1. Stack reference trajectory ──────────────────────────────────────
    for (int i = 0; i < k; ++i)
        X_ref_qp_.segment<N_STATE>(i * N_STATE) = x_ref_[i];

    // ── 2. Build Aqp (N_PRED × N_STATE): row-block i = Ad^(i+1) ───────────
    {
        Eigen::Matrix<double, N_STATE, N_STATE> Ad_pow = Ad_;
        for (int i = 0; i < k; ++i)
        {
            Aqp_.block<N_STATE, N_STATE>(i * N_STATE, 0) = Ad_pow;
            Ad_pow = Ad_ * Ad_pow;
        }
    }

    // ── 3. Build Bqp (N_PRED × N_VAR): lower-triangular Toeplitz ──────────
    // Bqp[i, j] = Ad^(i-j) · Bd[j]  for i >= j, else 0
    Bqp_.setZero();
    for (int j = 0; j < k; ++j)
    {
        Eigen::Matrix<double, N_STATE, N_FORCE> col = Bd_[j];
        for (int i = j; i < k; ++i)
        {
            Bqp_.block<N_STATE, N_FORCE>(i * N_STATE, j * N_FORCE) = col;
            col = Ad_ * col;
        }
    }

    // ── 4. Build diagonal cost vector L_diag (Q_WEIGHTS tiled k times) ────
    for (int i = 0; i < k; ++i)
        for (int s = 0; s < N_STATE; ++s)
            L_diag_[i * N_STATE + s] = Q_WEIGHTS[s];

    // ── 5. Build QP cost H and g ───────────────────────────────────────────
    // H = 2*(BqpᵀLBqp + α·I)  — positive-definite by construction
    // g = 2·Bqpᵀ·L·(Aqp·x0 − X_ref)
    const Eigen::Matrix<double, N_PRED, 1>  residual = Aqp_ * x0_ - X_ref_qp_;
    // L is diagonal so L*v = L_diag.cwiseProduct(v)
    const Eigen::Matrix<double, N_PRED, 1>  L_res    = L_diag_.cwiseProduct(residual);
    // Bqpᵀ·L·Bqp  using the diagonal structure: each row j of Bqp is scaled by L_diag[j]
    Eigen::Matrix<double, N_VAR, N_PRED>    BqpT_L   = Bqp_.transpose();
    for (int j = 0; j < N_PRED; ++j)
        BqpT_L.col(j) *= L_diag_[j];

    H_    = 2.0 * (BqpT_L * Bqp_ + alpha_ * Eigen::Matrix<double, N_VAR, N_VAR>::Identity());
    g_qp_ = 2.0 * Bqp_.transpose() * L_res;

    H_qp_ = H_;  // column-major → row-major copy for qpOASES

    // ── 6. Build constraint matrix C and bounds ────────────────────────────
    // Friction pyramid (4 rows per foot per step): [±1, 0, -μ]·f ≤ 0
    //                                              [0, ±1, -μ]·f ≤ 0
    // Swing feet: forced to zero via variable bounds (lb = ub = 0).
    C_.setZero();
    lb_.fill(-1e12);
    ub_.fill( 1e12);
    lbC_.fill(-1e12);
    ubC_.fill(0.0);   // all cone rows: C·u ≤ 0

    for (int n = 0; n < k; ++n)
    {
        for (int i = 0; i < nf; ++i)
        {
            const int vc = n * N_FORCE + 3 * i;  // variable column offset (fx)
            const int cr = (n * nf + i) * 4;     // constraint row offset

            if (!contact_schedule_[n][i])
            {
                // Swing: pin all three force components to zero
                lb_[vc + 0] = ub_[vc + 0] = 0.0;
                lb_[vc + 1] = ub_[vc + 1] = 0.0;
                lb_[vc + 2] = ub_[vc + 2] = 0.0;
                // Trivial constraints (C rows stay zero → satisfied for any u)
                lbC_[cr + 0] = ubC_[cr + 0] = 0.0;
                lbC_[cr + 1] = ubC_[cr + 1] = 0.0;
                lbC_[cr + 2] = ubC_[cr + 2] = 0.0;
                lbC_[cr + 3] = ubC_[cr + 3] = 0.0;
            }
            else
            {
                // Stance: normal force bounded, tangential forces constrained by cone
                lb_[vc + 2] = fz_min_;
                ub_[vc + 2] = fz_max_;

                //  fx − μ·fz ≤ 0
                C_(cr + 0, vc + 0) =  1.0;  C_(cr + 0, vc + 2) = -mu_;
                // −fx − μ·fz ≤ 0
                C_(cr + 1, vc + 0) = -1.0;  C_(cr + 1, vc + 2) = -mu_;
                //  fy − μ·fz ≤ 0
                C_(cr + 2, vc + 1) =  1.0;  C_(cr + 2, vc + 2) = -mu_;
                // −fy − μ·fz ≤ 0
                C_(cr + 3, vc + 1) = -1.0;  C_(cr + 3, vc + 2) = -mu_;
            }
        }
    }

    C_qp_ = C_;  // column-major → row-major copy for qpOASES

    // ── 7. Solve QP ────────────────────────────────────────────────────────
    // H and A both change every step (changing Bqp / contact schedule), so we
    // call init() each cycle. qpOASES resets its internal state on each init().
    qp_.setPrintLevel(qpOASES::PL_NONE);
    qpOASES::int_t nWSR = 200;
    const qpOASES::returnValue rv = qp_.init(
        H_qp_.data(), g_qp_.data(),
        C_qp_.data(),
        lb_.data(), ub_.data(),
        lbC_.data(), ubC_.data(),
        nWSR);

    if (rv != qpOASES::SUCCESSFUL_RETURN && rv != qpOASES::RET_MAX_NWSR_REACHED)
    {
        for (auto& f : grfs_) f.setZero();
        return;
    }

    // ── 8. Extract GRFs for the first horizon step ────────────────────────
    // U* = [f0, f1, f2, f3, ...] stacked by step — only first 12 elements used
    Eigen::Matrix<double, N_VAR, 1> u_opt;
    qp_.getPrimalSolution(u_opt.data());

    for (int i = 0; i < nf; ++i)
        grfs_[i] = u_opt.segment<3>(3 * i);
}

} // namespace quadro
