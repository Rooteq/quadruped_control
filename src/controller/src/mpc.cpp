#include "mpc.hpp"

namespace quadro
{

void MPC::update(const QuadroModel& model,
                 const Eigen::Vector3d& angular_vel_cmd,
                 const Eigen::Vector3d& linear_vel_cmd,
                 const std::array<Eigen::Matrix<double, 13, 1>, HORIZON_STEPS>& x_ref,
                 const GaitScheduler& gait_scheduler,
                 const std::array<Eigen::Vector3d, NUM_LEGS>& foot_positions)
{
    x0_           = model.stateVector();
    mass_         = model.mass();
    body_inertia_ = model.bodyInertia();

    for (int i = 0; i < static_cast<int>(NUM_LEGS); ++i)
        foot_positions_[i] = foot_positions[i];

    angular_vel_cmd_  = angular_vel_cmd;
    linear_vel_cmd_   = linear_vel_cmd;
    x_ref_            = x_ref;
    contact_schedule_ = gait_scheduler.contactTable<HORIZON_STEPS>(MPC_DT);
}

void MPC::calculateDynamicsMatrices()
{
    // ── Ac ────────────────────────────────────────────────────────
    // State layout: x = [φ θ ψ | px py pz | ωx ωy ωz | vx vy vz | -g]
    //               idx  0 1 2    3  4  5    6  7  8    9 10 11   12
    // Average yaw over horizon for Ac (Section IV-C of MPC paper).
    double avg_yaw = 0.0;
    for (int i = 0; i < HORIZON_STEPS; ++i)
        avg_yaw += x_ref_[i][2];
    avg_yaw /= HORIZON_STEPS;

    const double cy = std::cos(avg_yaw);
    const double sy = std::sin(avg_yaw);

    Eigen::Matrix3d Rz_T;
    Rz_T <<  cy, sy, 0.0,
            -sy, cy, 0.0,
             0.0, 0.0, 1.0;

    Ac_.setZero();
    Ac_.block<3, 3>(0, 6) = Rz_T;                        // Θ̇ = Rz^T · ω
    Ac_.block<3, 3>(3, 9) = Eigen::Matrix3d::Identity(); // ṗ = v
    Ac_(11, 12)           = 1.0;                          // p̈_z += gravity state

    // ── Ad: matrix exponential ZOH ────────────────────────────────
    Ad_ = (Ac_ * MPC_DT).exp();

    // ── Bc[n] and Bd[n] — per horizon step ───────────────────────
    // r = foot_world - ref_com_world_at_n : both in world frame, matches Python.
    // Rz_T is built from the DESIRED yaw at each horizon step (x_ref_[n][2]),
    // not the actual current yaw — the rotation changes as the robot turns.
    const Eigen::Matrix3d& I_hat     = body_inertia_;
    const Eigen::Matrix3d  I_hat_inv = I_hat.inverse();
    const Eigen::Matrix3d  I3_over_m = Eigen::Matrix3d::Identity() / mass_;
    const double half_dt2 = 0.5 * MPC_DT * MPC_DT;

    for (int n = 0; n < HORIZON_STEPS; ++n)
    {
        Bc_[n].setZero();
        Bd_[n].setZero();

        // Desired yaw at this horizon step → Rz^T for RPY rows of Bd
        const double psi_n = x_ref_[n][2];
        const double cp_n  = std::cos(psi_n), sp_n = std::sin(psi_n);
        Eigen::Matrix3d Rz_n_T;
        Rz_n_T <<  cp_n,  sp_n, 0.0,
                  -sp_n,  cp_n, 0.0,
                    0.0,   0.0, 1.0;

        // Reference COM at this step — foot lever arm in world frame
        const Eigen::Vector3d com_pos_n(x_ref_[n][3], x_ref_[n][4], x_ref_[n][5]);

        for (int i = 0; i < static_cast<int>(NUM_LEGS); ++i)
        {
            if (!contact_schedule_[n][i]) continue;

            // r = foot_world − ref_COM_world_at_n  (world frame, matches Python)
            const Eigen::Vector3d r        = foot_positions_[i] - com_pos_n;
            const Eigen::Matrix3d I_inv_sk = I_hat_inv * skewSymmetric(r);

            // Bc: continuous-time input map
            Bc_[n].block<3, 3>(6, 3 * i) = I_inv_sk;   // Δω rows
            Bc_[n].block<3, 3>(9, 3 * i) = I3_over_m;  // Δv rows

            // Bd: ZOH with 2nd-order terms
            Bd_[n].block<3, 3>(6, 3 * i) = MPC_DT * I_inv_sk;
            Bd_[n].block<3, 3>(9, 3 * i) = MPC_DT * I3_over_m;
            Bd_[n].block<3, 3>(0, 3 * i) = half_dt2 * Rz_n_T * I_inv_sk;  // RPY 2nd-order
            Bd_[n].block<3, 3>(3, 3 * i) = half_dt2 * I3_over_m;           // pos 2nd-order
        }
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

    // ── 7. Invalidate warm start on contact change ─────────────────────────
    // When a leg switches stance↔swing the QP active-set structure changes.
    // Initialising from the old warm start gives an infeasible primal point and
    // wastes solver iterations (or causes failure) at every gait transition.
    for (int n = 0; n < HORIZON_STEPS && has_warm_start_; ++n)
        for (int i = 0; i < static_cast<int>(NUM_LEGS) && has_warm_start_; ++i)
            if (contact_schedule_[n][i] != prev_contact_schedule_[n][i])
                has_warm_start_ = false;
    prev_contact_schedule_ = contact_schedule_;

    // ── 8. Solve QP ────────────────────────────────────────────────────────
    qp_.setPrintLevel(qpOASES::PL_NONE);
    qpOASES::int_t nWSR = 200;
    const qpOASES::returnValue rv = qp_.init(
        H_qp_.data(), g_qp_.data(),
        C_qp_.data(),
        lb_.data(), ub_.data(),
        lbC_.data(), ubC_.data(),
        nWSR,
        nullptr,
        has_warm_start_ ? u_warm_.data() : nullptr,
        has_warm_start_ ? y_warm_.data() : nullptr);

    if (rv != qpOASES::SUCCESSFUL_RETURN && rv != qpOASES::RET_MAX_NWSR_REACHED)
    {
        std::printf("[MPC run()] QP FAILED rv=%d  nWSR=%d  mass=%.3f\n",
                    static_cast<int>(rv), static_cast<int>(nWSR), mass_);
        for (auto& f : grfs_) f.setZero();
        return;
    }

    // ── 8. Extract GRFs and save warm start ───────────────────────────────
    Eigen::Matrix<double, N_VAR, 1> u_opt;
    qp_.getPrimalSolution(u_opt.data());
    qp_.getDualSolution(y_warm_.data());
    u_warm_         = u_opt;
    has_warm_start_ = true;

    for (int i = 0; i < nf; ++i)
        grfs_[i] = u_opt.segment<3>(3 * i);

    if (++print_counter_ % 2 == 0) {
        std::printf("[MPC GRF fz]  FL=%5.1f  FR=%5.1f  BL=%5.1f  BR=%5.1f  N  |"
                    "  pz=%.3f  vz=%.3f  mass=%.2f\n",
                    grfs_[0].z(), grfs_[1].z(), grfs_[2].z(), grfs_[3].z(),
                    x0_[5], x0_[11], mass_);
    }
}


void MPC::run_casadi()
{
    casadi::Opti opti;

    const int N = HORIZON_STEPS;
    const int NX = 13;
    const int NU = 12;
    const int nf = static_cast<int>(NUM_LEGS);

    casadi::MX X = opti.variable(NX, N);
    casadi::MX U = opti.variable(NU, N);

    casadi::MX cost = 0;

    std::vector<double> q_data(std::begin(Q_WEIGHTS), std::end(Q_WEIGHTS));
    casadi::MX Q = casadi::MX::diag(q_data);
    casadi::MX R = casadi::MX::eye(NU) * 1e-5;

    std::vector<double> Ad_data(Ad_.data(), Ad_.data() + Ad_.size());
    casadi::MX Ad_mx = casadi::MX::reshape(casadi::MX(Ad_data), NX, NX);

    std::vector<double> x0_data(x0_.data(), x0_.data() + x0_.size());
    casadi::MX X0_mx = casadi::MX(x0_data);

    for (int k = 0; k < N; ++k)
    {
        std::vector<double> x_ref_data(x_ref_[k].data(), x_ref_[k].data() + x_ref_[k].size());
        casadi::MX X_ref_k = casadi::MX(x_ref_data);

        casadi::MX state_err = X(casadi::Slice(), k) - X_ref_k;
        cost += casadi::MX::mtimes({state_err.T(), Q, state_err});
        cost += casadi::MX::mtimes({U(casadi::Slice(), k).T(), R, U(casadi::Slice(), k)});

        std::vector<double> Bd_data(Bd_[k].data(), Bd_[k].data() + Bd_[k].size());
        casadi::MX Bd_k = casadi::MX::reshape(casadi::MX(Bd_data), NX, NU);

        if (k == 0) {
            opti.subject_to(X(casadi::Slice(), k) == casadi::MX::mtimes(Ad_mx, X0_mx) + casadi::MX::mtimes(Bd_k, U(casadi::Slice(), k)));
        } else {
            opti.subject_to(X(casadi::Slice(), k) == casadi::MX::mtimes(Ad_mx, X(casadi::Slice(), k-1)) + casadi::MX::mtimes(Bd_k, U(casadi::Slice(), k)));
        }

        for (int leg = 0; leg < nf; ++leg)
        {
            auto u_leg = U(casadi::Slice(leg*3, leg*3+3), k);
            auto fx = u_leg(0);
            auto fy = u_leg(1);
            auto fz = u_leg(2);

            if (!contact_schedule_[k][leg])
            {
                opti.subject_to(fx == 0.0);
                opti.subject_to(fy == 0.0);
                opti.subject_to(fz == 0.0);
            }
            else
            {
                opti.subject_to(fz >= fz_min_);
                opti.subject_to(fz <= fz_max_);

                opti.subject_to(fx - mu_ * fz <= 0.0);
                opti.subject_to(-fx - mu_ * fz <= 0.0);
                opti.subject_to(fy - mu_ * fz <= 0.0);
                opti.subject_to(-fy - mu_ * fz <= 0.0);
            }
        }
    }

    opti.minimize(cost);

    casadi::Dict solver_opts;
    solver_opts["print_iter"] = false;
    solver_opts["print_header"] = false;
    solver_opts["error_on_fail"] = false;

    opti.solver("qrsqp", casadi::Dict(), solver_opts);

    try
    {
        casadi::OptiSol sol = opti.solve();
        std::vector<double> u_opt = std::vector<double>(sol.value(U(casadi::Slice(), 0)));

        for (int i = 0; i < nf; ++i)
        {
            grfs_[i] << u_opt[3*i], u_opt[3*i+1], u_opt[3*i+2];
        }
    }
    catch(const std::exception& e)
    {
        for (auto& f : grfs_) f.setZero();
    }

    if (++print_counter_ % 10 == 0)
        std::printf("[MPC GRF fz]  FL=%.1f  FR=%.1f  BL=%.1f  BR=%.1f  N\n",
                    grfs_[0].z(), grfs_[1].z(), grfs_[2].z(), grfs_[3].z());
}

} // namespace quadro
