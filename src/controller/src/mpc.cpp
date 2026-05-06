#include "mpc.hpp"

namespace quadro
{

void MPC::update(const QuadroModel& model,
                 const Eigen::Vector3d& angular_vel_cmd,
                 const Eigen::Vector3d& linear_vel_cmd,
                 const std::array<Eigen::Matrix<double, 13, 1>, HORIZON_STEPS>& x_ref,
                 const GaitScheduler& gait_scheduler,
                 const std::array<std::array<Eigen::Vector3d, NUM_LEGS>, HORIZON_STEPS>& levers)
{
    x0_           = model.stateVector();
    mass_         = model.mass();
    body_inertia_ = model.bodyInertia();

    levers_ = levers;

    angular_vel_cmd_  = angular_vel_cmd;
    linear_vel_cmd_   = linear_vel_cmd;
    x_ref_            = x_ref;
    contact_schedule_ = gait_scheduler.contactTable<HORIZON_STEPS>(MPC_DT);
}

void MPC::calculateDynamicsMatrices()
{
    // State layout: x = [φ θ ψ | px py pz | ωx ωy ωz | vx vy vz | -g]
    //               idx  0 1 2    3  4  5    6  7  8    9 10 11   12

    // ── Single average yaw for both Ac and Bd (matches Python) ───
    // Python builds RzT once from yaw_avg and reuses it for the rpy<-omega
    // term in Ac AND for the 2nd-order rpy ZOH term in every Bd[n]. We do the
    // same here — the previous per-step yaw in Bd was inconsistent with the
    // single-yaw Ac and diverges from the reference behaviour.
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

    // ── Ac ────────────────────────────────────────────────────────
    Ac_.setZero();
    Ac_.block<3, 3>(0, 6) = Rz_T;                        // Θ̇ = Rz^T · ω
    Ac_.block<3, 3>(3, 9) = Eigen::Matrix3d::Identity(); // ṗ = v
    Ac_(11, 12)           = 1.0;                          // v̇_z += -g  (via x[12] = -g)

    // ── Ad: Euler 1st-order + explicit 2nd-order gravity coupling ─
    // Python uses a pure Euler Ad and adds a separate gd vector for the
    // 2nd-order gravity terms (gd[0:3] = ½·g·dt² for pos, gd[6:9] = g·dt for vel).
    // In our 13-state formulation those terms live in Ad: vz gets the 1st-order
    // contribution from Ac(11,12)·dt, and pz gets the 2nd-order contribution
    // from (½·Ac²·dt²)(5,12) = ½·dt². Both then propagate the constant -g held
    // in x[12], so we don't need a separate gd vector.
    Ad_ = Eigen::Matrix<double, 13, 13>::Identity() + Ac_ * MPC_DT;
    Ad_(5, 12) = 0.5 * MPC_DT * MPC_DT;                  // pz += ½·(-g)·dt²

    // ── Bc[n] and Bd[n] — per horizon step ───────────────────────
    // r = foot_world − ref_com_world_at_n : both in world frame, matches Python.
    // Rz_T uses the SAME yaw_avg for every horizon step (matches Python).
    const Eigen::Matrix3d& I_hat     = body_inertia_;
    const Eigen::Matrix3d  I_hat_inv = I_hat.inverse();
    const Eigen::Matrix3d  I3_over_m = Eigen::Matrix3d::Identity() / mass_;
    const double half_dt2 = 0.5 * MPC_DT * MPC_DT;

    for (int n = 0; n < HORIZON_STEPS; ++n)
    {
        Bc_[n].setZero();
        Bd_[n].setZero();

        for (int i = 0; i < static_cast<int>(NUM_LEGS); ++i)
        {
            if (!contact_schedule_[n][i]) continue;

            // levers_[n][i] = foot_world − base_traj_world for this leg at step n
            // (already computed by TrajectoryGenerator::computeHorizonLevers,
            // mirrors Python's r_*_traj_world). Zero on swing steps; held
            // constant per stance phase at the value planned at takeoff.
            const Eigen::Vector3d& r        = levers_[n][i];
            const Eigen::Matrix3d  I_inv_sk = I_hat_inv * skewSymmetric(r);

            // Bc: continuous-time input map (no gravity row)
            Bc_[n].block<3, 3>(6, 3 * i) = I_inv_sk;   // ω̇ rows
            Bc_[n].block<3, 3>(9, 3 * i) = I3_over_m;  // v̇ rows

            // Bd: 1st-order ZOH for ω/v, 2nd-order ZOH for rpy/pos
            Bd_[n].block<3, 3>(6, 3 * i) = MPC_DT * I_inv_sk;
            Bd_[n].block<3, 3>(9, 3 * i) = MPC_DT * I3_over_m;
            Bd_[n].block<3, 3>(0, 3 * i) = half_dt2 * Rz_T * I_inv_sk;  // single yaw_avg RzT
            Bd_[n].block<3, 3>(3, 3 * i) = half_dt2 * I3_over_m;
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
    const int N     = HORIZON_STEPS;
    const int NX    = N_STATE;                // 13
    const int NU    = N_FORCE;                // 12
    const int NV    = N * NX + N * NU;        // 250  [X_traj | U_traj]
    const int NDYN  = N * NX;                 // 130  dynamics equality rows / U start col
    const int NFRIC = 4 * 4 * N;             // 160  friction inequality rows
    const int NCON  = NDYN + NFRIC;           // 290

    // ── 1. g: -2*Q*x_ref for state part, 0 for force part ─────────────
    std::vector<double> g_vec(NV, 0.0);
    for (int k = 0; k < N; ++k)
        for (int i = 0; i < NX; ++i)
            g_vec[k * NX + i] = -2.0 * Q_WEIGHTS[i] * x_ref_[k][i];

    // ── 2. A: dynamics equality (bidiagonal) + friction pyramid ────────
    // Always include ALL positions regardless of contact (constant sparsity).
    std::vector<casadi_int> A_rows, A_cols;
    std::vector<double>     A_vals;
    A_rows.reserve(3500); A_cols.reserve(3500); A_vals.reserve(3500);

    for (int k = 0; k < N; ++k)
    {
        for (int i = 0; i < NX; ++i) {              // I on X_k diagonal
            A_rows.push_back(k*NX + i);
            A_cols.push_back(k*NX + i);
            A_vals.push_back(1.0);
        }
        if (k > 0)
            for (int i = 0; i < NX; ++i)            // -Ad on X_{k-1} sub-diagonal
                for (int j = 0; j < NX; ++j) {
                    A_rows.push_back(k*NX + i);
                    A_cols.push_back((k-1)*NX + j);
                    A_vals.push_back(-Ad_(i, j));
                }
        for (int i = 0; i < NX; ++i)                // -Bd_k on U_k block
            for (int j = 0; j < NU; ++j) {
                A_rows.push_back(k*NX + i);
                A_cols.push_back(NDYN + k*NU + j);
                A_vals.push_back(-Bd_[k](i, j));
            }
    }
    for (int k = 0; k < N; ++k)                     // friction: 4 rows/leg/step
        for (int leg = 0; leg < 4; ++leg) {
            const int fc = NDYN + k*NU + 3*leg;      // fx col
            const int r0 = NDYN + (k*4 + leg)*4;
            A_rows.push_back(r0);   A_cols.push_back(fc);   A_vals.push_back( 1.0);
            A_rows.push_back(r0);   A_cols.push_back(fc+2); A_vals.push_back(-mu_);
            A_rows.push_back(r0+1); A_cols.push_back(fc);   A_vals.push_back(-1.0);
            A_rows.push_back(r0+1); A_cols.push_back(fc+2); A_vals.push_back(-mu_);
            A_rows.push_back(r0+2); A_cols.push_back(fc+1); A_vals.push_back( 1.0);
            A_rows.push_back(r0+2); A_cols.push_back(fc+2); A_vals.push_back(-mu_);
            A_rows.push_back(r0+3); A_cols.push_back(fc+1); A_vals.push_back(-1.0);
            A_rows.push_back(r0+3); A_cols.push_back(fc+2); A_vals.push_back(-mu_);
        }
    casadi::DM A_dm = casadi::DM::triplet(A_rows, A_cols, casadi::DM(A_vals), NCON, NV);

    // ── 3. Build solver once (constant sparsity for H and A) ───────────
    if (!casadi_solver_built_)
    {
        // libcasadi_conic_osqp.so was built from the same CasADi source tree as
        // /usr/local/lib/libcasadi.so.3.7, so there is no ABI mismatch.
        // It lives in the build dir until someone copies it to /usr/local/lib/.
        // CASADIPATH lets the plugin loader find it; LD_LIBRARY_PATH lets dlopen
        // resolve its runtime dependency on /opt/ros/jazzy/lib/libosqp.so.
        const std::string casadi_build = "/home/rooteq/dev/libs/casadi/build/lib";
        const std::string osqp_ros     = "/opt/ros/jazzy/lib";
        {
            const char* cp = getenv("CASADIPATH");
            const std::string np = cp && *cp ? casadi_build + ":" + cp : casadi_build;
            setenv("CASADIPATH", np.c_str(), 1);
        }
        {
            const char* lp = getenv("LD_LIBRARY_PATH");
            const std::string np = lp && *lp ? osqp_ros + ":" + lp : osqp_ros;
            setenv("LD_LIBRARY_PATH", np.c_str(), 1);
        }

        std::vector<casadi_int> H_rows, H_cols;
        std::vector<double>     H_vals;
        H_rows.reserve(NV); H_cols.reserve(NV); H_vals.reserve(NV);
        for (int k = 0; k < N; ++k)
            for (int i = 0; i < NX; ++i) {
                H_rows.push_back(k*NX + i); H_cols.push_back(k*NX + i);
                H_vals.push_back(2.0 * Q_WEIGHTS[i]);
            }
        for (int k = 0; k < N; ++k)
            for (int i = 0; i < NU; ++i) {
                H_rows.push_back(NDYN + k*NU + i); H_cols.push_back(NDYN + k*NU + i);
                H_vals.push_back(2.0 * alpha_);
            }
        casadi_H_dm_ = casadi::DM::triplet(H_rows, H_cols, casadi::DM(H_vals), NV, NV);

        casadi::Dict osqp_opts;
        osqp_opts["eps_abs"]               = 1e-4;
        osqp_opts["eps_rel"]               = 1e-4;
        osqp_opts["max_iter"]              = casadi_int(1000);
        osqp_opts["polish"]                = false;
        osqp_opts["verbose"]               = false;
        osqp_opts["adaptive_rho"]          = true;
        osqp_opts["check_termination"]     = casadi_int(10);
        osqp_opts["adaptive_rho_interval"] = casadi_int(25);
        osqp_opts["scaling"]               = casadi_int(5);
        osqp_opts["scaled_termination"]    = true;

        casadi::Dict solver_opts;
        solver_opts["osqp"]              = osqp_opts;
        solver_opts["warm_start_primal"] = true;
        solver_opts["warm_start_dual"]   = true;
        solver_opts["error_on_fail"]     = false;

        std::map<std::string, casadi::Sparsity> qp;
        qp["h"] = casadi_H_dm_.sparsity();
        qp["a"] = A_dm.sparsity();
        casadi_solver_ = casadi::conic("conic_mpc", "osqp", qp, solver_opts);
        casadi_solver_built_ = true;
    }

    // ── 4. lba / uba ────────────────────────────────────────────────────
    // Dynamics: lb = ub = [Ad*x0; 0; ...; 0] (gravity in Ad, no gd needed)
    // Friction:  lb = -inf; ub = 0 (stance) or +inf (swing, constraint inactive)
    const double INF = casadi::inf;
    std::vector<double> lba_vec(NCON, 0.0), uba_vec(NCON, 0.0);
    const Eigen::Matrix<double, N_STATE, 1> beq0 = Ad_ * x0_;
    for (int i = 0; i < NX; ++i) { lba_vec[i] = uba_vec[i] = beq0[i]; }
    for (int k = 0; k < N; ++k)
        for (int leg = 0; leg < 4; ++leg) {
            const int r0 = NDYN + (k*4 + leg)*4;
            const double ub = contact_schedule_[k][leg] ? 0.0 : INF;
            lba_vec[r0] = lba_vec[r0+1] = lba_vec[r0+2] = lba_vec[r0+3] = -INF;
            uba_vec[r0] = uba_vec[r0+1] = uba_vec[r0+2] = uba_vec[r0+3] = ub;
        }

    // ── 5. lbx / ubx ────────────────────────────────────────────────────
    // State vars: unconstrained. Force vars: swing→0, stance fz≥fz_min.
    std::vector<double> lbx_vec(NV, -INF), ubx_vec(NV, INF);
    for (int k = 0; k < N; ++k)
        for (int leg = 0; leg < 4; ++leg) {
            const int base = NDYN + k*NU + 3*leg;
            if (!contact_schedule_[k][leg]) {
                lbx_vec[base] = ubx_vec[base] = 0.0;
                lbx_vec[base+1] = ubx_vec[base+1] = 0.0;
                lbx_vec[base+2] = ubx_vec[base+2] = 0.0;
            } else {
                lbx_vec[base+2] = fz_min_;
            }
        }

    // ── 6. Solve ─────────────────────────────────────────────────────────
    casadi::DMDict args;
    args["h"]   = casadi_H_dm_;
    args["g"]   = casadi::DM(g_vec);
    args["a"]   = A_dm;
    args["lba"] = casadi::DM(lba_vec);
    args["uba"] = casadi::DM(uba_vec);
    args["lbx"] = casadi::DM(lbx_vec);
    args["ubx"] = casadi::DM(ubx_vec);
    if (has_casadi_warm_) {
        args["x0"]     = casadi_z_warm_;
        args["lam_x0"] = casadi_lam_x_warm_;
        args["lam_a0"] = casadi_lam_a_warm_;
    }

    casadi::DMDict sol;
    try {
        sol = casadi_solver_(args);
    } catch (const std::exception& e) {
        std::printf("[MPC run_casadi()] THREW: %s\n", e.what());
        has_casadi_warm_ = false;
        for (auto& f : grfs_) f.setZero();
        return;
    }

    // ── 7. Save warm start and extract GRFs ─────────────────────────────
    const casadi::DM& z_dm = sol.at("x");

    // Sanity check: NaN or wildly large forces → discard, clear warm start
    const double fz0 = double(z_dm(NDYN + 2));  // FL fz
    if (!std::isfinite(fz0) || std::abs(fz0) > 2000.0) {
        std::printf("[MPC run_casadi()] BAD SOLUTION (fz0=%.1f), skipping\n", fz0);
        has_casadi_warm_ = false;
        for (auto& f : grfs_) f.setZero();
        return;
    }

    casadi_z_warm_     = z_dm;
    casadi_lam_x_warm_ = sol.at("lam_x");
    casadi_lam_a_warm_ = sol.at("lam_a");
    has_casadi_warm_   = true;

    // ── 8. Extract GRFs (U_0 = z[NDYN .. NDYN+NU-1]) ───────────────────
    for (int leg = 0; leg < 4; ++leg)
        grfs_[leg] = Eigen::Vector3d(
            double(z_dm(NDYN + 3*leg)),
            double(z_dm(NDYN + 3*leg + 1)),
            double(z_dm(NDYN + 3*leg + 2)));

    if (++print_counter_ % 2 == 0)
        std::printf("[MPC GRF fz]  FL=%5.1f  FR=%5.1f  BL=%5.1f  BR=%5.1f  N  |"
                    "  pz=%.3f  vz=%.3f  mass=%.2f\n",
                    grfs_[0].z(), grfs_[1].z(), grfs_[2].z(), grfs_[3].z(),
                    x0_[5], x0_[11], mass_);
}

} // namespace quadro
