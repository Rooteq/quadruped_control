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

    // ── Per-step Ac/Ad — driven by per-step reference yaw ─────────
    // In the condensed formulation a single Ad (built from avg ψ_ref) is
    // required so that Aqp = [Ad; Ad²; …] and Bqp's block-Toeplitz product
    // are well defined. The sparse formulation does NOT need that — each
    // dynamics row block can hold a different Ad[n] at no structural cost.
    // We exploit that here by building Ac[n] / Ad[n] from the per-step
    // reference yaw ψ_ref[n] directly, so the Θ̇ = Rzᵀ(ψ)·ω row block is
    // accurate at every horizon step (no avg-yaw approximation).
    const Eigen::Matrix3d  I_world_inv0 = body_inertia_.inverse();   // I_world⁻¹(ψ0)
    const Eigen::Matrix3d  I3_over_m    = Eigen::Matrix3d::Identity() / mass_;
    const double yaw0                   = x0_[2];

    for (int n = 0; n < HORIZON_STEPS; ++n)
    {
        const double psi_n = x_ref_[n][2];   // reference yaw at horizon step n

        // Rzᵀ(ψ_n) — Euler-rate kinematics map at step n.
        const double cy = std::cos(psi_n);
        const double sy = std::sin(psi_n);
        Eigen::Matrix3d Rz_T_n;
        Rz_T_n <<  cy, sy, 0.0,
                  -sy, cy, 0.0,
                   0.0, 0.0, 1.0;

        // ── Ac[n] ─────────────────────────────────────────────────
        Ac_[n].setZero();
        Ac_[n].block<3, 3>(0, 6) = Rz_T_n;                         // Θ̇ = Rzᵀ(ψ_n)·ω
        Ac_[n].block<3, 3>(3, 9) = Eigen::Matrix3d::Identity();    // ṗ = v
        Ac_[n](11, 12)           = 1.0;                             // v̇_z += -g via x[12]

        // ── Ad[n]: pure first-order Euler ─────────────────────────
        Ad_[n] = Eigen::Matrix<double, 13, 13>::Identity() + Ac_[n] * MPC_DT;

        Bc_[n].setZero();

        // Rotate the world-frame inverse inertia to the body's predicted
        // orientation at step n. ψn comes from the reference trajectory
        // (ψ0 + yaw_rate·tn), so this tracks the desired angular velocity.
        const double dyaw = psi_n - yaw0;
        const double cd = std::cos(dyaw), sd = std::sin(dyaw);
        Eigen::Matrix3d RzD;
        RzD << cd, -sd, 0.0,
               sd,  cd, 0.0,
               0.0, 0.0, 1.0;
        const Eigen::Matrix3d I_world_inv_n = RzD * I_world_inv0 * RzD.transpose();

        for (int i = 0; i < static_cast<int>(NUM_LEGS); ++i)
        {
            if (!contact_schedule_[n][i]) continue;

            // levers_[n][i] = foot_world − base_traj_world for this leg at step n
            // (already computed by TrajectoryGenerator::computeHorizonLevers).
            // Zero on swing steps; held constant per stance phase at the
            // value planned at takeoff.
            const Eigen::Vector3d& r        = levers_[n][i];
            const Eigen::Matrix3d  I_inv_sk = I_world_inv_n * skewSymmetric(r);

            Bc_[n].block<3, 3>(6, 3 * i) = I_inv_sk;   // ω̇ rows
            Bc_[n].block<3, 3>(9, 3 * i) = I3_over_m;  // v̇ rows
        }

        // ── Bd[n]: pure first-order Euler ─────────────────────────
        //     Bd[n] = T · Bc[n]
        // Force only enters ω/v rows at step k+1; orientation/position
        // pick up the effect at step k+2 via Ad. The ½dt² cross-coupling
        // blocks the previous version added by hand are dropped.
        Bd_[n].noalias() = MPC_DT * Bc_[n];
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
            for (int i = 0; i < NX; ++i)            // -Ad[k] on X_{k-1} sub-diagonal
                for (int j = 0; j < NX; ++j) {
                    A_rows.push_back(k*NX + i);
                    A_cols.push_back((k-1)*NX + j);
                    A_vals.push_back(-Ad_[k](i, j));
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
    // Dynamics: lb = ub = [Ad[0]*x0; 0; ...; 0] (gravity in Ad, no gd needed)
    // Friction:  lb = -inf; ub = 0 (stance) or +inf (swing, constraint inactive)
    const double INF = casadi::inf;
    std::vector<double> lba_vec(NCON, 0.0), uba_vec(NCON, 0.0);
    const Eigen::Matrix<double, N_STATE, 1> beq0 = Ad_[0] * x0_;
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

    // if (++print_counter_ % 2 == 0)
    //     std::printf("[MPC GRF fz]  FL=%5.1f  FR=%5.1f  BL=%5.1f  BR=%5.1f  N  |"
    //                 "  pz=%.3f  vz=%.3f  mass=%.2f\n",
    //                 grfs_[0].z(), grfs_[1].z(), grfs_[2].z(), grfs_[3].z(),
    //                 x0_[5], x0_[11], mass_);
}

} // namespace quadro
