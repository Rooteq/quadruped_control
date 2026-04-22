#pragma once
#include <array>
#include <cmath>
#include <eigen3/Eigen/Dense>

#include "qpOASES.hpp"

#include "model.hpp"
#include "gait_scheduler.hpp"

namespace quadro
{

// ── MPC horizon ───────────────────────────────────────────────────────────────
// Defined here so both MPC and Controller use the same constants.
// Total prediction window: HORIZON_STEPS × MPC_DT = 0.33 s
static constexpr int    HORIZON_STEPS = 10;
static constexpr double MPC_DT        = 0.033;

class MPC
{
public:

    /// Snapshot all data needed for this solve from the model and controller.
    /// Called once per mpcCallback before any matrix computation.
    void update(const QuadroModel& model,
                const Eigen::Vector3d& angular_vel_cmd,
                const Eigen::Vector3d& linear_vel_cmd,
                const std::array<Eigen::Matrix<double, 13, 1>, HORIZON_STEPS>& x_ref,
                const GaitScheduler& gait_scheduler,
                const std::array<Eigen::Vector3d, NUM_LEGS>& foot_positions);

    /// Build Ac/Ad (single, from average yaw) and Bc[n]/Bd[n] (per step, from
    /// per-step yaw and contact schedule). Must be called after update().
    void calculateDynamicsMatrices();

    /// Solve QP and populate grfs_
    void run();

    // ── Accessors ─────────────────────────────────────────────────
    const Eigen::Matrix<double, 13, 13>& continuousA() const { return Ac_; }
    const Eigen::Matrix<double, 13, 13>& discreteA()   const { return Ad_; }

    // Per-step B matrices — index by horizon step [0, HORIZON_STEPS)
    const std::array<Eigen::Matrix<double, 13, 12>, HORIZON_STEPS>& continuousB() const { return Bc_; }
    const std::array<Eigen::Matrix<double, 13, 12>, HORIZON_STEPS>& discreteB()   const { return Bd_; }

    // Reference trajectory set by last update() — read by QP builder
    const std::array<Eigen::Matrix<double, 13, 1>, HORIZON_STEPS>& referenceTrajectory() const { return x_ref_; }

    // Contact schedule: contact_schedule_[k][leg] = true if leg is stance at step k
    const std::array<std::array<bool, NUM_LEGS>, HORIZON_STEPS>& contactSchedule() const { return contact_schedule_; }

    // Current state snapshot
    const Eigen::Matrix<double, 13, 1>& currentState() const { return x0_; }

    // GRFs produced by run() — one 3D force per leg (world frame), zero for swing
    const std::array<Eigen::Vector3d, NUM_LEGS>& groundReactionForces() const { return grfs_; }

private:

    // ── Snapshot (populated by update()) ──────────────────────────
    Eigen::Matrix<double, 13, 1>                             x0_           = Eigen::Matrix<double, 13, 1>::Zero();
    double                                                   mass_         = 0.0;
    Eigen::Matrix3d                                          body_inertia_ = Eigen::Matrix3d::Zero();
    std::array<Eigen::Vector3d, NUM_LEGS>                    foot_positions_{};
    Eigen::Vector3d                                          angular_vel_cmd_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d                                          linear_vel_cmd_  = Eigen::Vector3d::Zero();
    std::array<Eigen::Matrix<double, 13, 1>, HORIZON_STEPS>  x_ref_{};
    std::array<std::array<bool, NUM_LEGS>, HORIZON_STEPS>    contact_schedule_{};

    // ── Computed matrices ──────────────────────────────────────────
    Eigen::Matrix<double, 13, 13> Ac_ = Eigen::Matrix<double, 13, 13>::Zero();
    Eigen::Matrix<double, 13, 13> Ad_ = Eigen::Matrix<double, 13, 13>::Zero();

    // Per-step: Bc_[n] and Bd_[n] are 13×12 (4 legs × 3 forces)
    std::array<Eigen::Matrix<double, 13, 12>, HORIZON_STEPS> Bc_{};
    std::array<Eigen::Matrix<double, 13, 12>, HORIZON_STEPS> Bd_{};

    // ── Output (populated by run()) ───────────────────────────────
    // grfs_[i] = 3D GRF for leg i (world frame). Zero for swing legs.
    // Layout matches LegIdx: FL=0, FR=1, BL=2, BR=3
    std::array<Eigen::Vector3d, NUM_LEGS> grfs_{
        Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
        Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()
    };

    // ── QP dimensions ─────────────────────────────────────────────
    static constexpr int N_STATE = 13;
    static constexpr int N_FORCE = static_cast<int>(NUM_LEGS) * 3;          // 12
    static constexpr int N_VAR   = N_FORCE * HORIZON_STEPS;                 // 120
    static constexpr int N_PRED  = N_STATE * HORIZON_STEPS;                 // 130
    static constexpr int N_CON   = 4 * static_cast<int>(NUM_LEGS) * HORIZON_STEPS; // 160

    // ── Tuning parameters ─────────────────────────────────────────
    static constexpr double mu_     = 0.6;    // friction coefficient
    static constexpr double fz_min_ = 0.2;   // min normal GRF [N]
    static constexpr double fz_max_ = 70.0;  // max normal GRF [N]
    static constexpr double alpha_  = 1e-6;   // regularisation (force magnitude)

    // State cost weights: [roll, pitch, yaw, px, py, pz, wx, wy, wz, vx, vy, vz, -g]
    static constexpr double Q_WEIGHTS[N_STATE] = {
        10.0, 20.0,  1.0,  // roll(φ), pitch(θ), yaw(ψ)  — high: attitude stability
         1.0,  1.0, 50.0,  // px, py, pz                  — high pz: height tracking
         1.0,  1.0,  1.0,  // ωx, ωy, ωz
         2.0,  2.0,  1.0,  // vx, vy, vz
         0.0                // const, don't penalise
    };

    // ── Pre-allocated QP matrices ──────────────────────────────────
    // Condensed system: X = Aqp*x0 + Bqp*U
    Eigen::Matrix<double, N_PRED, N_STATE>  Aqp_     = Eigen::Matrix<double, N_PRED, N_STATE>::Zero();
    Eigen::Matrix<double, N_PRED, N_VAR>    Bqp_     = Eigen::Matrix<double, N_PRED, N_VAR>::Zero();

    // Diagonal of the block-diagonal state cost matrix L (N_PRED entries)
    Eigen::Matrix<double, N_PRED, 1>        L_diag_  = Eigen::Matrix<double, N_PRED, 1>::Zero();

    // Stacked reference: X_ref = [x_ref[0]; ...; x_ref[k-1]]
    Eigen::Matrix<double, N_PRED, 1>        X_ref_qp_ = Eigen::Matrix<double, N_PRED, 1>::Zero();

    // QP cost: ½UᵀHU + gᵀU
    Eigen::Matrix<double, N_VAR, N_VAR>     H_       = Eigen::Matrix<double, N_VAR, N_VAR>::Zero();
    Eigen::Matrix<double, N_VAR, 1>         g_qp_    = Eigen::Matrix<double, N_VAR, 1>::Zero();

    // Friction-pyramid constraint matrix and bounds: lbC <= C*U <= ubC
    // N_CON×N_VAR = 160×120 = 153 KB — exceeds Eigen's fixed-size stack limit,
    // so stored as dynamic matrices (heap allocated at construction).
    Eigen::MatrixXd C_   = Eigen::MatrixXd::Zero(N_CON, N_VAR);
    Eigen::Matrix<double, N_CON, 1>         lbC_     = Eigen::Matrix<double, N_CON, 1>::Zero();
    Eigen::Matrix<double, N_CON, 1>         ubC_     = Eigen::Matrix<double, N_CON, 1>::Zero();

    // Per-variable bounds (stance: fz in [fz_min, fz_max]; swing: lb=ub=0)
    Eigen::Matrix<double, N_VAR, 1>         lb_      = Eigen::Matrix<double, N_VAR, 1>::Zero();
    Eigen::Matrix<double, N_VAR, 1>         ub_      = Eigen::Matrix<double, N_VAR, 1>::Zero();

    // Row-major copies required by qpOASES (which expects C-order arrays).
    // H_qp_ is fine fixed-size (115 KB); C_qp_ also dynamic for the same reason as C_.
    Eigen::Matrix<double, N_VAR, N_VAR, Eigen::RowMajor> H_qp_;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        C_qp_ = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Zero(N_CON, N_VAR);

    // ── qpOASES solver ────────────────────────────────────────────
    // H and A change every MPC step (Bqp / contact schedule), so init() is
    // called each cycle. qpOASES resets internally on each init() call.
    qpOASES::QProblem qp_{N_VAR, N_CON, qpOASES::HST_POSDEF};

    static Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d& v);

    static constexpr double g_ = 9.81;
};

} // namespace quadro
