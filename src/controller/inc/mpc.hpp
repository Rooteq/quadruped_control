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
                const GaitScheduler& gait_scheduler);

    /// Build Ac/Ad (single, from average yaw) and Bc[n]/Bd[n] (per step, from
    /// per-step yaw and contact schedule). Must be called after update().
    void calculateDynamicsMatrices();

    /// Solve QP and populate grfs_. QP logic to be added — currently zeroes.
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

    static Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d& v);

    static constexpr double g_ = 9.81;
};

} // namespace quadro
