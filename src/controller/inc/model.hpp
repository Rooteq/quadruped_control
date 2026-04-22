#pragma once

#include <array>
#include <string>
#include <eigen3/Eigen/Dense>

#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/rnea.hpp>

namespace quadro
{

        
static constexpr size_t NUM_JOINTS = 12;
static constexpr size_t NUM_LEGS = 4;
static constexpr size_t JOINTS_PER_LEG = 3;

// ── Canonical joint order ────────────────────────────────────────
// Grouped by leg: FL, FR, BL, BR × (hip, knee, ankle)
// Use JointIdx to index into q, dq, effort, Kp, Kd arrays.

enum LegIdx  { FL = 0, FR = 1, BL = 2, BR = 3 };
enum JointIdx {
    FL_HIP = 0, FL_KNEE, FL_ANKLE,
    FR_HIP,     FR_KNEE, FR_ANKLE,
    BL_HIP,     BL_KNEE, BL_ANKLE,
    BR_HIP,     BR_KNEE, BR_ANKLE
};

// EXPECTED_JOINT_NAMES[i] is the URDF joint name for JointIdx i
static const std::array<std::string, NUM_JOINTS> EXPECTED_JOINT_NAMES = {
    "fl_m1_s1",   "fl_m2_s2",   "fl_l4_l3",   // FL: hip, knee, ankle
    "fr_m1_s1",   "fr_m2_s2",   "fr_l4_l3",   // FR
    "bl_m1_s1",   "bl_m2_s2",   "bl_l4_l3",   // BL
    "br_m1_s1",   "br_m2_s2",   "br_l4_l3",   // BR
};

class QuadroModel
{
public:
    QuadroModel() = default;

    /// Build model from URDF. joint_names defines the canonical ordering —
    /// the mapping from your JointIdx to Pinocchio's internal order is
    /// computed once here.
    explicit QuadroModel(const std::string& urdf_path);

    /// Update joint state and run Pinocchio forward algorithms.
    /// q, dq, effort are in canonical (JointIdx) order.
    void updateState(const Eigen::VectorXd& q, const Eigen::VectorXd& dq,
                     const Eigen::VectorXd& effort);

    // State in canonical order — index with JointIdx: q_[FL_HIP], dq_[BR_ANKLE], etc.
    const Eigen::VectorXd& jointPositions() const { return q_; }
    const Eigen::VectorXd& jointVelocities() const { return dq_; }
    const Eigen::VectorXd& jointEfforts() const { return effort_; }
    const Eigen::VectorXd& gravityCompensation() const { return gravity_canonical_; }

    /// Foot position in world frame (from Pinocchio FK, updated by updateState)
    Eigen::Vector3d footPosition(int leg_idx) const;

    /// Foot linear velocity in world frame (from Pinocchio FK, updated by updateState)
    Eigen::Vector3d footVelocity(int leg_idx) const;

    /// 3×3 foot Jacobian: linear velocity in world frame, columns = leg's 3 joints only
    Eigen::Matrix3d footJacobian(int leg_idx) const;

    /// Hip joint position in world frame
    Eigen::Vector3d hipPosition(int leg_idx) const;

    /// Update base state from odometry. Populates the 13-state vector:
    /// x = [roll, pitch, yaw, px, py, pz, wx, wy, wz, vx, vy, vz, -g]
    void updateBaseState(const Eigen::Vector3d& position,
                         const Eigen::Quaterniond& orientation,
                         const Eigen::Vector3d& linear_velocity,
                         const Eigen::Vector3d& angular_velocity);

    const Eigen::Matrix<double, 13, 1>& stateVector() const { return x; }

    /// Total robot mass (from data.Ig)
    double mass() const { return data_.Ig.mass(); }

    /// Composite inertia tensor of the whole robot about its COM,
    /// expressed in the world frame at neutral config (cached at construction).
    /// Used as _BI in the MPC rigid-body model (eq. 14-15 in the paper).
    Eigen::Matrix3d bodyInertia() const { return data_.Ig.inertia().matrix(); }

    /// Yaw-only rotation matrix R_z(ψ): rotates body-frame vectors to world frame.
    /// Equivalent to go2.R_z in the Python reference. Updated by updateBaseState().
    const Eigen::Matrix3d& bodyYawRotation() const { return R_z_; }
    const Eigen::Matrix3d& bodyToWorldRotation() const { return R_b_w_; }

    const pinocchio::Model& pinocchioModel() const { return model_; }
    pinocchio::Data& pinocchioData() { return data_; }
    const pinocchio::Data& pinocchioData() const { return data_; }

private:
    pinocchio::Model model_;
    mutable pinocchio::Data data_;

    // Joint state in canonical (JointIdx) order
    Eigen::VectorXd q_;
    Eigen::VectorXd dq_;
    Eigen::VectorXd effort_;

    // Temporary vectors in Pinocchio order (avoid reallocation)
    Eigen::VectorXd q_pin_;
    Eigen::VectorXd dq_pin_;

    Eigen::VectorXd gravity_canonical_;

    // canonical_to_pin_[i] = Pinocchio's q-index for canonical joint i
    std::array<int, NUM_JOINTS> canonical_to_pin_{};
    // canonical_to_pin_v_[i] = Pinocchio's v-index for canonical joint i
    std::array<int, NUM_JOINTS> canonical_to_pin_v_{};
    // leg_pin_v_cols_[leg][j] = Pinocchio's v-index for the j-th joint of leg
    std::array<std::array<int, JOINTS_PER_LEG>, NUM_LEGS> leg_pin_v_cols_{};

    // Cached frame IDs (set once in constructor)
    std::array<pinocchio::FrameIndex, NUM_LEGS> foot_frame_ids_;
    std::array<pinocchio::FrameIndex, NUM_LEGS> hip_frame_ids_;

    // Cached physical parameters (computed once in constructor)
    Eigen::Matrix3d R_z_          = Eigen::Matrix3d::Identity(); // yaw-only body→world rotation
    Eigen::Matrix3d R_b_w_        = Eigen::Matrix3d::Identity(); // full body→world rotation


    Eigen::Matrix<double, 13, 13> Ac;
    Eigen::Matrix<double, 13, 12> Bc;
    Eigen::Matrix<double, 13, 1> x;

    double g = 9.81;

};
    
} // namespace quadro

