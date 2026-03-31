#pragma once

#include <array>
#include <string>
#include <eigen3/Eigen/Dense>

#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/center-of-mass.hpp>

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

    /// Foot position in body frame (from Pinocchio FK, updated by updateState)
    Eigen::Vector3d footPosition(int leg_idx) const;

    /// Hip joint position in body frame
    Eigen::Vector3d hipPosition(int leg_idx) const;

    /// Update base state from odometry.
    void updateBaseState(const Eigen::Vector3d& position,
                         const Eigen::Quaterniond& orientation,
                         const Eigen::Vector3d& linear_velocity,
                         const Eigen::Vector3d& angular_velocity);

    const Eigen::Vector3d&    basePosition()        const { return base_position_; }
    const Eigen::Quaterniond& baseOrientation()     const { return base_orientation_; }
    const Eigen::Vector3d&    baseLinearVelocity()  const { return base_linear_vel_; }
    const Eigen::Vector3d&    baseAngularVelocity() const { return base_angular_vel_; }

    const pinocchio::Model& pinocchioModel() const { return model_; }
    pinocchio::Data& pinocchioData() { return data_; }
    const pinocchio::Data& pinocchioData() const { return data_; }

private:
    pinocchio::Model model_;
    pinocchio::Data data_;

    // Joint state in canonical (JointIdx) order
    Eigen::VectorXd q_;
    Eigen::VectorXd dq_;
    Eigen::VectorXd effort_;

    // Temporary vectors in Pinocchio order (avoid reallocation)
    Eigen::VectorXd q_pin_;
    Eigen::VectorXd dq_pin_;

    // Base state (from odometry)
    Eigen::Vector3d    base_position_    = Eigen::Vector3d::Zero();
    Eigen::Quaterniond base_orientation_ = Eigen::Quaterniond::Identity();
    Eigen::Vector3d    base_linear_vel_  = Eigen::Vector3d::Zero();
    Eigen::Vector3d    base_angular_vel_ = Eigen::Vector3d::Zero();

    // canonical_to_pin_[i] = Pinocchio's q-index for canonical joint i
    std::array<int, 12> canonical_to_pin_;

    // Cached frame IDs (set once in constructor)
    std::array<pinocchio::FrameIndex, NUM_LEGS> foot_frame_ids_;
    std::array<pinocchio::FrameIndex, NUM_LEGS> hip_frame_ids_;



};
    
} // namespace quadro

