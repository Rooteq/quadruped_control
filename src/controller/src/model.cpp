
#include "model.hpp"
#include "quadro_controller.hpp"   // for EXPECTED_JOINT_NAMES, NUM_JOINTS

namespace quadro
{

QuadroModel::QuadroModel(const std::string& urdf_path)
{
    try
    {
        pinocchio::urdf::buildModel(urdf_path, model_);
        data_ = pinocchio::Data(model_);

        q_.resize(model_.nq);
        dq_.resize(model_.nv);
        q_.setZero();
        dq_.setZero();

        q_pin_.resize(model_.nq);
        dq_pin_.resize(model_.nv);
        q_pin_.setZero();
        dq_pin_.setZero();

        // Build mapping: canonical JointIdx order → Pinocchio's internal q-index.
        // Pinocchio joint IDs start at 1 (0 is the "universe" joint).
        for (size_t i = 0; i < NUM_JOINTS; ++i)
        {
            const auto& name = EXPECTED_JOINT_NAMES[i];
            if (!model_.existJointName(name))
            {
                std::cerr << "Joint '" << name << "' not found in URDF model!" << std::endl;
                throw std::runtime_error("Joint '" + name + "' not found in URDF");
            }
            pinocchio::JointIndex jid = model_.getJointId(name);
            canonical_to_pin_[i] = model_.idx_qs[jid];
        }

        // Cache foot and hip frame IDs (order: FL, FR, BL, BR)
        // const std::array<std::string, NUM_LEGS> foot_frame_names = {
        //     "fl_feet", "fr_feet", "bl_feet", "br_feet"
        // };
        const std::array<std::string, NUM_LEGS> foot_frame_names = {
            "feet_4", "feet_3", "feet_2", "feet"
        };

        const std::array<std::string, NUM_LEGS> hip_frame_names = {
            "fl_m1_s1", "fr_m1_s1", "bl_m1_s1", "br_m1_s1"
        };

        for (size_t i = 0; i < NUM_LEGS; ++i)
        {
            if (!model_.existFrame(foot_frame_names[i]))
                throw std::runtime_error("Foot frame '" + foot_frame_names[i] + "' not found in URDF");
            if (!model_.existFrame(hip_frame_names[i]))
                throw std::runtime_error("Hip frame '" + hip_frame_names[i] + "' not found in URDF");

            foot_frame_ids_[i] = model_.getFrameId(foot_frame_names[i]);
            hip_frame_ids_[i] = model_.getFrameId(hip_frame_names[i]);
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to load URDF: " << e.what() << std::endl;
        throw;
    }
}

void QuadroModel::updateState(const Eigen::VectorXd& q, const Eigen::VectorXd& dq,
                               const Eigen::VectorXd& effort)
{
    // Store in canonical order (for user access via JointIdx)
    q_ = q;
    dq_ = dq;
    effort_ = effort;

    // Remap to Pinocchio order for algorithms
    for (size_t i = 0; i < NUM_JOINTS; ++i)
    {
        q_pin_[canonical_to_pin_[i]] = q[i];
        dq_pin_[canonical_to_pin_[i]] = dq[i];
    }

    pinocchio::forwardKinematics(model_, data_, q_pin_, dq_pin_);
    pinocchio::updateFramePlacements(model_, data_);
    pinocchio::computeJointJacobians(model_, data_, q_pin_);
    pinocchio::centerOfMass(model_, data_, q_pin_, dq_pin_);
    pinocchio::computeGeneralizedGravity(model_, data_, q_pin_);
    pinocchio::centerOfMass(model_, data_, q_pin_);

    // Remap gravity torques from Pinocchio order → canonical order
    gravity_canonical_.resize(NUM_JOINTS);
    for (size_t i = 0; i < NUM_JOINTS; ++i)
    {
        gravity_canonical_[i] = data_.g[canonical_to_pin_[i]];
    }
}

Eigen::Vector3d QuadroModel::footPosition(int leg_idx) const
{
    return data_.oMf[foot_frame_ids_[leg_idx]].translation();
}

// Robot's model should include frames in the hip position, now manually calculated hip location is used
Eigen::Vector3d QuadroModel::hipPosition(int leg_idx) const 
{
    return data_.oMf[hip_frame_ids_[leg_idx]].translation();
}

} // namespace quadro
