
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
}

} // namespace quadro
