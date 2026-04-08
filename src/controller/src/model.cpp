
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

        x.setZero();
        x[12] = -g;

        // ── Cache mass and composite body inertia ─────────────────
        total_mass_ = pinocchio::computeTotalMass(model_);

        // Compute composite inertia about the robot's COM at neutral config.
        // For each body: rotate its inertia (stored at body COM, in joint frame)
        // to world frame, then apply the parallel axis theorem to move it to
        // the robot's total COM.
        auto q_neutral = pinocchio::neutral(model_);
        pinocchio::forwardKinematics(model_, data_, q_neutral);
        pinocchio::centerOfMass(model_, data_, q_neutral);   // fills data_.com[0]

        Eigen::Vector3d com = data_.com[0];
        body_inertia_.setZero();

        // Then Rz * I Rz^T - rotate it to the world frame
        for (pinocchio::JointIndex i = 1; i < (pinocchio::JointIndex)model_.njoints; ++i)
        {
            const pinocchio::Inertia& Y  = model_.inertias[i];
            const pinocchio::SE3&     oMi = data_.oMi[i];

            double mi = Y.mass();
            if (mi <= 0.0) continue;

            // Body COM in world frame
            Eigen::Vector3d ci = oMi.act(Y.lever());

            // Vector from robot COM to this body's COM
            Eigen::Vector3d r = ci - com;

            // Rotate body inertia (about its own COM) into world frame
            Eigen::Matrix3d Ri = oMi.rotation();
            Eigen::Matrix3d Ii = Ri * Y.inertia().matrix() * Ri.transpose();

            // Parallel axis theorem: shift to robot COM
            body_inertia_ += Ii + mi * (r.squaredNorm() * Eigen::Matrix3d::Identity() - r * r.transpose());
        }

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
    pinocchio::crba(model_,data_,q_pin_);
    pinocchio::computeGeneralizedGravity(model_, data_, q_pin_);
    pinocchio::centerOfMass(model_, data_, q_pin_);

    // Remap gravity torques from Pinocchio order → canonical order
    gravity_canonical_.resize(NUM_JOINTS);
    for (size_t i = 0; i < NUM_JOINTS; ++i)
    {
        gravity_canonical_[i] = data_.g[canonical_to_pin_[i]];
    }
    // pinocchio::centerOfMass(model_, data_, q_pin_, dq_pin_);

}

void QuadroModel::updateBaseState(const Eigen::Vector3d& position,
                                   const Eigen::Quaterniond& orientation,
                                   const Eigen::Vector3d& linear_velocity,
                                   const Eigen::Vector3d& angular_velocity)
{
    // Z-Y-X Euler angles: eulerAngles(2,1,0) returns [yaw, pitch, roll]
    Eigen::Vector3d euler = orientation.toRotationMatrix().eulerAngles(2, 1, 0);
    x[0]  = euler[2];              // roll  (φ)
    x[1]  = euler[1];              // pitch (θ)
    x[2]  = euler[0];              // yaw   (ψ)
    x[3]  = position[0];
    x[4]  = position[1];
    x[5]  = position[2];
    x[6]  = angular_velocity[0];
    x[7]  = angular_velocity[1];
    x[8]  = angular_velocity[2];
    x[9]  = linear_velocity[0];
    x[10] = linear_velocity[1];
    x[11] = linear_velocity[2];
    x[12] = -g;
}

Eigen::Vector3d QuadroModel::footPosition(int leg_idx) const
{
    return data_.oMf[foot_frame_ids_[leg_idx]].translation();
}

Eigen::Matrix<double, 3, 12> QuadroModel::footJacobianLinear(int leg_idx) const
{
    // Full 6×nv frame Jacobian (linear rows on top) in LOCAL_WORLD_ALIGNED mode,
    // meaning foot velocities are expressed in the base/world frame of the fixed-base model.
    Eigen::MatrixXd J_full = Eigen::MatrixXd::Zero(6, model_.nv);
    pinocchio::getFrameJacobian(model_, const_cast<pinocchio::Data&>(data_),
                                foot_frame_ids_[leg_idx],
                                pinocchio::LOCAL_WORLD_ALIGNED, J_full);

    // Extract linear part (rows 0-2) and remap Pinocchio column order → canonical order
    Eigen::Matrix<double, 3, 12> J;
    for (size_t i = 0; i < NUM_JOINTS; ++i)
        J.col(i) = J_full.block<3, 1>(0, canonical_to_pin_[i]);

    return J;
}

// Robot's model should include frames in the hip position, now manually calculated hip location is used
Eigen::Vector3d QuadroModel::hipPosition(int leg_idx) const 
{
    return data_.oMf[hip_frame_ids_[leg_idx]].translation();
}

} // namespace quadro
