#include <array>
#include <string>
#include <vector>

#include "quadro_controller.hpp"

#include <eigen3/Eigen/Dense>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "ament_index_cpp/get_package_share_directory.hpp"

#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "geometry_msgs/msg/transform_stamped.hpp"

using namespace std::chrono_literals;

// TF link name → Pinocchio frame name mapping (FL, FR, BL, BR)
static const std::array<std::string, 4> TF_FOOT_FRAMES   = {"feet_4", "feet_3", "feet_2", "feet"};
static const std::array<std::string, 4> PIN_FOOT_FRAMES   = {"fl_feet", "fr_feet", "bl_feet", "br_feet"};
static const std::array<std::string, 4> LEG_NAMES         = {"FL", "FR", "BL", "BR"};
static const std::string                BASE_FRAME        = "back_plate";

class TestNode : public rclcpp::Node
{
public:
    TestNode() : Node("test_node")
    {
        std::string package_path = ament_index_cpp::get_package_share_directory("quadruped_control");
        std::string urdf_file = package_path + "/description/isaac_sim/quadro.urdf";

        pinocchio::urdf::buildModel(urdf_file, model_);
        data_ = pinocchio::Data(model_);

        q_pin_.resize(model_.nq);
        dq_pin_.resize(model_.nv);
        q_pin_.setZero();
        dq_pin_.setZero();

        // Build joint name → pinocchio q-index map
        for (size_t i = 0; i < quadro::NUM_JOINTS; ++i)
        {
            const auto& name = quadro::EXPECTED_JOINT_NAMES[i];
            if (!model_.existJointName(name))
            {
                RCLCPP_ERROR(get_logger(), "Joint '%s' not found in URDF!", name.c_str());
                throw std::runtime_error("Joint not found: " + name);
            }
            pinocchio::JointIndex jid = model_.getJointId(name);
            canonical_to_pin_[i] = model_.idx_qs[jid];
        }

        // Cache pinocchio foot frame IDs
        for (size_t i = 0; i < 4; ++i)
        {
            if (!model_.existFrame(PIN_FOOT_FRAMES[i]))
            {
                RCLCPP_ERROR(get_logger(), "Pinocchio frame '%s' not found!", PIN_FOOT_FRAMES[i].c_str());
                throw std::runtime_error("Frame not found: " + PIN_FOOT_FRAMES[i]);
            }
            foot_frame_ids_[i] = model_.getFrameId(PIN_FOOT_FRAMES[i]);
        }

        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        sub_joint_states_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/joint_states", 10,
            [this](const sensor_msgs::msg::JointState::SharedPtr msg) { jointStateCallback(msg); });

        RCLCPP_INFO(get_logger(), "TestNode ready — waiting for /joint_states...");
    }

private:
    void jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg)
    {
        // --- Build q_pin_ from joint_states ---
        q_pin_.setZero();
        for (size_t i = 0; i < quadro::NUM_JOINTS; ++i)
        {
            const auto& name = quadro::EXPECTED_JOINT_NAMES[i];
            auto it = std::find(msg->name.begin(), msg->name.end(), name);
            if (it == msg->name.end())
            {
                RCLCPP_WARN_ONCE(get_logger(), "Joint '%s' missing from /joint_states", name.c_str());
                return;
            }
            size_t idx = std::distance(msg->name.begin(), it);
            q_pin_[canonical_to_pin_[i]] = msg->position[idx];
        }

        // --- Pinocchio FK ---
        pinocchio::forwardKinematics(model_, data_, q_pin_);
        pinocchio::updateFramePlacements(model_, data_);

        // --- TF lookup + comparison (FL only) ---
        constexpr size_t FL = 0;

        Eigen::Vector3d pin_t = data_.oMf[foot_frame_ids_[FL]].translation();

        geometry_msgs::msg::TransformStamped tf_msg;
        try
        {
            tf_msg = tf_buffer_->lookupTransform(BASE_FRAME, TF_FOOT_FRAMES[FL], tf2::TimePointZero);
        }
        catch (const tf2::TransformException& ex)
        {
            RCLCPP_WARN(get_logger(), "[FL] TF lookup failed: %s", ex.what());
            return;
        }

        const auto& tf_t = tf_msg.transform.translation;

        if((tf_t.x - pin_t.x()) > 0.03 || (tf_t.y - pin_t.y()) > 0.05 || (tf_t.z - pin_t.z()) > 0.05)
        {
            RCLCPP_INFO(get_logger(),
                "\n[FL] Pinocchio: [%.4f, %.4f, %.4f]  |  \n TF: [%.4f, %.4f, %.4f]  | \n diff: [%.4f, %.4f, %.4f]",
                pin_t.x(), pin_t.y(), pin_t.z(),
                tf_t.x,    tf_t.y,    tf_t.z,
                tf_t.x - pin_t.x(), tf_t.y - pin_t.y(), tf_t.z - pin_t.z());
        }
    }

    pinocchio::Model model_;
    pinocchio::Data  data_;

    Eigen::VectorXd q_pin_;
    Eigen::VectorXd dq_pin_;

    std::array<int, quadro::NUM_JOINTS>              canonical_to_pin_;
    std::array<pinocchio::FrameIndex, 4>             foot_frame_ids_;

    std::shared_ptr<tf2_ros::Buffer>            tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr sub_joint_states_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<TestNode>());
    rclcpp::shutdown();
    return 0;
}
