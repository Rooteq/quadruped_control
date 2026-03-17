#include <array>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "quadro_controller.hpp"

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "geometry_msgs/msg/twist.hpp"

using namespace std::chrono_literals;



// ── Snapshot structs ─────────────────────────────────────────────

struct JointSnapshot {
    std::array<double, quadro::NUM_JOINTS> position{};
    std::array<double, quadro::NUM_JOINTS> velocity{};
    bool valid = false;
};

struct CmdVelSnapshot {
    double vx = 0.0;
    double vy = 0.0;
    double vz = 0.0;
    double wx = 0.0;
    double wy = 0.0;
    double wz = 0.0;
};

// ── Node ─────────────────────────────────────────────────────────

class ConvexMpcController : public rclcpp::Node {
public:
    ConvexMpcController()
    : Node("convex_mpc_controller")
    {

        sub_joint_states_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/joint_states", 10,
            std::bind(&ConvexMpcController::jointStateCallback, this, std::placeholders::_1));

        sub_cmd_vel_ = this->create_subscription<geometry_msgs::msg::Twist>(
            "/cmd_vel", 10,
            std::bind(&ConvexMpcController::cmdVelCallback, this, std::placeholders::_1));


        pub_joint_cmd_ = this->create_publisher<sensor_msgs::msg::JointState>(
            "/joint_command", 10);


        control_timer_ = this->create_wall_timer(
            3333us,  // ~300 Hz
            std::bind(&ConvexMpcController::controlCallback, this));

        mpc_timer_ = this->create_wall_timer(
            33ms,    // ~30 Hz
            std::bind(&ConvexMpcController::mpcCallback, this));

        planning_timer_ = this->create_wall_timer(
            33ms,    // ~30 Hz
            std::bind(&ConvexMpcController::planningCallback, this));

        RCLCPP_INFO(this->get_logger(), "Controller started");
    }

private:

    void jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg)
    {
        // On first message, build sim↔internal index mapping
        if (!joint_map_ready_) {
            buildJointMap(msg->name);
        }

        std::lock_guard<std::mutex> lock(state_mutex_);
        for (size_t i = 0; i < quadro::NUM_JOINTS; ++i) {
            size_t sim_idx = internal_to_sim_[i];
            if (sim_idx < msg->position.size())
                joint_snap_.position[i] = msg->position[sim_idx];
            if (sim_idx < msg->velocity.size())
                joint_snap_.velocity[i] = msg->velocity[sim_idx];
        }
        joint_snap_.valid = true;
    }

    void cmdVelCallback(const geometry_msgs::msg::Twist::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        cmd_vel_snap_.vx = msg->linear.x;
        cmd_vel_snap_.vy = msg->linear.y;
        cmd_vel_snap_.vz = msg->linear.z;
        cmd_vel_snap_.wx = msg->angular.x;
        cmd_vel_snap_.wy = msg->angular.y;
        cmd_vel_snap_.wz = msg->angular.z;
    }


    void controlCallback()
    {
        JointSnapshot joints;
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            joints = joint_snap_;
        }

        if (!joints.valid) return;

        // For now: echo back current positions (hold in place)
        // Later: stance torques from MPC, swing PD tracking
        publishJointCommand(joints.position);
    }

    void mpcCallback()
    {
        // Will contain:
        //   1. Build reference trajectory from cmd_vel
        //   2. Build Ac, Bc dynamics matrices
        //   3. Discretize (ZOH)
        //   4. Form condensed QP (H, g, C)
        //   5. Solve QP
        //   6. Store GRFs for controlCallback
    }

    void planningCallback()
    {
        // Will contain:
        //   1. Advance gait phase
        //   2. Determine stance/swing per leg
        //   3. Raibert foot placement for swing legs
        //   4. Init/update swing trajectories
    }

    // ── Helpers ──────────────────────────────────────────────────

    void buildJointMap(const std::vector<std::string>& sim_names)
    {
        // Build name→sim_index lookup
        std::unordered_map<std::string, size_t> name_to_sim;
        for (size_t i = 0; i < sim_names.size(); ++i) {
            name_to_sim[sim_names[i]] = i;
        }

        // For each internal index, find the corresponding sim index
        bool all_found = true;
        for (size_t i = 0; i < quadro::NUM_JOINTS; ++i) {
            auto it = name_to_sim.find(quadro::EXPECTED_JOINT_NAMES[i]);
            if (it != name_to_sim.end()) {
                internal_to_sim_[i] = it->second;
                if (it->second != i) {
                    RCLCPP_WARN(this->get_logger(),
                        "Joint '%s': sim index %zu != internal index %zu (will remap)",
                        quadro::EXPECTED_JOINT_NAMES[i].c_str(), it->second, i);
                }
            } else {
                RCLCPP_ERROR(this->get_logger(),
                    "Joint '%s' not found in /joint_states!", quadro::EXPECTED_JOINT_NAMES[i].c_str());
                internal_to_sim_[i] = i;  // fallback: identity
                all_found = false;
            }
        }

        sim_joint_names_ = sim_names;
        joint_map_ready_ = true;

        if (all_found) {
            RCLCPP_INFO(this->get_logger(), "Joint map built: all %zu joints matched", quadro::NUM_JOINTS);
        }
    }

    void publishJointCommand(const std::array<double, quadro::NUM_JOINTS>& positions)
    {
        if (!joint_map_ready_) return;

        sensor_msgs::msg::JointState msg;
        msg.header.stamp = this->now();
        msg.name = sim_joint_names_;
        msg.position.resize(sim_joint_names_.size(), 0.0);
        msg.velocity.resize(sim_joint_names_.size(), 0.0);
        msg.effort.resize(sim_joint_names_.size(), 0.0);

        // Map internal order back to sim order
        for (size_t i = 0; i < quadro::NUM_JOINTS; ++i) {
            msg.position[internal_to_sim_[i]] = positions[i];
        }

        pub_joint_cmd_->publish(msg);
    }

    // ── State (mutex-protected, written by subscribers) ──────────
    std::mutex state_mutex_;
    JointSnapshot joint_snap_;
    CmdVelSnapshot cmd_vel_snap_;

    // ── Joint mapping ────────────────────────────────────────────
    bool joint_map_ready_ = false;
    std::array<size_t, quadro::NUM_JOINTS> internal_to_sim_{};  // our index → sim index
    std::vector<std::string> sim_joint_names_;           // original names from sim

    // ── ROS interfaces ───────────────────────────────────────────
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr sub_joint_states_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr sub_cmd_vel_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr pub_joint_cmd_;
    rclcpp::TimerBase::SharedPtr control_timer_;   // 300 Hz
    rclcpp::TimerBase::SharedPtr mpc_timer_;        // 30 Hz
    rclcpp::TimerBase::SharedPtr planning_timer_;   // 30 Hz
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ConvexMpcController>());
    rclcpp::shutdown();
    return 0;
}
