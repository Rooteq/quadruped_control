#pragma once

#include <array>
#include <cmath>
#include <eigen3/Eigen/Dense>

#include "gait_scheduler.hpp"

namespace quadro
{

// Forward declaration — full definition in model.hpp, included by .cpp
class QuadroModel;

struct LegTarget {
    Eigen::Vector3d foot_pos = Eigen::Vector3d::Zero();  // world frame
    Eigen::Vector3d foot_vel = Eigen::Vector3d::Zero();  // world frame
    Eigen::Vector3d foot_acc = Eigen::Vector3d::Zero();  // world frame
};

struct SwingState {
    Eigen::Vector3d liftoff_pos = Eigen::Vector3d::Zero();  // body frame
    Eigen::Vector3d landing_pos = Eigen::Vector3d::Zero();  // body frame
    double apex_height = 0.10;  // meters above liftoff/landing z
    bool active = false;
    bool stance_initialized = false;
    Eigen::Vector3d stance_foot_pos = Eigen::Vector3d::Zero();  // foot target held during stance
};

class TrajectoryGenerator
{
public:
    TrajectoryGenerator() = default;
    explicit TrajectoryGenerator(double dt);

    /// Generate desired foot Cartesian targets for all legs.
    /// Swing legs track Bezier arc; stance legs hold last landing position.
    std::array<LegTarget, NUM_LEGS> generate(
        const QuadroModel& model,
        const GaitScheduler& gait,
        const Eigen::Vector3d& desired_linear_vel,
        const Eigen::Vector3d& desired_angular_vel,
        const Eigen::Vector3d& current_vel);

    static constexpr double NOMINAL_HEIGHT = -0.27;  // body-frame z of feet when standing

    Eigen::Vector3d nominalFootPosition(int leg) const
    {
        return {hipPos[leg].x(), hipPos[leg].y(), NOMINAL_HEIGHT};
    }

    Eigen::Vector3d getLandingPos(int leg) const
    {
        return swing_states_[leg].landing_pos;
    }

    /// Per-horizon-step foot lever arms r_i = foot_world − base_traj_world,
    /// matching Python's ComTraj.generate_traj loop. Lever for a swing step is
    /// zero (leg can't apply force); lever for a stance step is the value computed
    /// at the leg's last takeoff (or the current foot lever for legs already in
    /// stance at horizon start) and held constant across that stance phase.
    ///
    /// This is what feeds MPC's B[n] construction — replaces the old "single foot
    /// snapshot per leg" model so legs that lift off / touch down within the
    /// horizon get the right lever per step.
    ///
    /// Body trajectory is integrated as base_traj = current_CoM + vel_des_world·t,
    /// matching Python's pos_traj_world (no position anchor). The touchdown plan
    /// uses only nominal + drift + rotation correction (no k_v/k_p) — same as
    /// Python's compute_touchdown_world_for_traj_purpose_only.
    template<int N>
    void computeHorizonLevers(
        const QuadroModel& model,
        const GaitScheduler& gait,
        const Eigen::Vector3d& desired_linear_vel_body,
        const Eigen::Vector3d& desired_angular_vel_body,
        double mpc_dt,
        std::array<std::array<Eigen::Vector3d, NUM_LEGS>, N>& levers) const
    {
        const Eigen::VectorXd& state = model.stateVector();
        const Eigen::Vector3d com_world(state[3], state[4], state[5]);   // CoM
        const double yaw_initial = state[2];

        const Eigen::Matrix3d& R_z_init = model.bodyYawRotation();
        const Eigen::Vector3d vel_des_world = R_z_init * desired_linear_vel_body;
        const double yaw_rate = desired_angular_vel_body[2];

        // Initial r_next_td = current foot − current CoM (Python: get_foot_lever_world).
        // For legs already in stance at horizon start, this becomes the held lever.
        std::array<Eigen::Vector3d, NUM_LEGS> r_next_td;
        for (int leg = 0; leg < static_cast<int>(NUM_LEGS); ++leg)
            r_next_td[leg] = model.footPosition(leg) - com_world;

        // Sentinel != stance(1) and != swing(0) — guarantees a transition is
        // detected at i=0 so every leg's lever is initialised in the first step.
        std::array<int, NUM_LEGS> mask_previous = {2, 2, 2, 2};

        const double T_pred    = gait.swingTime() + 0.5 * gait.stanceTime();
        const double pred_time = T_pred * 0.5;

        for (int i = 0; i < N; ++i)
        {
            // Python: dummy.base_pos = pos_traj_world[:, i] uses (i+1)*dt offset,
            // while compute_current_mask is called at i*dt. Match that off-by-one.
            const double t_pos  = (i + 1) * mpc_dt;
            const double t_mask = i * mpc_dt;

            const Eigen::Vector3d base_pos_traj = com_world + vel_des_world * t_pos;
            const double yaw_traj = yaw_initial + yaw_rate * t_pos;
            const double cy = std::cos(yaw_traj), sy = std::sin(yaw_traj);
            Eigen::Matrix3d R_z_traj;
            R_z_traj << cy, -sy, 0.0,
                        sy,  cy, 0.0,
                        0.0, 0.0, 1.0;

            const auto current_mask = gait.contactMaskAt(t_mask);

            for (int leg = 0; leg < static_cast<int>(NUM_LEGS); ++leg)
            {
                const int curr = current_mask[leg] ? 1 : 0;
                const int prev = mask_previous[leg];

                if (curr != prev && curr == 0)
                {
                    // ── takeoff: plan next touchdown, lever = 0 this step ────
                    const Eigen::Vector3d hip_pos_world = base_pos_traj + R_z_traj * hipPos[leg];
                    const Eigen::Vector3d pos_nominal(hip_pos_world.x(), hip_pos_world.y(), 0.02);

                    // Drift: Python uses dummy.base_vel (BODY-frame velocity from dq[0:3])
                    // in a world-frame term. Match literally — for trot in place vel=0 anyway.
                    const Eigen::Vector3d drift(desired_linear_vel_body.x() * pred_time,
                                                desired_linear_vel_body.y() * pred_time, 0.0);

                    const double dtheta = yaw_rate * pred_time;
                    const double r_x = hip_pos_world.x() - base_pos_traj.x();
                    const double r_y = hip_pos_world.y() - base_pos_traj.y();
                    const Eigen::Vector3d rot_correction(-dtheta * r_y, dtheta * r_x, 0.0);

                    const Eigen::Vector3d touchdown_world = pos_nominal + drift + rot_correction;
                    r_next_td[leg] = touchdown_world - base_pos_traj;

                    levers[i][leg] = Eigen::Vector3d::Zero();
                }
                else if (curr != prev && curr == 1)
                {
                    // ── touchdown: stance starts, use planned lever ───────────
                    levers[i][leg] = r_next_td[leg];
                }
                else
                {
                    // ── no transition: hold last value ────────────────────────
                    // i==0 with no transition is unreachable (sentinel guarantees
                    // a "transition" on the first step), but keep a safe fallback.
                    levers[i][leg] = (i > 0) ? levers[i-1][leg] : r_next_td[leg];
                }

                mask_previous[leg] = curr;
            }
        }
    }

private:
    /// Raibert foot placement heuristic (body frame).
    /// `pos_des_world` is the slowly-tracked anchor used for the k_p position-
    /// correction term (matches Python's `go2.{x,y}_pos_des_world`).
    Eigen::Vector3d computeLandingPos(
        const QuadroModel& model,
        const GaitScheduler& gait,
        int leg_idx,
        const Eigen::Vector3d& current_vel,
        const Eigen::Vector3d& desired_vel,
        double yaw_rate_des,
        const Eigen::Vector3d& pos_des_world) const;

    /// Bezier swing arc: smooth-step XY, cubic Bezier Z
    Eigen::Vector3d evaluateSwing(const SwingState& state, double phase) const;

    /// Analytical time-derivative of evaluateSwing
    Eigen::Vector3d evaluateSwingVelocity(const SwingState& state, double phase,
                                          double phase_rate) const;

    /// Analytical second time-derivative of evaluateSwing (world frame)
    Eigen::Vector3d evaluateSwingAcceleration(const SwingState& state, double phase,
                                              double phase_rate) const;

    std::array<SwingState, NUM_LEGS> swing_states_;

    // Slowly-tracked anchor for the Raibert k_p position-correction term.
    // Initialised to the CoM on the first generate() call, then clamped each
    // tick to within ±MAX_POS_ERROR of the actual CoM. Matches Python's
    // ComTraj.pos_des_world (no integration with vel_des — it's a anchor, not
    // a setpoint trajectory).
    Eigen::Vector3d pos_des_world_      = Eigen::Vector3d::Zero();
    bool            pos_des_initialized_ = false;
    static constexpr double MAX_POS_ERROR = 0.1;  // metres

    double dt_ = 0.033;
    Eigen::Vector3d hipPos[NUM_LEGS] = {
                    {0.215, 0.13, 0.0},
                    {0.215, -0.13, 0.0},
                    {-0.180, 0.13, 0.0},
                    {-0.180, -0.13, 0.0}
                };

    // Eigen::Vector3d hipPos[NUM_LEGS] = {
    //                 {0.12, -0.188, 0.0},
    //                 {-0.12, -0.188, 0.0},
    //                 {0.12, 0.188, 0.0},
    //                 {-0.12, 0.188, 0.0}
    //             };

    // Eigen::Vector3d legs_origin[NUM_LEGS] = {{0.185, 0.0628, 0.0},
    //                 {0.185, -0.0628, 0.0},
    //                 {-0.185, 0.0628, 0.0},
    //                 {-0.185, -0.0628, 0.0}};

};

} // namespace quadro
