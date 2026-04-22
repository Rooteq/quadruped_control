import sys

def patch_file(fname):
    with open(fname, 'r') as f:
        content = f.read()

    # Apply changes to header
    if fname.endswith('.hpp'):
        content = content.replace(
            "const Eigen::Vector3d& body_velocity) const;",
            "const Eigen::Vector3d& current_vel,\n        const Eigen::Vector3d& desired_vel) const;"
        )
        content = content.replace(
            "const Eigen::Vector3d& desired_angular_vel);",
            "const Eigen::Vector3d& desired_angular_vel,\n        const Eigen::Vector3d& current_vel);"
        )

    # Apply changes to source
    if fname.endswith('.cpp'):
        content = content.replace(
            "const Eigen::Vector3d& /*desired_angular_vel*/)",
            "const Eigen::Vector3d& /*desired_angular_vel*/,\n    const Eigen::Vector3d& current_vel)"
        )
        
        # update the call to computeLandingPos
        content = content.replace(
            "swing_states_[leg].landing_pos = computeLandingPos(\n                    model, gait, leg, base_vel_w);",
            "swing_states_[leg].landing_pos = computeLandingPos(\n                    model, gait, leg, current_vel, desired_linear_vel);"
        )

        old_compute = """Eigen::Vector3d TrajectoryGenerator::computeLandingPos(
    const QuadroModel& model,
    const GaitScheduler& gait,
    int leg_idx,
    const Eigen::Vector3d& body_velocity) const
{
    // Raibert heuristic: land under hip + velocity compensation
    const Eigen::VectorXd& state = model.stateVector();
    Eigen::Vector3d base_pos(state[3], state[4], 0.0);
    Eigen::Matrix3d R_z = model.bodyYawRotation();

    // Rotate hip offset into world and add base position
    Eigen::Vector3d hip_ground = base_pos + R_z * hipPos[leg_idx];
    hip_ground.z() = NOMINAL_HEIGHT;

    double stance_duration = gait.gait().duty_cycle * gait.gait().period;
    Eigen::Vector3d landing = hip_ground + body_velocity * (stance_duration / 2.0);
    landing.z() = NOMINAL_HEIGHT;

    return landing;
}"""
        
        new_compute = """Eigen::Vector3d TrajectoryGenerator::computeLandingPos(
    const QuadroModel& model,
    const GaitScheduler& gait,
    int leg_idx,
    const Eigen::Vector3d& current_vel,
    const Eigen::Vector3d& desired_vel) const
{
    // Raibert heuristic matching Python reference implementation
    const Eigen::VectorXd& state = model.stateVector();
    Eigen::Vector3d base_pos(state[3], state[4], 0.0);
    Eigen::Matrix3d R_z = model.bodyYawRotation();

    Eigen::Vector3d hip_pos_world = base_pos + R_z * hipPos[leg_idx];
    
    double t_swing = (1.0 - gait.gait().duty_cycle) * gait.gait().period;
    double t_stance = gait.gait().duty_cycle * gait.gait().period;
    double T = t_swing + 0.5 * t_stance;
    double pred_time = T / 2.0;

    double k_v_x = 0.4 * T;
    double k_v_y = 0.2 * T;

    Eigen::Vector3d pos_nominal(hip_pos_world.x(), hip_pos_world.y(), 0.02);
    Eigen::Vector3d drift(desired_vel.x() * pred_time, desired_vel.y() * pred_time, 0.0);
    Eigen::Vector3d vel_correction(k_v_x * (current_vel.x() - desired_vel.x()), 
                                   k_v_y * (current_vel.y() - desired_vel.y()), 
                                   0.0);

    return pos_nominal + drift + vel_correction;
}"""
        content = content.replace(old_compute, new_compute)

    with open(fname, 'w') as f:
        f.write(content)

patch_file('src/controller/inc/trajectory_generator.hpp')
patch_file('src/controller/src/trajectory_generator.cpp')
