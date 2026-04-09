import os
import xacro

# TODO: fix use sim time as a settable parameter

from ament_index_python.packages import get_package_share_directory
from pathlib import Path
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
)
from launch.actions import RegisterEventHandler, SetEnvironmentVariable, TimerAction, Shutdown, OpaqueFunction
from launch.event_handlers import OnProcessExit, OnProcessStart
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def launch_setup(context, *args, **kwargs):

    pkg_path = os.path.join(get_package_share_directory("quadruped_control"))

    # Check if we're told to use sim time
    use_sim_time = LaunchConfiguration("use_sim_time", default=True)

    xacro_file = os.path.join(pkg_path, "description", "quadro.xacro")
    robot_controllers = os.path.join(pkg_path, "config", "controllers.yaml")    
    robot_desc = Command(["xacro ", xacro_file])

    robot_state_publisher_params = {
        "robot_description": robot_desc,
        "use_sim_time": use_sim_time,
    }

    nodes = []

    # Create a robot_state_publisher node
    nodes.append(Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[{"use_sim_time": use_sim_time}, robot_state_publisher_params],
    ))

    nodes.append(Node(
        package="mujoco_ros2_control",
        executable="ros2_control_node",
        emulate_tty="True",
        output="both",
        parameters=[
            {"use_sim_time": True},
            {"robot_description": robot_desc},
            robot_controllers,
        ],
        on_exit=Shutdown()
    ))

    controllers_to_spawn = ["joint_state_broadcaster", "forward_effort_controller"]
    for controller in controllers_to_spawn:
       nodes.append(
            Node(
                package="controller_manager",
                executable="spawner",
                arguments=[controller, "--param-file", robot_controllers],
                output="both",
            )
        ) 

    rviz_config_file = os.path.join(pkg_path, "config", "rviz_config.rviz")

    nodes.append(Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_config_file],
    ))

    torque_controller = Node(
        package='quadro',
        executable='torque_controller',
        name='torque_controller',
        arguments=[robot_desc, 'link'],
        output='screen'
    )

    return nodes

    # return LaunchDescription(
    #     [
    #         DeclareLaunchArgument(
    #             'urdf_file',
    #             default_value=xacro_file,
    #             description='Path to URDF file'
    #         ),
            
    #         DeclareLaunchArgument(
    #             'end_effector_frame',
    #             default_value='link',
    #             description='Name of end-effector frame'
    #         ),

    #         node_robot_state_publisher,
    #         rviz,
    #     ]
    # )

def generate_launch_description():
    return LaunchDescription(
        [
            OpaqueFunction(function=launch_setup),
        ]
    )