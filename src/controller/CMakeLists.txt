add_library(controller STATIC 
    src/gait_scheduler.cpp
    src/quadro_controller.cpp
    src/inverse_kinematics.cpp
    src/model.cpp
    src/trajectory_generator.cpp
    src/mpc.cpp
    src/dynamic_controller.cpp
    src/dynamic_controller.cpp
    )

find_package(Eigen3 REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
find_package(pinocchio REQUIRED)
find_package(qpOASES REQUIRED)
find_package(casadi REQUIRED)

target_include_directories(controller PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/inc)

# Precompile heavy C++ templates to drastically speed up compilation
target_precompile_headers(controller PRIVATE
    <Eigen/Dense>
    <casadi/casadi.hpp>
    <pinocchio/fwd.hpp>
    <pinocchio/multibody/model.hpp>
    <pinocchio/multibody/data.hpp>
)

target_link_libraries(controller
    pinocchio::pinocchio
    qpOASES 
    casadi
    )