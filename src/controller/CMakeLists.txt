add_library(controller STATIC 
    src/gait_scheduler.cpp
    src/quadro_controller.cpp
    src/inverse_kinematics.cpp
    src/model.cpp
    src/trajectory_generator.cpp
    src/dynamic_controller.cpp
    )

find_package(Eigen3 REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
find_package(pinocchio REQUIRED)

target_include_directories(controller PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/inc)

target_link_libraries(controller
    pinocchio::pinocchio)