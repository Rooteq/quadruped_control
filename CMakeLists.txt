cmake_minimum_required(VERSION 3.16)
project(quadruped_control)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Use the LLVM linker for lightning-fast linking
# add_link_options("-fuse-ld=lld")

find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)

add_executable(control_node src/control_node.cpp)
add_executable(test_node src/test_node.cpp)

add_subdirectory(src/controller)

target_include_directories(control_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
  src/controller/inc
)

target_include_directories(test_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
  src/controller/inc
)

target_link_libraries(control_node controller)
target_link_libraries(test_node controller)

# Precompile ROS2 core headers for the executables
target_precompile_headers(control_node PRIVATE <rclcpp/rclcpp.hpp>)
target_precompile_headers(test_node PRIVATE <rclcpp/rclcpp.hpp>)

target_compile_features(control_node PUBLIC cxx_std_17)
target_compile_features(test_node PUBLIC cxx_std_17)

ament_target_dependencies(control_node
  rclcpp
  sensor_msgs
  geometry_msgs
  nav_msgs
  ament_index_cpp
)

ament_target_dependencies(test_node
  rclcpp
  sensor_msgs
  geometry_msgs
  ament_index_cpp
  tf2
  tf2_ros
)

install(TARGETS
  control_node
  test_node
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY
    description
    config
    launch
  DESTINATION "share/${PROJECT_NAME}"
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# set_target_properties(controller PROPERTIES UNITY_BUILD ON)

ament_package()
