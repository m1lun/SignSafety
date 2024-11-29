cmake_minimum_required(VERSION 3.5)
project(driver_pkg)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

# Compiler options
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
foreach(PACKAGE 
  ament_cmake 
  ament_cmake_python
  rclcpp
  rclpy 
  geometry_msgs
  nav_msgs
  ackermann_msgs
  sensor_msgs
  std_msgs)

  find_package(${PACKAGE} REQUIRED)
endforeach()

# Include directories for C++ code
include_directories(include)

# Add C++ executable
add_executable(reactive_node src/reactive_node.cpp)
ament_target_dependencies(reactive_node 
  rclcpp 
  geometry_msgs 
  ackermann_msgs 
  nav_msgs 
  sensor_msgs 
  std_msgs
)

# Install C++ executables
install(TARGETS
  reactive_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install Python package
ament_python_install_package(${PROJECT_NAME})

# Install Python executables
install(PROGRAMS
  scripts/reactive_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# Testing setup (optional, but useful for CI/CD pipelines)
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

# Mark this package for ament
ament_package()
