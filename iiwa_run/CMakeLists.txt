cmake_minimum_required(VERSION 3.0.2)
project(iiwa_run)

set(CATKIN_ENABLE_TESTING OFF)
add_compile_options(-std=c++17)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  moveit_core  # ?
  moveit_ros_planning
  moveit_ros_planning_interface
  tf2_ros
  geometry_msgs
  )

find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED system filesystem date_time thread)

catkin_package(
  #  INCLUDE_DIRS include
  #  LIBRARIES iiwa_run
  CATKIN_DEPENDS
  moveit_core
  moveit_visual_tools
  moveit_ros_planning_interface
  #      interactive_markers
  #      tf2_geometry_msgs
  DEPENDS
  EIGEN3
)

include_directories(
  # include
  SYSTEM ${catkin_INCLUDE_DIRS} ${Boost_INCLUDE_DIR} ${EIGEN3_INCLUDE_DIRS}
)

## Declare a C++ library
# add_library(${PROJECT_NAME}
#   src/${PROJECT_NAME}/iiwa_run.cpp
# )
# add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})


add_executable(test1 src/test1.cpp)
set_target_properties(test1 PROPERTIES OUTPUT_NAME node PREFIX "")
add_dependencies(test1 ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(test1
  ${catkin_LIBRARIES}
  )

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_iiwa_run.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
