cmake_minimum_required(VERSION 3.0.2)
project(iiwa_run)

set(CATKIN_ENABLE_TESTING OFF)

find_package(catkin REQUIRED COMPONENTS
  rospy
  )


catkin_python_setup()

catkin_package()