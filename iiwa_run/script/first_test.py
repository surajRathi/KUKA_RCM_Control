#!/usr/bin/python3

import sys
from pathlib import Path
from typing import Union

import geometry_msgs.msg
import moveit_commander
import moveit_msgs.msg
import rospkg
import rospy
import tf2_geometry_msgs
import tf2_ros
from sensor_msgs.msg import JointState


# from std_msgs.msg import String
# from moveit_commander.conversions import pose_to_list


def main():
    # Initialize ROS
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("test1")

    # tf2
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)

    # Initialize Moveit
    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()

    group_name = "All"
    move_group = moveit_commander.MoveGroupCommander(group_name)
    display_trajectory_publisher = rospy.Publisher(
        "/move_group/display_planned_path",
        moveit_msgs.msg.DisplayTrajectory,
        queue_size=20,
    )

    initialize_robot_and_scene(group_name, move_group, robot, scene, tf_buffer)

    insertion_pose = geometry_msgs.msg.PoseStamped()
    insertion_pose.header.frame_id = "Insertion_Pose"
    insertion_pose.pose.position.z -= 0.30
    insertion_pose.pose.orientation.w = 1
    transform = tf_buffer.lookup_transform(move_group.get_planning_frame(),
                                           insertion_pose.header.frame_id,
                                           rospy.Time(0), rospy.Duration(0))

    for i in range(10):
        if rospy.is_shutdown():
            break
        insertion_pose.pose.position.z += 0.05
        insertion_pose_transformed = tf2_geometry_msgs.do_transform_pose(insertion_pose, transform)
        move_group.set_pose_target(insertion_pose_transformed.pose)
        (success, plan, planning_time, error_code,) = move_group.plan()
        rospy.loginfo(
            f"{'Successfully' if success else 'Failed'} plan for z={insertion_pose.pose.position.z:.2f} " + (
                f"in {planning_time:.4f}s, i.e {1.0 / planning_time:.2f} Hz" if success else f"with {error_code_to_string(error_code)}"
            ))

        if not success:
            break
        success = move_group.execute(plan, wait=True)
        move_group.stop()  # ensures that there is no residual movement

        # It is always good to clear your targets after planning with poses.
        # Note: there is no equivalent function for clear_joint_value_targets().
        move_group.clear_pose_targets()

        if not success:
            rospy.logwarn(f"Failed to execute the plan.")
            break


def initialize_robot_and_scene(group_name, move_group, robot, scene, tf_buffer):
    # Print some simple diagnostics
    print("============ Planning frame: %s" % move_group.get_planning_frame())
    print("============ End effector link: %s" % move_group.get_end_effector_link())
    print("============ Available Planning Groups:", robot.get_group_names())
    # print("============ Printing robot state")
    # print(robot.get_current_state())
    # print("")
    # joint_goal = move_group.get_current_joint_values()
    # move_group.go(joint_goal, wait=True)
    # move_group.stop()  # ensures that there is no residual movement
    # Initialize the Scene:
    # The abdomen should be placed at the frame: "abdomen_base"
    abdomen_pose = geometry_msgs.msg.PoseStamped()
    abdomen_pose.header.frame_id = "abdomen_base"
    transform = tf_buffer.lookup_transform(move_group.get_planning_frame(),
                                           abdomen_pose.header.frame_id,
                                           rospy.Time(0), rospy.Duration(5))
    abdomen_pose_transformed = tf2_geometry_msgs.do_transform_pose(abdomen_pose, transform)
    p = Path(rospkg.RosPack().get_path('iiwa_needle_description')) / "meshes" / "visual" / "abdomen.obj"
    scene.clear()
    scene.add_mesh("abdomen", abdomen_pose_transformed, str(p), size=(0.001, 0.001, 0.001))
    # Go to the zero state
    # print(','.join([robot.get_joint(j).value().__str__() for j in robot.get_active_joint_names(group=group_name)]))
    joint_state = JointState()
    joint_state.header.stamp = rospy.Time.now()
    joint_state.name = robot.get_active_joint_names(group=group_name)
    joint_state.position = [0.0] * len(robot.get_active_joint_names(group=group_name))  # Assumes each joint has 1dof
    move_group.go(joints=joint_state)
    print("============ Reached Start Position")


def error_code_to_string(code: Union[moveit_msgs.msg.MoveItErrorCodes, int]):
    if isinstance(code, moveit_msgs.msg.MoveItErrorCodes):
        code = code.val
    codes = moveit_msgs.msg.MoveItErrorCodes
    for name, val in codes.__dict__.items():
        if val == code:
            return name
    rospy.logwarn(f"Invalid moveit error code: {code}")
    return 'InvalidMOVEITError'


if __name__ == '__main__':
    main()
