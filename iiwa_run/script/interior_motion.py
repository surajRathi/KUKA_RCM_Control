#!/usr/bin/python3
import sys
import time
from pathlib import Path
from typing import Union

import geometry_msgs.msg
import moveit_commander
import moveit_msgs.msg
import rospkg
import rospy
import tf2_geometry_msgs
import tf2_ros
from moveit_msgs.msg import RobotState
from sensor_msgs.msg import JointState
from std_msgs.msg import Header

# zero_state = [0.0] * 7
# pre_insertion_state = [0.7719714393359073, -1.162405065007553, -1.128065382969054, -1.4368011875273539,
#                        -1.181244222973306, 2.0290526312957797, -0.7484887519715488]


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

    abdomen_pose = geometry_msgs.msg.PoseStamped()
    abdomen_pose.header.frame_id = "abdomen_base"
    transform1 = tf_buffer.lookup_transform(move_group.get_planning_frame(),
                                            abdomen_pose.header.frame_id,
                                            rospy.Time(0), rospy.Duration(5))
    abdomen_pose_transformed = tf2_geometry_msgs.do_transform_pose(abdomen_pose, transform1)
    p = Path(rospkg.RosPack().get_path('iiwa_needle_description')) / "meshes" / "visual" / "abdomen.obj"
    scene.clear()
    scene.add_mesh("abdomen", abdomen_pose_transformed, str(p), size=(0.001, 0.001, 0.001))
    # joint_state = JointState()
    # joint_state.header.stamp = rospy.Time.now()
    # joint_state.name = robot.get_active_joint_names(group=group_name)
    # joint_state.position = [0.0] * len(robot.get_active_joint_names(group=group_name))  # Assumes each joint has 1dof
    # moveit_robot_state = RobotState()
    # moveit_robot_state.joint_state = joint_state
    # move_group.set_start_state(moveit_robot_state)

    zero_joint_state = JointState(header=Header(stamp=rospy.Time.now()),
                                  name=robot.get_active_joint_names(group=group_name),
                                  position=[0.0] * len(
                                      robot.get_active_joint_names(group=group_name)),
                                  )
    insertion_pose = geometry_msgs.msg.PoseStamped()
    insertion_pose.header.frame_id = "Insertion_Pose"
    insertion_pose.pose.position.z -= 0.30
    insertion_pose.pose.orientation.w = 1
    transform = tf_buffer.lookup_transform(move_group.get_planning_frame(),
                                           insertion_pose.header.frame_id,
                                           rospy.Time(0), rospy.Duration(0))

    insertion_pose.pose.position.z += 0.05
    insertion_pose_transformed = tf2_geometry_msgs.do_transform_pose(insertion_pose, transform)

    move_group.set_start_state(RobotState(joint_state=zero_joint_state))
    move_group.set_pose_target(insertion_pose_transformed.pose)
    success, plan, planning_time, error_code = move_group.plan()
    rospy.loginfo(
        f"{'Successfully' if success else 'Failed'} plan for z={insertion_pose.pose.position.z:.2f} " + (
            f"in {planning_time:.4f}s, i.e {1.0 / planning_time:.2f} Hz" if success else f"with {error_code_to_string(error_code)}"
        ))

    # Note: using wall clock here to allow the joint_state to be published and received
    joint_pub = rospy.Publisher('/move_group/fake_controller_joint_states', JointState, queue_size=1)
    joint_pub.publish(zero_joint_state)
    time.sleep(0.2)

    move_group.execute(plan, wait=True)


# noinspection DuplicatedCode
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
