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
from geometry_msgs.msg import Pose
from moveit_msgs.msg import RobotState
from sensor_msgs.msg import JointState
from std_msgs.msg import Header


class Orchestrator:
    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node("test1")

        # tf2
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Initialize Moveit
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()

        self.group_name = "All"
        self.move_group = moveit_commander.MoveGroupCommander(self.group_name)

        self.enact_constraint()

        active_joints = self.robot.get_active_joint_names(group=self.group_name)
        self.zero_joint_state = JointState(header=Header(stamp=rospy.Time.now()),
                                           name=active_joints,
                                           position=[0.0] * len(active_joints))

        assert (len(active_joints) == 7)
        self.insertion_joint_state = JointState(header=Header(stamp=rospy.Time.now()),
                                                name=active_joints,
                                                position=[0.7719714393359073, -1.162405065007553, -1.128065382969054,
                                                          -1.4368011875273539, -1.181244222973306, 2.0290526312957797,
                                                          -0.7484887519715488])

    def enact_constraint(self):
        abdomen_pose = geometry_msgs.msg.PoseStamped()
        abdomen_pose.header.frame_id = "abdomen_base"
        transform1 = self.tf_buffer.lookup_transform(self.move_group.get_planning_frame(),
                                                     abdomen_pose.header.frame_id,
                                                     rospy.Time(0), rospy.Duration(5))
        abdomen_pose_transformed = tf2_geometry_msgs.do_transform_pose(abdomen_pose, transform1)
        p = Path(rospkg.RosPack().get_path('iiwa_needle_description')) / "meshes" / "visual" / "abdomen.obj"
        self.scene.clear()
        self.scene.add_mesh("abdomen", abdomen_pose_transformed, str(p), size=(0.001, 0.001, 0.001))

    def transform_pose(self, msg: geometry_msgs.msg.PoseStamped) -> Pose:
        transform = self.tf_buffer.lookup_transform(self.move_group.get_planning_frame(),
                                                    msg.header.frame_id,
                                                    rospy.Time(0), rospy.Duration(0))
        return tf2_geometry_msgs.do_transform_pose(msg, transform).pose

    def run(self):
        insertion_pose = geometry_msgs.msg.PoseStamped()
        insertion_pose.header.frame_id = "Insertion_Pose"
        insertion_pose.pose.orientation.w = 1

        insertion_pose.pose.position.z -= 0.30
        insertion_pose.pose.position.z += 0.05

        self.move_group.set_start_state(RobotState(joint_state=self.zero_joint_state))
        self.move_group.set_pose_target(self.transform_pose(insertion_pose))
        success, plan, planning_time, error_code = self.move_group.plan()

        rospy.loginfo(
            f"{'Successfully' if success else 'Failed'} plan for z={insertion_pose.pose.position.z:.2f} " + (
                f"in {planning_time:.4f}s, i.e {1.0 / planning_time:.2f} Hz" if success else f"with {self.error_code_to_string(error_code)}"
            ))

        self.execute(plan)

    def execute(self, plan):
        # Note: using wall clock here to allow the joint_state to be published and received
        joint_pub = rospy.Publisher('/move_group/fake_controller_joint_states', JointState, queue_size=1)
        joint_pub.publish(self.zero_joint_state)
        time.sleep(0.2)

        self.move_group.execute(plan, wait=True)

    # noinspection DuplicatedCode
    @staticmethod
    def error_code_to_string(code: Union[moveit_msgs.msg.MoveItErrorCodes, int]):
        if isinstance(code, moveit_msgs.msg.MoveItErrorCodes):
            code = code.val
        codes = moveit_msgs.msg.MoveItErrorCodes
        for name, val in codes.__dict__.items():
            if val == code:
                return name
        rospy.logwarn(f"Invalid moveit error code: {code}")
        return 'InvalidMOVEITError'


def main():
    orc = Orchestrator()
    orc.run()


if __name__ == '__main__':
    main()
