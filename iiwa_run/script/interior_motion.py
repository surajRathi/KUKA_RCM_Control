#!/usr/bin/python3
import sys
import time
from pathlib import Path
from typing import Union, Optional

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

INITIAL_HEIGHT = 0.300
SECOND_HEIGHT = 0.100
FIRST_DEPTH = 0.100
NEEDLE_LENGTH = 0.300


class MoveitFailure(Exception):
    pass


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

        self.move_group.set_end_effector_link("tool_link_ee")

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
        # Note: using wall clock here to allow the joint_state to be published and received
        print(self.move_group.get_current_joint_values())

        joint_pub = rospy.Publisher('/move_group/fake_controller_joint_states', JointState, queue_size=5)
        for i in range(5):
            joint_pub.publish(self.zero_joint_state)
            rospy.rostime.wallsleep(0.1)

        print(self.move_group.get_current_joint_values())

        insertion_pose = geometry_msgs.msg.PoseStamped()
        insertion_pose.header.frame_id = "Insertion_Pose"
        insertion_pose.pose.orientation.w = 1

        try:
            insertion_pose.pose.position.z = -INITIAL_HEIGHT
            self.move_group.set_start_state(RobotState(joint_state=self.zero_joint_state))
            self.plan_and_execute(target=self.transform_pose(insertion_pose),
                                  msg=f" for z={insertion_pose.pose.position.z:.2f}")

            insertion_pose.pose.position.z = -SECOND_HEIGHT
            self.move_group.set_start_state_to_current_state()
            self.plan_and_execute(target=self.transform_pose(insertion_pose),
                                  msg=f" for z={insertion_pose.pose.position.z:.2f}")

            from copy import deepcopy

            pose_list = [self.transform_pose(insertion_pose)]
            insertion_pose.pose.position.z = FIRST_DEPTH
            pose_list.append(self.transform_pose(insertion_pose))
            self.move_group.set_start_state_to_current_state()

            path, fraction = self.move_group.compute_cartesian_path(pose_list, 1e-3, 0.0)
            print(fraction)

            # def retime_trajectory(self, ref_state_in, traj_in, velocity_scaling_factor=1.0, acceleration_scaling_factor=1.0, algorithm="iterative_time_parameterization"):
            # chrome-extension://oemmndcbldboiebfnladdacbdfmadadm/https://picknik.ai/docs/moveit_workshop_macau_2019/TOTG.pdfk
            t_exec = self.execute(path)
            if t_exec is not None:
                rospy.loginfo(f'Executed trajectory in {t_exec}s.')
            else:
                rospy.loginfo(f'Trajectory execution failed.')
                raise MoveitFailure()



        except MoveitFailure:
            rospy.logerr("Planning pipeline failed.")

    def plan_and_execute(self, target=None, msg=''):
        success, plan, planning_time, error_code = self.move_group.plan(target)
        rospy.loginfo(
            f"{'Successfully' if success else 'Failed'} plan {msg}" + (
                f"in {planning_time:.4f}s, i.e {1.0 / planning_time:.2f} Hz" if success else f"with {self.error_code_to_string(error_code)}"
            ))
        if success:
            t_exec = self.execute(plan)
            if t_exec is not None:
                rospy.loginfo(f'Executed trajectory in {t_exec}s.')
            else:
                rospy.loginfo(f'Trajectory execution failed.')
                raise MoveitFailure()
        else:
            raise MoveitFailure()

    def execute(self, plan) -> Optional[float]:
        t_start = time.time()
        success = self.move_group.execute(plan, wait=True)
        return time.time() - t_start if success else None

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
