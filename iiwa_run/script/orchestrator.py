#!/usr/bin/python3
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Union, Optional

import geometry_msgs.msg
import moveit_commander
import moveit_msgs.msg
import numpy as np
import rospkg
import rospy
import tf2_geometry_msgs
import tf2_ros
from geometry_msgs.msg import Pose, PoseStamped, Quaternion, Vector3
from moveit_msgs.msg import RobotState
from sensor_msgs.msg import JointState
from std_msgs.msg import Header, ColorRGBA
from tf.transformations import quaternion_about_axis, quaternion_multiply
from visualization_msgs.msg import Marker

INITIAL_HEIGHT = 0.300
SECOND_HEIGHT = 0.100
FIRST_DEPTH = 0.05
NEEDLE_LENGTH = 0.300
FIRST_TRANSLATION = 0.05


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

        self.fake_obs_pub = rospy.Publisher("/viz/fake_obstacle", Marker, queue_size=5, latch=True)

        self.add_abdomen()

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

        self.inserted_joint_state = JointState(header=Header(stamp=rospy.Time.now()),
                                               name=active_joints,
                                               position=[0.7974687140058682, -0.7059608762250215, -2.0557765983620215,
                                                         -1.7324525588466884, -0.685107459772449, 1.136261558580683,
                                                         -1.0476528028679626])

        self.move_group.set_end_effector_link("tool_link_ee")

        self.goal_pose = rospy.Publisher('/viz/goal_pose', PoseStamped, queue_size=1)

    def add_abdomen(self, as_collision_object=False):
        abdomen_pose = geometry_msgs.msg.PoseStamped()
        abdomen_pose.header.frame_id = "abdomen_base"
        transform1 = self.tf_buffer.lookup_transform(self.move_group.get_planning_frame(),
                                                     abdomen_pose.header.frame_id,
                                                     rospy.Time(0), rospy.Duration(5))
        abdomen_pose_transformed = tf2_geometry_msgs.do_transform_pose(abdomen_pose, transform1)
        p = Path(rospkg.RosPack().get_path('iiwa_needle_description')) / "meshes" / "visual" / "abdomen.obj"
        self.scene.clear()

        msg = Marker()
        msg.ns = "btp_rcm"
        msg.id = 17
        msg.action = msg.DELETE
        self.fake_obs_pub.publish(msg)

        if as_collision_object:
            self.scene.add_mesh("abdomen", abdomen_pose_transformed, str(p), size=(0.001, 0.001, 0.001))
        else:
            msg = Marker()
            msg.mesh_resource = "package://iiwa_needle_description/meshes/rviz/abdomen.stl"
            msg.mesh_use_embedded_materials = False  # Need this to use textures for mesh
            msg.color = ColorRGBA(r=0.58, g=0.76, b=1.0, a=0.1)
            msg.header.frame_id = "abdomen_base"
            msg.pose.orientation.w = 1.0

            msg.ns = "btp_rcm"
            msg.id = 17
            msg.action = msg.ADD
            msg.type = msg.MESH_RESOURCE
            msg.scale = Vector3(0.001, 0.001, 0.001)

            self.fake_obs_pub.publish(msg)

    def set_robot_state(self, state):
        # Note: using wall clock here to allow the joint_state to be published and received
        joint_pub = rospy.Publisher('/move_group/fake_controller_joint_states', JointState, queue_size=5)
        for i in range(5):
            joint_pub.publish(state)
            rospy.rostime.wallsleep(0.1)
        joint_pub.unregister()

    def transform_pose(self, msg: geometry_msgs.msg.PoseStamped) -> Pose:
        transform = self.tf_buffer.lookup_transform(self.move_group.get_planning_frame(),
                                                    msg.header.frame_id,
                                                    rospy.Time(0), rospy.Duration(0))
        return tf2_geometry_msgs.do_transform_pose(msg, transform).pose

    def transform_pose_stamped(self, msg: geometry_msgs.msg.PoseStamped) -> PoseStamped:
        transform = self.tf_buffer.lookup_transform(self.move_group.get_planning_frame(),
                                                    msg.header.frame_id,
                                                    rospy.Time(0), rospy.Duration(0))
        return tf2_geometry_msgs.do_transform_pose(msg, transform)

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

    def execute(self, plan) -> Optional[float]:
        t_start = time.time()
        success = self.move_group.execute(plan, wait=True)
        return time.time() - t_start if success else None

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

    def cartesian_plan_and_execute(self, pose_list, msg):
        t_start = time.time()
        path, fraction = self.move_group.compute_cartesian_path(pose_list, 1e-3, 0.0)
        planning_time = time.time() - t_start
        success = (1 - fraction) < 1e-9
        rospy.loginfo(f"{'Successfully' if success else 'Failed'} plan {msg}" + (
            f"in {planning_time:.4f}s, i.e {1.0 / planning_time:.2f} Hz" if success else f"with fraction: {fraction}"
        ))
        if not success:
            raise MoveitFailure()
        # def retime_trajectory(self, ref_state_in, traj_in, velocity_scaling_factor=1.0, acceleration_scaling_factor=1.0, algorithm="iterative_time_parameterization"):
        # chrome-extension://oemmndcbldboiebfnladdacbdfmadadm/https://picknik.ai/docs/moveit_workshop_macau_2019/TOTG.pdfk
        t_exec = self.execute(path)
        if t_exec is not None:
            rospy.loginfo(f'Executed trajectory in {t_exec}s.')
        else:
            rospy.loginfo(f'Trajectory execution failed.')
            raise MoveitFailure()

    def multiple_cartesian_plan_and_execute(self, pose_list, soft_timeout=5, msg=''):
        t_start = time.time()
        n_tries = 0.0
        sum_fractions = 0.0
        while (time.time() - t_start) < soft_timeout:
            path, fraction = self.move_group.compute_cartesian_path(pose_list, 1e-3, 0.0)
            n_tries += 1
            sum_fractions += fraction
            success = (1 - fraction) < 1e-9
            if success:
                break
        else:
            planning_time = time.time() - t_start
            rospy.loginfo(
                f"Timed out for plan {msg} in {planning_time:.2f}s and {n_tries} tries with average frac: {sum_fractions / n_tries:.4f}"
            )
            raise MoveitFailure()

        planning_time = time.time() - t_start

        rospy.loginfo(
            f"Successfully planned {msg} in {planning_time:.2f}s and {n_tries} tries with fraction: {fraction:.4f}"

        )

        t_exec = self.execute(path)
        if t_exec is not None:
            rospy.loginfo(f'Executed trajectory in {t_exec}s.')
        else:
            rospy.loginfo(f'Trajectory execution failed.')
            raise MoveitFailure()


def insertion_routine(orc):
    orc.set_robot_state(orc.zero_joint_state)

    insertion_pose = geometry_msgs.msg.PoseStamped()
    insertion_pose.header.frame_id = "Insertion_Pose"
    insertion_pose.pose.orientation.w = 1

    try:
        insertion_pose.pose.position.z = -INITIAL_HEIGHT
        orc.move_group.set_start_state(RobotState(joint_state=orc.zero_joint_state))
        orc.goal_pose.publish(insertion_pose)
        orc.plan_and_execute(target=orc.transform_pose(insertion_pose),
                             msg=f" for z={insertion_pose.pose.position.z:.2f}")

        insertion_pose.pose.position.z = -SECOND_HEIGHT
        orc.move_group.set_start_state_to_current_state()
        orc.goal_pose.publish(insertion_pose)
        orc.plan_and_execute(target=orc.transform_pose(insertion_pose),
                             msg=f" for z={insertion_pose.pose.position.z:.2f}")

        pose_list = [orc.transform_pose(insertion_pose)]
        insertion_pose.pose.position.z = FIRST_DEPTH
        pose_list.append(orc.transform_pose(insertion_pose))
        orc.move_group.set_start_state_to_current_state()
        orc.goal_pose.publish(insertion_pose)
        orc.cartesian_plan_and_execute(pose_list,
                                       msg=f" for z={insertion_pose.pose.position.z:.2f}")

        pose_stamped = orc.move_group.get_current_pose()
        pose_list = [deepcopy(pose_stamped.pose)]
        x0 = pose_stamped.pose.position.x
        pose_stamped.pose.position.x += 0.005
        pose_list.append(pose_stamped.pose)
        orc.move_group.set_start_state_to_current_state()
        orc.goal_pose.publish(pose_stamped)
        orc.cartesian_plan_and_execute(pose_list,
                                       msg=f" for Δx={pose_stamped.pose.position.x - x0:.2f}")

    except MoveitFailure:
        rospy.logerr("Planning pipeline failed.")


def interior_motion_routine(orc):
    orc.set_robot_state(orc.inserted_joint_state)

    insertion_pose = PoseStamped()
    insertion_pose.header.frame_id = "Insertion_Pose"
    insertion_pose.pose.orientation.w = 1.0
    try:
        pose_list = [deepcopy(orc.move_group.get_current_pose().pose)]
        target_pose = orc.move_group.get_current_pose().pose
        x0 = target_pose.position.x
        target_pose.position.x += FIRST_TRANSLATION
        target_pose.orientation = get_target_orientation(orc.transform_pose_stamped(insertion_pose),
                                                         target_pose.position)
        pose_list.append(target_pose)

        pp = orc.move_group.get_current_pose()
        pp.pose = target_pose
        orc.goal_pose.publish(pp)

        orc.move_group.set_start_state_to_current_state()
        orc.multiple_cartesian_plan_and_execute(pose_list, msg=f" for Δx={target_pose.position.x - x0:.2f}")

        pose_list = [deepcopy(orc.move_group.get_current_pose().pose)]
        target_pose = orc.move_group.get_current_pose().pose
        y0 = target_pose.position.y
        target_pose.position.y -= FIRST_TRANSLATION
        target_pose.orientation = get_target_orientation(orc.transform_pose_stamped(insertion_pose),
                                                         target_pose.position)
        pose_list.append(target_pose)

        pp = orc.move_group.get_current_pose()
        pp.pose = target_pose
        orc.goal_pose.publish(pp)

        orc.move_group.set_start_state_to_current_state()
        orc.multiple_cartesian_plan_and_execute(pose_list, msg=f" for Δy={target_pose.position.y - y0:.2f}")

    except MoveitFailure:
        rospy.logerr("Planning pipeline failed.")

    rospy.spin()


def get_target_orientation(insertion_pose, target_point) -> Quaternion:
    insertion_point = insertion_pose.pose.position
    dx = target_point.x - insertion_point.x
    dy = target_point.y - insertion_point.y
    dz = target_point.z - insertion_point.z

    orig = np.array((0, 0, -1))
    new = np.array((dx, dy, dz))
    new /= np.linalg.norm(new)

    n_c = np.cross(orig, new)
    n_c /= np.linalg.norm(n_c)

    theta = np.arccos(np.dot(orig, new))
    q_rot = quaternion_about_axis(axis=n_c, angle=theta)

    o = insertion_pose.pose.orientation
    q_initial = np.array((o.x, o.y, o.z, o.w))
    q_net = quaternion_multiply(q_rot, q_initial)

    return Quaternion(x=q_net[0], y=q_net[1], z=q_net[2], w=q_net[3])


def main():
    orc = Orchestrator()
    insertion_routine(orc)


if __name__ == '__main__':
    main()
