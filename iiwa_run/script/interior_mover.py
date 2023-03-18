#! /usr/bin/python3
import numpy as np
import rospy
from geometry_msgs.msg import Quaternion, PoseStamped, Point
from tf.transformations import quaternion_about_axis, quaternion_multiply

from orchestrator import Orchestrator, MoveitFailure


class InteriorMover:
    def __init__(self, o: Orchestrator = None):
        self.orc = Orchestrator() if o is None else o
        self.orc.set_robot_state(self.orc.inserted_joint_state)

        self.insertion_pose = PoseStamped()
        self.insertion_pose.header.frame_id = "Insertion_Pose"
        self.insertion_pose.pose.orientation.w = 1.0
        self.insertion_pose = self.orc.transform_pose(self.insertion_pose)

        self.to_execute = False
        self.target_point = self.orc.move_group.get_current_pose().pose.position

    def __enter__(self) -> Point:
        return self.target_point

    def __exit__(self, exc_type, exc_val, exc_tb):
        pose_list = [self.orc.move_group.get_current_pose().pose]

        target = self.orc.move_group.get_current_pose()
        target.pose.position = self.target_point
        target.pose.orientation = self.get_target_orientation(self.target_point)

        pose_list.append(target.pose)

        self.orc.goal_pose.publish(target)

        self.orc.move_group.set_start_state_to_current_state()
        self.orc.multiple_cartesian_plan_and_execute(pose_list, msg=f" for Δ")

    def get_target_orientation(self, target_point: Point) -> Quaternion:
        insertion_point = self.insertion_pose.position
        dx = target_point.x - insertion_point.x
        dy = target_point.y - insertion_point.y
        dz = target_point.z - insertion_point.z

        if abs(dz) < 1e-9:
            if abs(dx) < 1e-9 and abs(dy) < 1e-9:
                dx = 0
                dy = 0
                dz = 1
            else:
                rospy.logerr(
                    f"Invalid interior point: {target_point} compared to insertion point: {self.insertion_pose.position}")
                raise MoveitFailure

        orig = np.array((0, 0, -1))
        new = np.array((dx, dy, dz))
        new /= np.linalg.norm(new)

        n_c = np.cross(orig, new)
        n_c /= np.linalg.norm(n_c)

        theta = np.arccos(np.dot(orig, new))
        q_rot = quaternion_about_axis(axis=n_c, angle=theta)

        o = self.insertion_pose.orientation
        q_initial = np.array((o.x, o.y, o.z, o.w))
        q_net = quaternion_multiply(q_rot, q_initial)

        return Quaternion(x=q_net[0], y=q_net[1], z=q_net[2], w=q_net[3])


def main():
    mover = InteriorMover()

    with mover as pt:
        pt.z += 0.05

    with mover as pt:
        pt.x += 0.025

    with mover as pt:
        pt.y += 0.025

    with mover as pt:
        pt.x -= 0.05

    with mover as pt:
        pt.y -= 0.05


if __name__ == '__main__':
    main()
