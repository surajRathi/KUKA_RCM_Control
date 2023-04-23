#! /usr/bin/python3
import numpy as np
import rospy
from geometry_msgs.msg import Point, Quaternion, Vector3
from sensor_msgs.msg import JointState
from std_msgs.msg import Header, ColorRGBA
from visualization_msgs.msg import MarkerArray, Marker

from iiwa_run.helper.specifications import from_param


class JointPlayer:
    def __init__(self, path_player="path_player", joint_topic='/joint_states', viz_topic='/viz/volumes', wall_dt=0.1):
        self.wall_dt = wall_dt

        rospy.init_node(path_player)

        self.joint_pub = rospy.Publisher(joint_topic, JointState, queue_size=5)

        self.viz_pub = rospy.Publisher(viz_topic, MarkerArray, queue_size=1, latch=True)
        self.show_rcm()

        self.active_joints = [
            'iiwa_joint_1',
            'iiwa_joint_2',
            'iiwa_joint_3',
            'iiwa_joint_4',
            'iiwa_joint_5',
            'iiwa_joint_6',
            'iiwa_joint_7'
        ]

    def show_rcm(self):
        spec = from_param()

        # RCM
        m = Marker()
        m.header.frame_id = "Insertion_Pose"
        m.header.stamp = rospy.Time.now()
        m.ns = "viz"
        m.id = 0
        m.type = Marker.CYLINDER
        m.action = Marker.ADD
        m.pose.position = Point(x=0, y=0, z=0)
        m.pose.orientation = Quaternion(w=1)
        m.scale = Vector3(x=2 * spec.R, y=2 * spec.R, z=0.001)
        m.color = ColorRGBA(r=1, g=0, b=0, a=0.5)

        marker_array = MarkerArray()
        marker_array.markers = [m]
        self.viz_pub.publish(marker_array)

    def play(self, joints):
        for joint in joints:
            if np.isnan(joint).any():
                rospy.logerr("Path Contains NaNs, stopping.")
                break

            js = JointState(header=Header(stamp=rospy.Time.now()),
                            name=self.active_joints,
                            position=joint)

            self.joint_pub.publish(js)
            rospy.rostime.wallsleep(self.wall_dt)
            if rospy.is_shutdown():
                break

        else:
            return True

        return False


def main():
    filename = 'joint_vals.npy'
    joints = np.load(filename)

    p = JointPlayer()
    # p = JointPlayer(joint_topic='/move_group/fake_controller_joint_states')

    if p.play(joints):
        rospy.loginfo("Path Successfully Played")
    else:
        rospy.logwarn("Failed")


if __name__ == '__main__':
    main()
