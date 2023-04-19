#! /usr/bin/python3
import numpy as np
import rospy
from geometry_msgs.msg import Point, Quaternion, Vector3
from sensor_msgs.msg import JointState
from std_msgs.msg import Header, ColorRGBA
from visualization_msgs.msg import MarkerArray, Marker

from iiwa_run.helper.specifications import from_param


def show_rcm(pub):
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
    pub.publish(marker_array)


def main():
    filename = 'joint_vals.npy'
    dt = 0.1
    joints = np.load(filename)

    rospy.init_node("path_player")
    # joint_pub = rospy.Publisher('/move_group/fake_controller_joint_states', JointState, queue_size=5)
    joint_pub = rospy.Publisher('/joint_states', JointState, queue_size=5)
    viz_pub = rospy.Publisher('/viz/volumes', MarkerArray, queue_size=1, latch=True)
    show_rcm(viz_pub)
    active_joints = [
        'iiwa_joint_1',
        'iiwa_joint_2',
        'iiwa_joint_3',
        'iiwa_joint_4',
        'iiwa_joint_5',
        'iiwa_joint_6',
        'iiwa_joint_7'
    ]

    for joint in joints:
        if np.isnan(joint).any():
            print("Partial Path Received, Quitting.")
            break

        js = JointState(header=Header(stamp=rospy.Time.now()),
                        name=active_joints,
                        position=joint)

        joint_pub.publish(js)
        rospy.rostime.wallsleep(dt)

    else:
        print("Path Successfully Played")


if __name__ == '__main__':
    main()
