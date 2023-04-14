#! /usr/bin/python3
import numpy as np
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Header


def main():
    filename = 'joint_vals.npy'
    dt = 0.1
    joints = np.load(filename)

    rospy.init_node("path_player")
    joint_pub = rospy.Publisher('/move_group/fake_controller_joint_states', JointState, queue_size=5)
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
