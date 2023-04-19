#! /usr/bin/python3
import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import Quaternion, PoseStamped, Point, Vector3, TransformStamped
from sensor_msgs.msg import Range
from std_msgs.msg import ColorRGBA
from tf.transformations import quaternion_about_axis, quaternion_multiply
from visualization_msgs.msg import Marker, MarkerArray

from iiwa_run.moveit_orchestrator import MoveitOrchestrator, MoveitFailure
from iiwa_run.helper.specifications import from_param


class DirectIKMover:
    def __init__(self, o: MoveitOrchestrator = None):
        self.spec = from_param('/spec')

        self.orc = MoveitOrchestrator() if o is None else o
        self.orc.set_robot_state(self.orc.create_joint_state(self.spec.rest_joint_states))

        insertion_pose = PoseStamped()
        insertion_pose.header.frame_id = "Insertion_Pose"
        insertion_pose.pose.orientation.w = 1.0
        self.insertion_pose = self.orc.transform_pose(insertion_pose)

        self.to_execute = False
        self.target_point = self.orc.move_group.get_current_pose().pose.position

        # Note: All measurements in mm
        self.viz_pub = rospy.Publisher('/viz/volumes', MarkerArray, queue_size=1, latch=True)
        self.viz_range_pub = rospy.Publisher('/viz/range', Range, queue_size=1, latch=True)
        self.viz_tf = tf2_ros.StaticTransformBroadcaster()

        self.publish_viz()

    def publish_viz(self):
        trans = TransformStamped()
        trans.header.stamp = rospy.Time.now()
        trans.header.frame_id = "Insertion_Pose"
        trans.child_frame_id = "Workspace_Range"
        trans.transform.rotation.y = -1 / np.sqrt(2)
        trans.transform.rotation.w = 1 / np.sqrt(2)
        self.viz_tf.sendTransform(trans)

        r = Range()
        r.header.frame_id = "Workspace_Range"
        r.header.stamp = rospy.Time.now()
        r.min_range = 0
        r.max_range = 2
        r.radiation_type = r.ULTRASOUND  # ???
        # r.range = self.spec.l2
        r.range = self.spec.H2
        r.field_of_view = 2 * self.spec.theta
        self.viz_range_pub.publish(r)

        spec = self.spec

        marker_array = MarkerArray()
        marker_array.markers = []

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
        m.scale = Vector3(x=2 * spec.R, y=2 * spec.R, z=0)
        m.color = ColorRGBA(r=1, g=0, b=0, a=0.5)

        marker_array.markers.append(m)

        # Workspace
        m = Marker()
        m.header.frame_id = "Insertion_Pose"
        m.header.stamp = rospy.Time.now()
        m.ns = "viz"
        m.id = 1
        m.type = Marker.CYLINDER
        m.action = Marker.ADD
        m.pose.position = Point(x=0, y=0, z=spec.H1 + spec.H / 2)
        m.pose.orientation = Quaternion(w=1)
        m.scale = Vector3(x=2 * spec.rl, y=2 * spec.rl, z=spec.H)
        m.color = ColorRGBA(r=0, g=1, b=0, a=0.3)
        marker_array.markers.append(m)

        self.viz_pub.publish(marker_array)

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
        if np.isclose(n_c, 0).all():
            n_c = np.array([1.0, 0.0, 0.0])

        theta = np.arccos(np.dot(orig, new))
        q_rot = quaternion_about_axis(axis=n_c, angle=theta)

        o = self.insertion_pose.orientation
        q_initial = np.array((o.x, o.y, o.z, o.w))
        q_net = quaternion_multiply(q_rot, q_initial)

        return Quaternion(x=q_net[0], y=q_net[1], z=q_net[2], w=q_net[3])


def main():
    mover = DirectIKMover()

    # # To get the new initial joint position of different Spec params.
    # with mover as pt:
    #     pt.x, pt.y, pt.z = mover.spec.rcm
    #     pt.z -= (mover.spec.H1 + mover.spec.H)

    with mover as pt:
        pt.x += 20e-3

    rospy.spin()


if __name__ == '__main__':
    main()
