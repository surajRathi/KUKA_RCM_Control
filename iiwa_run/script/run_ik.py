#!/usr/bin/python3
import kdl_parser_py.urdf as kdl_parser
import rospy
from PyKDL import ChainFkSolverPos_recursive, JntArray, Frame

from no_print import NoPrint


def main():
    rospy.init_node("kdl_test")

    # success, tree = kdl_parser.treeFromParam("/robot_description_semantic")
    with NoPrint(stdout=True, stderr=True):
        success, tree = kdl_parser.treeFromParam("/robot_description")
    if not success:
        rospy.logfatal("Could not load the kdl_tree from the urdf.")
        return

    chain = tree.getChain('iiwa_link_0', 'tool_link_ee')
    print(tree.getNrOfJoints(), tree.getNrOfSegments())

    j_vals = [0.7974687140058682, -0.7059608762250215, -2.0557765983620215, -1.7324525588466884, -0.685107459772449,
              1.136261558580683, -1.0476528028679626]

    #  JntArray(size: int): not enough arguments
    # JntArray(arg: JntArray): not enough arguments
    print("A")
    j = JntArray(len(j_vals))

    for i, val in enumerate(j_vals):
        j[i] = val

    # http://docs.ros.org/en/hydro/api/orocos_kdl/html/classKDL_1_1ChainFkSolverPos__recursive.html
    # JntToCart(self, q_in: JntArray, p_out: Frame, segmentNr: int = -1):
    f = Frame()
    print(ChainFkSolverPos_recursive(chain).JntToCart(q_in=j, p_out=f))

    print()
    print(f.p)
    print()
    print(f.M)
    # for a in ('DH', 'DH_Craig1989', 'Identity', 'Integrate', 'Inverse'):
    # for a in ('Identity', 'Integrate', 'Inverse'):
    #     print(getattr(f, a)())
    # print(dir(f))


if __name__ == '__main__':
    main()
