#!/usr/bin/python3
import kdl_parser_py.urdf as kdl_parser
import rospy
from PyKDL import ChainFkSolverPos_recursive, JntArray, Frame, ChainIkSolverPos_NR, ChainIkSolverVel_pinv

from no_print import NoPrint

# Insertion Position in World: [0.000, -0.500, 0.300]
rest_joints = [0.7974687140058682, -0.7059608762250215, -2.0557765983620215, -1.7324525588466884, -0.685107459772449,
               1.136261558580683, -1.0476528028679626]

x0_005_joints = [0.7689340476041333, -0.7210911824056456, -2.0337923890649736, -1.7324947791391565, -0.6571485308662672,
                 1.1218431579518005, -1.0715009958889714]


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

    fk_solver = ChainFkSolverPos_recursive(chain)

    # ChainIkSolverPos_NR_JL(chain: Chain, q_min: JntArray, q_max: JntArray, fksolver: ChainFkSolverPos, iksolver: ChainIkSolverVel, maxiter: int = 100, eps: float = epsilon)
    # ik_solver = ChainIkSolverPos_NR_JL(chain)

    # ChainIkSolverVel_pinv(chain: Chain, eps: float = 1e-05, maxiter: int = 150)
    ikv_solver = ChainIkSolverVel_pinv(chain)

    # ChainIkSolverPos_NR(chain: Chain, fksolver: ChainFkSolverPos, iksolver: ChainIkSolverVel, maxiter: int = 100, eps: float = epsilon):
    ik_solver = ChainIkSolverPos_NR(chain, fk_solver, ikv_solver)

    # JntArray(size: int) or JntArray(arg: JntArray)
    j_start = JntArray(len(rest_joints))
    for i, val in enumerate(rest_joints):
        j_start[i] = val

    j_end_target = JntArray(len(x0_005_joints))
    for i, val in enumerate(x0_005_joints):
        j_end_target[i] = val

    # http://docs.ros.org/en/hydro/api/orocos_kdl/html/classKDL_1_1ChainFkSolverPos__recursive.html
    # JntToCart(self, q_in: JntArray, p_out: Frame, segmentNr: int = -1):
    f = Frame()

    ret = (fk_solver.JntToCart(q_in=j_start, p_out=f))
    print(ret)
    print(f"Position: {f.p}\nRotation:\n{f.M.GetQuaternion()}\n")
    # print(f"Position: {f.p}\nRotation:\n{f.M}\n")
    # print(f"Rotation:\n{f.M.GetQuaternion()}\n")
    # print(f"Rotation:\n{f.M.GetEulerZYZ()}\n")
    # print(f"Rotation:\n{f.M.GetRPY()}\n")

    j_ik = JntArray(len(rest_joints))
    f = Frame()
    ret = (fk_solver.JntToCart(q_in=j_end_target, p_out=f))
    print(ret)
    print(f"Position: {f.p}\nRotation:\n{f.M.GetQuaternion()}\n")

    # f = Frame()
    # f.p.x = 0.0
    # f.p.y = -0.5
    # f.p.z = 0.2
    # Quaternion(x: float, y: float, z: float, w: float):
    f.M.Quaternion(x=0.0, y=0.0, z=0.0, w=0.0)
    # CartToJnt(self, q_init: JntArray, p_in: Frame, q_out: JntArray)
    ret = ik_solver.CartToJnt(j_start, f, j_ik)
    print(ret)
    print(j_ik)
    print(rest_joints)
    print(x0_005_joints)

    start_err = 0
    for i, val in enumerate(rest_joints):
        start_err += (j_ik[i] - val) ** 2

    target_err = 0
    for i, val in enumerate(x0_005_joints):
        target_err += (j_ik[i] - val) ** 2

    print(f"{start_err=}")
    print(f"{target_err=}")


if __name__ == '__main__':
    main()
