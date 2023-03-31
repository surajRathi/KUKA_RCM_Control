#!/usr/bin/python3
import kdl_parser_py.urdf as kdl_parser
import numpy as np
from PyKDL import ChainFkSolverPos_recursive, JntArray, Frame, ChainIkSolverPos_NR, ChainIkSolverVel_pinv, Vector, \
    Rotation
from tf.transformations import quaternion_about_axis, quaternion_multiply

from no_print import NoPrint


class IKOrchestrator:
    def __init__(self):
        # Insertion Position in World: [0.000, -0.500, 0.300]
        self.insertion_pt = Vector()
        self.insertion_pt.x(0.0)
        self.insertion_pt.y(-0.5)
        self.insertion_pt.z(0.3)

        self.insertion_rot = Rotation().Quaternion(x=0.000, y=1.000, z=0.000, w=0.000)

        self.zero_state_joints = [0.7974687140058682, -0.7059608762250215, -2.0557765983620215, -1.7324525588466884,
                                  -0.685107459772449,
                                  1.136261558580683, -1.0476528028679626]

        self.nj = len(self.zero_state_joints)

    def do_ik(self, old_joints, new_frame):
        pass

    def get_target_orientation(self, target_point: Vector) -> Rotation:
        dx = target_point.x() - self.insertion_pt.x()
        dy = target_point.y() - self.insertion_pt.y()
        dz = target_point.z() - self.insertion_pt.z()

        if abs(dz) < 1e-9:
            if abs(dx) < 1e-9 and abs(dy) < 1e-9:
                dx = 0
                dy = 0
                dz = 1
            else:
                print(
                    f"Invalid interior point: {target_point} compared to insertion point: {self.insertion_pt}")
                raise RuntimeError()

        orig = np.array((0, 0, -1))
        new = np.array((dx, dy, dz))
        new /= np.linalg.norm(new)

        n_c = np.cross(orig, new)
        n_c /= np.linalg.norm(n_c)

        theta = np.arccos(np.dot(orig, new))
        q_rot = quaternion_about_axis(axis=n_c, angle=theta)

        q_x, q_y, q_z, q_w = self.insertion_rot.GetQuaternion()
        q_initial = np.array((q_x, q_y, q_z, q_w))
        q_net = quaternion_multiply(q_rot, q_initial)

        return Rotation().Quaternion(x=q_net[0], y=q_net[1], z=q_net[2], w=q_net[3])


def main():
    orc = IKOrchestrator()

    with NoPrint(stdout=True, stderr=True):
        success, tree = kdl_parser.treeFromFile("/home/suraj/ws/src/btp/iiwa_run/robot.urdf")
        # success, tree = kdl_parser.treeFromParam("/robot_description")
    if not success:
        print("Could not load the kdl_tree from the urdf.")
        return

    chain = tree.getChain('iiwa_link_0', 'tool_link_ee')

    fk_solver = ChainFkSolverPos_recursive(chain)

    # ChainIkSolverVel_pinv(chain: Chain, eps: float = 1e-05, maxiter: int = 150)
    ikv_solver = ChainIkSolverVel_pinv(chain)

    # ChainIkSolverPos_NR(chain: Chain, fksolver: ChainFkSolverPos, iksolver: ChainIkSolverVel, maxiter: int = 100, eps: float = epsilon):
    ik_solver = ChainIkSolverPos_NR(chain, fk_solver, ikv_solver)

    # JntArray(size: int) or JntArray(arg: JntArray)
    j_start = JntArray(orc.nj)
    for i, val in enumerate(orc.zero_state_joints):
        j_start[i] = val

    # http://docs.ros.org/en/hydro/api/orocos_kdl/html/classKDL_1_1ChainFkSolverPos__recursive.html
    # JntToCart(self, q_in: JntArray, p_out: Frame, segmentNr: int = -1):
    f = Frame()

    j_ik = JntArray(orc.nj)
    f = Frame()
    f.p = orc.insertion_pt
    f.p.z(f.p.z() - 0.1)
    f.p.x(f.p.x() + 0.005)
    f.M = orc.get_target_orientation(f.p)

    # CartToJnt(self, q_init: JntArray, p_in: Frame, q_out: JntArray)
    ret = ik_solver.CartToJnt(j_start, f, j_ik)
    if ret != 0:
        print("Ik failed")
        return

    joint_diff = 0
    for i, val in enumerate(orc.zero_state_joints):
        joint_diff += (j_ik[i] - val) ** 2

    print(f"{joint_diff=}")

    f_c = Frame()
    ret = (fk_solver.JntToCart(q_in=j_ik, p_out=f_c))
    if ret != 0:
        print("Fk failed")
        return

    position_error = (f_c.p.x() - f.p.x()) ** 2 + (f_c.p.y() - f.p.y()) ** 2 + (f_c.p.z() - f.p.z()) ** 2
    print(f"{position_error=}")

    dot_prod = 0
    for d, d_c in zip(f.M.GetQuaternion(), f_c.M.GetQuaternion()):
        dot_prod += d * d_c
    orientation_error = np.arccos(dot_prod)
    print(f"{orientation_error=}")


if __name__ == '__main__':
    main()
