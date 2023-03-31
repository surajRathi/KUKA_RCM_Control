#!/usr/bin/python3
import sys
from typing import Tuple

import kdl_parser_py.urdf as kdl_parser
import numpy as np
from PyKDL import ChainFkSolverPos_recursive, JntArray, Frame, ChainIkSolverPos_NR, ChainIkSolverVel_pinv, Vector, \
    Rotation
from tf.transformations import quaternion_about_axis, quaternion_multiply

from no_print import NoPrint


class IKOrchestrator:
    def __init__(self):
        self.pos_delta = 2e-3

        filename = f"{sys.argv[0][:sys.argv[0].find('/')]}/robot.urdf"
        # Set up KDL
        with NoPrint(stdout=True, stderr=True):
            success, tree = kdl_parser.treeFromFile(filename)
            # success, tree = kdl_parser.treeFromFile("/home/suraj/ws/src/btp/iiwa_run/robot.urdf")
            # success, tree = kdl_parser.treeFromParam("/robot_description")
        if not success:
            print("Could not load the kdl_tree from the urdf.")
            return

        self.chain = tree.getChain('iiwa_link_0', 'tool_link_ee')

        self.fk_solver = ChainFkSolverPos_recursive(self.chain)

        # ChainIkSolverVel_pinv(chain: Chain, eps: float = 1e-05, maxiter: int = 150)
        self.ikv_solver = ChainIkSolverVel_pinv(self.chain)

        # ChainIkSolverPos_NR(chain: Chain, fksolver: ChainFkSolverPos, iksolver: ChainIkSolverVel, maxiter: int = 100, eps: float = epsilon):
        self.ik_solver = ChainIkSolverPos_NR(self.chain, self.fk_solver, self.ikv_solver)

        # Set up reference pose
        self.insertion_pt = Vector()
        self.insertion_pt.x(0.0)
        self.insertion_pt.y(-0.5)
        self.insertion_pt.z(0.3)

        self.insertion_rot = Rotation().Quaternion(x=0.000, y=1.000, z=0.000, w=0.000)

        self.zero_state_joints = [0.7974687140058682, -0.7059608762250215, -2.0557765983620215, -1.7324525588466884,
                                  -0.685107459772449,
                                  1.136261558580683, -1.0476528028679626]

        self.nj = len(self.zero_state_joints)

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

    def do_ik(self, j_in: JntArray, f_target: Frame) -> Tuple[float, float, float, JntArray]:
        f_t = f_target

        # Check that initial joints are correct
        f_init = Frame()
        ret = (self.fk_solver.JntToCart(q_in=j_in, p_out=f_init))
        if ret != 0:
            return np.inf, np.inf, np.inf, JntArray(0)

        # Sum of squares of position error
        position_error = \
            (f_init.p.x() - f_t.p.x()) ** 2 + (f_init.p.y() - f_t.p.y()) ** 2 + (f_init.p.z() - f_t.p.z()) ** 2
        if position_error > 2 * self.pos_delta:
            return np.inf, np.inf, np.inf, JntArray(0)

        # Do the inverse kinematics
        j_ik = JntArray(self.nj)
        ret = self.ik_solver.CartToJnt(j_in, f_t, j_ik)
        if ret != 0:
            return np.inf, np.inf, np.inf, JntArray(0)

        # Sum of square of delta joint angle
        joint_diff = 0
        for i in range(self.nj):
            joint_diff += (j_ik[i] - j_in[i]) ** 2

            # Fk on the joings
        f_c = Frame()
        ret = (self.fk_solver.JntToCart(q_in=j_ik, p_out=f_c))
        if ret != 0:
            return np.inf, np.inf, np.inf, JntArray(0)

        # Sum of squares of position error
        position_error = (f_c.p.x() - f_t.p.x()) ** 2 + (f_c.p.y() - f_t.p.y()) ** 2 + (f_c.p.z() - f_t.p.z()) ** 2

        # Angle between the orientations
        dot_prod = 0
        for d, d_c in zip(f_t.M.GetQuaternion(), f_c.M.GetQuaternion()):
            dot_prod += d * d_c
        orientation_error = np.arccos(dot_prod)

        return joint_diff, position_error, orientation_error, j_ik


def main():
    orc = IKOrchestrator()

    # JntArray(size: int) or JntArray(arg: JntArray)
    j_start = JntArray(orc.nj)
    for i, val in enumerate(orc.zero_state_joints):
        j_start[i] = val

    f = Frame()
    f.p = orc.insertion_pt
    f.p.z(f.p.z() - 0.1)
    f.p.x(f.p.x() + 0.005)
    f.M = orc.get_target_orientation(f.p)

    joint_diff, pos_error, orien_error, joints = orc.do_ik(j_start, f)

    print(f"{joint_diff=:.2E}\t{pos_error=:.2E}\t{orien_error=:.2E}")


if __name__ == '__main__':
    main()
