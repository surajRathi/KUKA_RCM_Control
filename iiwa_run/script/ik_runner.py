#!/usr/bin/python3
import sys
from pathlib import Path
from queue import Queue
from typing import Tuple, Optional

import kdl_parser_py.urdf as kdl_parser
import numpy as np
import rospkg
import tqdm
from PyKDL import ChainFkSolverPos_recursive, JntArray, Frame, ChainIkSolverPos_NR, ChainIkSolverVel_pinv, Vector, \
    Rotation
from numpy.random import Generator
from tf.transformations import quaternion_about_axis, quaternion_multiply

from no_print import NoPrint
from specifications import from_yaml, Specifications


class Lim:
    def __init__(self, mi, ma):
        self.min = mi
        self.max = ma

        self.delta = ma - mi

    def __call__(self, val):
        return self.min <= val <= self.max

    def __str__(self):
        return f"({self.min}->{self.max})"


class Indexer:
    """ Maps the real world xyz coordinates to the indexes for the backing array. """

    def __init__(self, spec: Specifications, res: float):
        s = spec.rl / np.sqrt(2)
        self.res = res

        self.x0, self.y0, self.z0 = spec.rcm
        self.z0 -= (spec.H1 + spec.H)

        self.xoff = s / 2
        self.yoff = s / 2
        self.zoff = 0

        self.xlim = Lim(self.x0 - s / 2, self.x0 + s / 2)
        self.ylim = Lim(self.y0 - s / 2, self.y0 + s / 2)
        self.zlim = Lim(self.z0, self.z0 + spec.H)

        self.shape = (
            int(np.floor(self.xlim.delta / res)),
            int(np.floor(self.ylim.delta / res)),
            int(np.floor(self.zlim.delta / res)),
        )

        self.xilim = Lim(0, self.shape[0] - 1)
        self.yilim = Lim(0, self.shape[1] - 1)
        self.zilim = Lim(0, self.shape[2] - 1)

    def is_index_valid(self, xi, yi, zi):
        return self.xilim(xi) and self.yilim(yi) and self.zilim(zi)

    def index_to_coord(self, xi, yi, zi):
        if self.xilim(xi) and self.yilim(yi) and self.zilim(zi):
            return (
                xi * self.res + self.x0 - self.xoff,
                yi * self.res + self.y0 - self.yoff,
                zi * self.res + self.z0 - self.zoff,
            )
        raise IndexError(f"{xi}\t{yi}\t{zi}")

    def coord_to_index(self, x, y, z):
        if self.xlim(x) and self.ylim(y) and self.zlim(z):
            return (
                int(np.floor(((x - self.x0) + self.xoff) / self.res)),
                int(np.floor(((y - self.y0) + self.yoff) / self.res)),
                int(np.floor(((z - self.z0) + self.zoff) / self.res)),
            )

        raise IndexError(f"{x}\t{y}\t{z}")


class IKOrchestrator:
    def __init__(self, resolution=2e-3):
        robot_desc = Path(rospkg.RosPack().get_path('iiwa_needle_moveit')) / 'config/gazebo_iiwa7_tool.urdf'
        spec_desc = Path(rospkg.RosPack().get_path('iiwa_needle_description')) / 'param/world.yaml'
        run_dir = Path(sys.argv[0]).parent.parent / "run"

        self.spec = from_yaml(spec_desc)
        self.initial_joint_states = self.spec.rest_joint_states
        self.nj = len(self.initial_joint_states)

        self.res = resolution
        self.indexer = Indexer(self.spec, self.res)

        self.rng: Generator = np.random.default_rng(seed=self.spec.seed)
        self.num_inner = self.spec.n_inner

        # Set up KDL
        with NoPrint(stdout=True, stderr=True):
            success, tree = kdl_parser.treeFromFile(robot_desc)
        if not success:
            print("Could not load the kdl_tree from the urdf.")
            raise RuntimeError()

        self.chain = tree.getChain('iiwa_link_0', 'tool_link_ee')

        self.fk_solver = ChainFkSolverPos_recursive(self.chain)

        # ChainIkSolverVel_pinv(chain: Chain, eps: float = 1e-05, maxiter: int = 150)
        self.ikv_solver = ChainIkSolverVel_pinv(self.chain)

        # ChainIkSolverPos_NR(chain: Chain, fksolver: ChainFkSolverPos, iksolver: ChainIkSolverVel, maxiter: int = 100, eps: float = epsilon):
        self.ik_solver = ChainIkSolverPos_NR(self.chain, self.fk_solver, self.ikv_solver)

        # Set up reference pose
        self.insertion_pt = Vector(*self.spec.rcm)
        # TODO: Dont Hardcode! However the rest of the code relies on this orientation being valid.
        self.insertion_rot = Rotation().Quaternion(x=0.000, y=1.000, z=0.000, w=0.000)

        array_shape = tuple(list(self.indexer.shape) + [3 + self.nj])
        # frontier_array_file = run_dir / "front.csv"
        data_array_file = run_dir / f"{self.spec.id}.npy"

        if not Path(data_array_file).exists():
            # arr = np.zeros(array_shape, dtype=np.float32) + np.nan
            arr = np.full(array_shape, np.nan, dtype=np.float32)
            data = [0.0, 0.0, 0.0] + list(self.initial_joint_states)
            arr[self.indexer.coord_to_index(self.indexer.x0, self.indexer.y0, self.indexer.z0)] = data
            np.save(str(data_array_file), arr)
        else:
            print(f"Resuming is not supported! File {data_array_file} already exists.")
            raise RuntimeError()

        # self.arr: np.memmap = np.memmap(str(data_array_file), mode='w+', shape=tuple(array_shape), dtype=np.float32)
        # self.arr: np.memmap = np.memmap(data_array_file, mode='w+', shape=array_shape, dtype=np.float32, offset=0)
        self.arr: np.memmap = np.load(str(data_array_file), mmap_mode='r+')

        self.N = array_shape[0] * array_shape[1] * array_shape[2]
        assert (self.arr[:, :, :, 0].ravel().shape[0] == self.N), f"{self.arr.shape}\t{array_shape}\t{self.N}"
        assert (array_shape == self.arr.shape)

        self.done = np.count_nonzero(~np.isnan(self.arr[:, :, :, 0]))
        # TODO: Floodfill, get next algo, and run the floodfill.

        self.frontier = Queue()
        self.add_next(self.indexer.coord_to_index(self.indexer.x0, self.indexer.y0, self.indexer.z0))

    def run(self):
        with tqdm.tqdm(total=self.N, leave=True) as bar:
            bar.update(self.done)
            while not self.frontier.empty():
                ind, ind_p = self.frontier.get()

                # get parent joints
                pj = self.arr[ind_p][3:]

                j_start = JntArray(self.nj)
                for i, val in enumerate(pj):
                    j_start[i] = val

                    joint_diff, pos_error, orien_error, joints = min(
                        (self.do_ik(j_start, frame)
                         for frame in self.generate_frames(ind)),
                        # for frame in tqdm.tqdm(self.generate_frames(ind), total=self.num_inner, leave=False)),
                        key=lambda k: k[0]
                    )
                # save data
                self.arr[ind] = [joint_diff, pos_error, orien_error] + [joints[i] for i in range(self.nj)]

                # print(f"Initial:\t{orc.indexer.x0:.2f}\t{orc.indexer.y0:.2f}\t{orc.indexer.z0:.2f}")
                # print(f"Final  :\t{f.p.x():.2f}\t{f.p.y():.2f}\t{f.p.z():.2f}")
                # print(f"{joint_diff=:.2E}\t{pos_error=:.2E}\t{orien_error=:.2E}")

                self.add_next(ind)
                bar.update(1)
                bar.set_description(f"F: {self.frontier.qsize()}")

    def get_target_orientation(self, target_point: Vector,
                               start_point: Optional[np.ndarray] = np.array([0, 0, 0])) -> Rotation:
        dx = target_point.x() - self.insertion_pt.x() - start_point[0]
        dy = target_point.y() - self.insertion_pt.y() - start_point[1]
        dz = target_point.z() - self.insertion_pt.z() - start_point[2]

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
        if position_error > 2 * self.res:
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

            # Fk on the joints
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
        if dot_prod > 1:
            dot_prod = 1.0
        orientation_error = np.arccos(dot_prod)

        return joint_diff, position_error, orientation_error, j_ik

    def add_next(self, ind: Tuple[int, int, int]):
        index = list(ind)
        for ax in (0, 1, 2):
            for val in (-1, 1):
                index[ax] += val
                if self.indexer.is_index_valid(*index):
                    ind_t = tuple(index)
                    if np.isnan(self.arr[ind_t][0]):
                        self.frontier.put((ind_t, ind))
                        self.arr[ind_t][0] = -1
                index[ax] -= val

    def generate_frames(self, ind):
        rng: Generator = np.random.default_rng(self.rng.integers(0, 2 ** 31 - 1, endpoint=True, dtype=np.int32))
        center_r = self.spec.R - self.spec.r

        target_coords = self.indexer.index_to_coord(*ind)

        def get_locs():
            yield np.array((0, 0, 0))
            while True:
                r, theta = rng.random(2)
                r *= center_r
                theta *= 2 * np.pi

                yield np.array((r * np.cos(theta), r * np.sin(theta), 0))

        N = np.linalg.norm

        def dist_impl(O, Q, P):
            return (self.spec.R - N(Q - O)) * N(np.cross(P - Q, O - Q)) / N(P - Q) / N(O - Q)

        def is_valid(loc: np.ndarray) -> bool:
            O = np.array((self.insertion_pt.x(), self.insertion_pt.y(), self.insertion_pt.z()))
            Q = O + loc
            P = np.array(target_coords)

            if N(Q - O) <= 1e-9:
                return True

            return dist_impl(O, Q, P) > self.spec.r

        # Method 1: Brute force

        i = 0
        for loc in get_locs():
            if is_valid(loc):
                i += 1
                f = Frame()
                f.p = Vector(*target_coords)
                f.M = self.get_target_orientation(f.p, start_point=loc)
                yield f

            if i >= self.num_inner:
                return


def main():
    orc = IKOrchestrator()
    orc.run()


if __name__ == '__main__':
    main()
