#!/usr/bin/python3
import random
import sys
from math import sqrt, acos, sin, cos, floor, isinf
from pathlib import Path
from queue import Queue
from typing import Tuple

import kdl_parser_py.urdf as kdl_parser
import numpy as np
import rospkg
import tqdm
from PyKDL import ChainFkSolverPos_recursive, JntArray, Frame, ChainIkSolverPos_NR, ChainIkSolverVel_pinv, Vector, \
    Rotation
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
        s = spec.rl / sqrt(2)
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
            int(floor(self.xlim.delta / res)),
            int(floor(self.ylim.delta / res)),
            int(floor(self.zlim.delta / res)),
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
                int(floor(((x - self.x0) + self.xoff) / self.res)),
                int(floor(((y - self.y0) + self.yoff) / self.res)),
                int(floor(((z - self.z0) + self.zoff) / self.res)),
            )

        raise IndexError(f"{x}\t{y}\t{z}")


class IKOrchestrator:
    def __init__(self, resolution=2e-3):
        robot_desc = Path(rospkg.RosPack().get_path('iiwa_needle_moveit')) / 'config/gazebo_iiwa7_tool.urdf'
        spec_desc = Path(rospkg.RosPack().get_path('iiwa_needle_description')) / 'param/world.yaml'

        self.spec = from_yaml(spec_desc)
        self.initial_joint_states = self.spec.rest_joint_states
        self.nj = len(self.initial_joint_states)

        self.res = resolution
        self.indexer = Indexer(self.spec, self.res)

        random.seed(self.spec.seed)
        self.rng_state = random.getstate()
        self.num_inner = self.spec.n_inner
        self.max_n_inner = self.spec.max_n_inner
        self.num_sample_out = 0

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

        run_dir = Path(sys.argv[0]).parent.parent / "run"
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

    def get_target_orientation(self, target_point: Vector,
                               start_point: Tuple[float, float, float] = (0, 0, 0)) -> Rotation:
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

        nn = sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        nx, ny, nz = (dx / nn, dy / nn, dz / nn)

        nn = sqrt(ny ** 2 + nx ** 2)

        q_rot = quaternion_about_axis(angle=acos(-nz), axis=(-ny / nn, -nx / nn, 0))

        q_initial = self.insertion_rot.GetQuaternion()
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
        orientation_error = acos(dot_prod)
        if (j_ik.rows()) != self.nj:
            print("AAAAA", self.nj, j_ik, (j_ik.rows()), j_in, (j_in.rows()), f_c)
        return joint_diff, position_error, orientation_error, j_ik

    def generate_frames(self, ind):
        random.setstate(self.rng_state)
        this_seed = random.getrandbits(32)
        self.rng_state = random.getstate()
        random.seed(this_seed)

        center_r = self.spec.R - self.spec.r

        target_coords = self.indexer.index_to_coord(*ind)
        px, py, pz = target_coords

        def get_locs():
            yield 0, 0, 0
            while True:
                r, theta = random.random(), random.random()
                r *= center_r
                theta *= 2 * np.pi

                yield r * cos(theta), r * sin(theta), 0

        def is_valid(lx: float, ly: float, lz: float) -> bool:
            n_qo_s = (lx ** 2 + ly ** 2 + lz ** 2)
            n_qo = sqrt(n_qo_s)
            if n_qo <= 1e-9:
                return True

            ox, oy, oz = self.insertion_pt.x(), self.insertion_pt.y(), self.insertion_pt.z()
            qx, qy, qz = ox + lx, oy + ly, oz + lz

            n_pq_s = sqrt((qx - px) ** 2 + (qy - py) ** 2 + (qz - pz) ** 2)

            ux, uy, uz = px - qx, py - qy, pz - qz

            cross_s = (uy * lz - uz * ly) ** 2 + (uz * lx - ux * lz) ** 2 + (ux * ly - uy * lx) ** 2

            d = (self.spec.R - n_qo) * sqrt(cross_s / n_pq_s / n_qo_s)
            return d > self.spec.r

        # Method 1: Brute force

        i = 0
        j = 0
        for lx, ly, lz in get_locs():
            j += 1
            if is_valid(lx, ly, lz):
                i += 1
                f = Frame()
                f.p = Vector(*target_coords)
                f.M = self.get_target_orientation(f.p, start_point=(lx, ly, lz))
                yield f

            if i >= self.num_inner:
                return

            if j >= self.max_n_inner:
                self.num_sample_out += 1
                return

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
                if not isinf(joint_diff):
                    self.arr[ind] = [joint_diff, pos_error, orien_error] + [joints[i] for i in range(self.nj)]
                    self.add_next(ind)
                    bar.update(1)
                else:
                    self.arr[ind] = [joint_diff, pos_error, orien_error] + [np.inf for i in range(self.nj)]

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


def main():
    orc = IKOrchestrator()
    orc.run()
    print("Times solve failed:", np.isnan(orc.arr[:, :, :, 0]).sum())
    print("Times could not sample:", orc.num_sample_out)
    print("Total N:", orc.N)


if __name__ == '__main__':
    main()
