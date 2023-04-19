#!/usr/bin/python3
import sys
from math import sqrt, floor, isinf
from pathlib import Path
from queue import Queue
from typing import Tuple

import numpy as np
import tqdm
from PyKDL import JntArray

from iiwa_run.helper.specifications import Specifications
from iiwa_run.sampling_ik_orchestrator import SamplingIKOrchestrator


class Limits:
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

        self.xlim = Limits(self.x0 - s / 2, self.x0 + s / 2)
        self.ylim = Limits(self.y0 - s / 2, self.y0 + s / 2)
        self.zlim = Limits(self.z0, self.z0 + spec.H)

        self.shape = (
            int(floor(self.xlim.delta / res)),
            int(floor(self.ylim.delta / res)),
            int(floor(self.zlim.delta / res)),
        )

        self.xilim = Limits(0, self.shape[0] - 1)
        self.yilim = Limits(0, self.shape[1] - 1)
        self.zilim = Limits(0, self.shape[2] - 1)

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


class FloodFillCheck(SamplingIKOrchestrator):
    def __init__(self, resolution=2e-3):
        super(FloodFillCheck, self).__init__(resolution)

        self.indexer = Indexer(self.spec, self.res)

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

                    joint_diff, pos_error, orien_error, joints = self.get_solution(j_start,
                                                                                   self.indexer.index_to_coord(*ind))
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
    orc = FloodFillCheck()
    orc.run()
    print("Times solve failed:", np.isnan(orc.arr[:, :, :, 0]).sum())
    print("Times could not sample:", orc.num_sample_out)
    print("Total N:", orc.N)


if __name__ == '__main__':
    main()
