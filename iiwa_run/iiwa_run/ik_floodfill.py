#!/usr/bin/python3
import sys
from math import isinf
from pathlib import Path
from queue import Queue
from typing import Tuple

import numpy as np
import rospkg
import tqdm
from PyKDL import JntArray

from iiwa_run.helper.indexer import Indexer
from iiwa_run.sampling_ik_orchestrator import SamplingIKOrchestrator


class FloodFillCheck(SamplingIKOrchestrator):
    def __init__(self, resolution=2e-3, spec_name=None):
        if spec_name is not None:
            spec_desc = Path(rospkg.RosPack().get_path('iiwa_run')) / f"run/{spec_name}.yaml"
        else:
            spec_desc = None
        super(FloodFillCheck, self).__init__(resolution, spec_desc=spec_desc)

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
        print(f"Running id: {self.spec.id}")
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


class DirectedFloodFill(FloodFillCheck):
    def add_next(self, ind: Tuple[int, int, int]):
        center = self.indexer.coord_to_index(self.indexer.x0, self.indexer.y0, self.indexer.z0)
        index = list(ind)

        for ax in (0, 1, 2):
            ddl = index[ax] - center[ax]
            vals = [-1, 1]
            if ddl < 0:
                vals.remove(1)
            elif ddl > 0:
                vals.remove(-1)
            for val in vals:
                index[ax] += val
                if self.indexer.is_index_valid(*index):
                    ind_t = tuple(index)
                    if np.isnan(self.arr[ind_t][0]):
                        self.frontier.put((ind_t, ind))
                        self.arr[ind_t][0] = -1
                index[ax] -= val


def main():
    cls = DirectedFloodFill if len(sys.argv) >= 3 and sys.argv[2] == 'directed' else FloodFillCheck
    print(f"Using the {cls.__name__}")
    orc: FloodFillCheck = cls(spec_name=sys.argv[1] if len(sys.argv) >= 2 else None)
    orc.run()
    print("Times solve failed:", np.isnan(orc.arr[:, :, :, 0]).sum())
    print("Times could not sample:", orc.num_sample_out)
    print("Total N:", orc.N)


if __name__ == '__main__':
    main()
