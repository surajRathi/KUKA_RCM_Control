#! /usr/bin/python3
import time
from math import isfinite

import numpy as np
from PyKDL import JntArray
from tqdm import tqdm

from iiwa_run.sampling_ik_orchestrator import SamplingIKOrchestrator


class PathToJoint(SamplingIKOrchestrator):
    def __init__(self, resolution=2e-3):
        super(PathToJoint, self).__init__(resolution)

        self.r0 = np.array(self.spec.rcm)
        self.r0[2] -= (self.spec.H1 + self.spec.H)

    def get_line_path(self, x1, x2):
        # x1 and x2 are in base of the cylinder originated world frame
        x2 = np.array(x2)
        num_pts = int(np.ceil(np.linalg.norm(x2 - x1) / self.res))
        return np.linspace(x1, x2, num_pts)

    def run(self, path):
        t_start = time.time()
        assert all(path[0, :] == (0, 0, 0))
        tjd_s = np.zeros(path.shape[0]) * np.nan
        joints = np.zeros((path.shape[0], self.nj)) * np.nan

        tjd_s[0] = 0
        joints[0, :] = self.initial_joint_states

        path = self.r0 + path
        cur_joints = JntArray(self.nj)
        for i, val in enumerate(self.initial_joint_states):
            cur_joints[i] = val
        for i, pt in enumerate(tqdm(path[1:], total=path.shape[0] - 1), start=1):
            joint_diff, pos_error, orien_error, next_joints = self.get_solution(cur_joints, pt)
            if not isfinite(joint_diff):
                print(f"Could not find solution at {i=} of {path.shape[0]}")
                print(f"Loc: {path[i, :]}")
                break
            tjd_s[i] = np.degrees(np.sqrt(joint_diff))
            joints[i, :] = [next_joints[i] for i in range(self.nj)]

            cur_joints = next_joints
        else:
            t_end = time.time()
            print(f"Path Successfully found in {(t_end - t_start) * 1000:.0f}ms.")
            print(f"n_inner: {self.num_inner}")
            print(f"Mean TJDs: {tjd_s.mean():.4f}")

        np.save('joint_vals.npy', joints)


def main():
    pc = PathToJoint()
    path = pc.get_line_path((0, 0, 0), (0.3, 0.0, 0.0))
    pc.run(path)


if __name__ == '__main__':
    main()