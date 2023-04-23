#! /usr/bin/python3
import time
from math import isfinite
from typing import Tuple

import numpy as np
from PyKDL import JntArray
from tqdm import tqdm

from iiwa_run.sampling_ik_orchestrator import SamplingIKOrchestrator


class PathToJoints(SamplingIKOrchestrator):
    def __init__(self, resolution=2e-3):
        super(PathToJoints, self).__init__(resolution)

        self.r0 = np.array(self.spec.rcm)
        self.r0[2] -= (self.spec.H1 + self.spec.H)

    def get_line_path(self, x1, x2):
        # x1 and x2 are in base of the cylinder originated world frame
        x2 = np.array(x2)
        num_pts = int(np.ceil(np.linalg.norm(x2 - x1) / self.res))
        return np.linspace(x1, x2, num_pts)

    def get_circle_path(self, center, start, angle=2 * np.pi):
        # all points are in base of the cylinder originated world frame
        center = np.array(center)
        start = np.array(start)

        r = np.linalg.norm(start - center)
        d_theta = self.res / r

        start_angle = np.arctan2(start[2] - center[2], start[1] - center[1])
        angles = np.arange(start_angle, start_angle + angle, d_theta)

        ret = np.zeros((angles.shape[0], 3))
        ret[:, 0] = r * np.cos(angles)
        ret[:, 1] = r * np.sin(angles)

        return center + ret

    def run(self, path):
        assert all(path[0, :] == (0, 0, 0))
        joints = np.zeros((path.shape[0], self.nj)) * np.nan

        joints[0, :] = self.initial_joint_states

        path = self.r0 + path
        cur_joints = JntArray(self.nj)
        for i, val in enumerate(self.initial_joint_states):
            cur_joints[i] = val
        for i, pt in enumerate(tqdm(path[1:], total=path.shape[0] - 1), start=1):
            joint_diff, pos_error, orien_error, next_joints = self.get_solution(cur_joints, pt)
            if not isfinite(joint_diff):
                break
            joints[i, :] = [next_joints[i] for i in range(self.nj)]

            cur_joints = next_joints

        return joints


def main():
    pc = PathToJoints()
    # path = pc.get_line_path((0.0, 0, 0), (0.25, 0.0, 0.0))
    path = np.vstack((
        pc.get_line_path((0, 0, 0), (0.25, 0.0, 0.0)),
        pc.get_circle_path((0, 0, 0), (0.25, 0.0, 0.0)),
        pc.get_line_path((0.25, 0, 0), (0.0, 0.0, 0.0))))

    t_start = time.time()
    joints = pc.run(path)
    t_end = time.time()

    def index(array, item) -> Tuple[int, ...]:
        for idx, val in np.ndenumerate(array):
            if val == item:
                return idx
        return tuple([-1] * len(array.shape))

    if (i := index(joints, np.nan)) != -1:
        print(f"Path Successfully found in {(t_end - t_start) * 1000:.0f}ms.")
        print(f"n_inner: {pc.num_inner}")

        initial_joints = np.array(pc.initial_joint_states).reshape(1, -1)
        tjd_s = np.degrees(np.sqrt(
            (np.diff(joints, axis=0, prepend=initial_joints) ** 2).sum(axis=-1)
        ))
        print(f"Mean TJDs: {tjd_s.mean():.4f}")

        np.save('joint_vals.npy', joints)
    else:
        print(f"Could not find solution at {i=} of {path.shape[0]}")
        print(f"Loc: {path[i, :]}")


if __name__ == '__main__':
    main()
