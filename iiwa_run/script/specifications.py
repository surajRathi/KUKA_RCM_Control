from dataclasses import dataclass, field
from math import sqrt, sin, cos, acos
from typing import Tuple


@dataclass
class Specifications:
    id: str
    n_inner: int
    seed: int

    rcm: Tuple[float, float, float]
    rest_joint_states: Tuple[float, ...]
    R: float
    r: float
    L: float

    l1: float
    l2: float

    theta: float = field(init=False)
    H: float = field(init=False)
    H1: float = field(init=False)
    H2: float = field(init=False)
    rl: float = field(init=False)

    def __post_init__(self):
        self.theta = acos(self.r / self.R)

        self.H1 = self.l1 * cos(self.theta)
        self.rl = self.l1 * sin(self.theta)

        self.H2 = sqrt(self.l2 ** 2 - self.rl ** 2)
        self.H = self.H2 - self.H1


# This assumes that a ROS node has been inited.
def from_param(root="/spec"):
    import rospy

    def get(name):
        return rospy.get_param(f"{root}/{name}")

    def err(k):
        rospy.logfatal(f"Specifications requires the {k.args[0]} param to be set.")

    return _load_impl(get, err)


def from_yaml(filename="./world.yaml"):
    import yaml
    yml = yaml.safe_load(open(filename, 'r').read())

    def get(name):
        el = yml
        for n in name.split('/'):
            el = el[n]

        return el

    def err(k):
        print(f"Specifications requires the {k.args[0]} key to be set.")

    return _load_impl(get, err)


def _load_impl(get, err):
    try:
        return Specifications(
            id=str(get("id")),
            n_inner=int(get("n_inner")),
            seed=int(get("seed")),
            rest_joint_states=tuple(get("rest_joint_states")),
            rcm=(get("rcm/x") / 1e3, get("rcm/y") / 1e3, get("rcm/z") / 1e3),
            R=get("rcm/r") / 1e3,
            r=get("tool/r") / 1e3,
            L=get("tool/L") / 1e3,
            l1=get("tool/l1") / 1e3,
            l2=get("tool/l2") / 1e3,
        )
    except KeyError as k:
        err(k)
        return
