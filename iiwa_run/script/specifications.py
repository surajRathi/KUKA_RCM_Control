from dataclasses import dataclass, field
from math import atan, sqrt, asin, pi, sin, cos


@dataclass
class Specifications:
    R: float
    t: float
    r: float
    L: float

    l1: float
    l2: float

    theta: float = field(init=False)
    Hb: float = field(init=False)
    H: float = field(init=False)
    rl1: float = field(init=False)

    def __post_init__(self):
        a_1 = atan(self.t / 2 / self.R)
        a_2 = asin(2 * self.r / sqrt(4 * (self.R ** 2) + self.t ** 2))

        self.alpha = a_1 + a_2
        self.theta = pi / 2 - self.alpha
        self.lb = (self.R - self.r / sin(self.alpha)) / sin(self.alpha)

        self.Hb = self.lb * cos(self.theta)
        self.H1 = (self.lb + self.l1) * cos(self.theta)
        self.H = (self.l2 - self.l1) * cos(self.theta)

        self.rl1 = (self.lb + self.l1) * sin(self.theta)
        # self.rl2 = (self.lb + self.l1 + self.l1) * sin(self.theta)

        # # Taking a cylinder as the workspace
        # self.V = pi * self.rl1 ** 2 * self.H


import rospy


# This assumes that a ROS node has been inited.
def create_specification(root="/spec"):
    def get(name):
        return rospy.get_param(f"{root}/{name}")

    try:
        return Specifications(
            R=get("rcm/r") / 1e3,
            t=get("rcm/t") / 1e3,
            r=get("tool/r") / 1e3,
            L=get("tool/L") / 1e3,
            l1=get("tool/l1") / 1e3,
            l2=get("tool/l2") / 1e3,
        )
    except KeyError as k:
        rospy.logfatal(f"Specifications requires the {k.args[0]} param to be set.")
        return
