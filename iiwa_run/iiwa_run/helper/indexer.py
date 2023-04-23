from math import sqrt, floor

from iiwa_run.helper.specifications import Specifications


class Limits:
    def __init__(self, mi, ma):
        self.min = mi
        self.max = ma

        self.delta = ma - mi

    def __call__(self, val):
        return self.min <= val <= self.max

    def __str__(self):
        return f"({self.min}->{self.max})"

    def get(self):
        return self.min, self.max


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
