"""Device models for dynamic energy management.

See Section 3 of DEM paper for details.
"""

import cvxpy as cvx

from dem.network import Device, Terminal

class Generator(Device):
    """Generator with quadratic cost."""
    def __init__(
            self,
            name=None,
            p_min=0,
            p_max=0,
            alpha=0,
            beta=0):
        super(Generator, self).__init__([Terminal()], name)
        self.p_min = p_min
        self.p_max = p_max
        self.alpha = alpha
        self.beta = beta

    @property
    def cost(self):
        p = self.terminals[0].power
        return self.alpha*cvx.square(p) - self.beta*p

    @property
    def constraints(self):
        return [
            -self.terminals[0].power <= self.p_max,
            -self.terminals[0].power >= self.p_min
        ]


class Load(Device):
    """Fixed load."""
    def __init__(self, name=None, p=None):
        super(Load, self).__init__([Terminal()], name)
        self.p = p

    @property
    def constraints(self):
        return [self.terminals[0].power == self.p]


class CurtailableLoad(Device):
    """Curtailable load."""
    def __init__(self, name=None, p=None, alpha=None):
        super(CurtailableLoad, self).__init__([Terminal()], name)
        self.p = p
        self.alpha = alpha

    @property
    def cost(self):
        return self.alpha*cvx.pos(self.p - self.terminals[0].power)


class TransmissionLine(Device):
    """Transmission line."""
    def __init__(self, name=None, p_max=None):
        super(TransmissionLine, self).__init__([Terminal(), Terminal()], name)
        self.p_max = p_max

    @property
    def constraints(self):
        return [
            self.terminals[0].power + self.terminals[1].power == 0,
            (cvx.abs((self.terminals[0].power - self.terminals[1].power)/2)
             <= self.p_max)
        ]
