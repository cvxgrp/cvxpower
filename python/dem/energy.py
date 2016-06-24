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

        p = self.terminals[0].power
        self.cost = alpha*cvx.square(p) - beta*p
        self.constraints = [
            -p <= p_max,
            -p >= p_min
        ]


class Load(Device):
    """Fixed load."""
    def __init__(self, name=None, p=None, curtailable_alpha=None):
        super(Load, self).__init__([Terminal()], name)
        self.constraints = [self.terminals[0].power == p]


class CurtailableLoad(Device):
    """Curtailable load."""
    def __init__(self, name=None, p=None, alpha=None):
        super(CurtailableLoad, self).__init__([Terminal()], name)
        self.cost = alpha*cvx.pos(p - self.terminals[0].power)



class TransmissionLine(Device):
    """Transmission line."""
    def __init__(self, name=None, p_max=None):
        super(TransmissionLine, self).__init__([Terminal(), Terminal()], name)

        self.constraints = [
            self.loss == 0,
            cvx.abs(self.power_flow) <= p_max
        ]

    @property
    def loss(self):
        """Loss in the line."""
        return self.terminals[0].power + self.terminals[1].power

    @property
    def power_flow(self):
        """Power flow between terminals."""
        return (self.terminals[0].power - self.terminals[1].power)/2
