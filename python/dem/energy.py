"""Device models for dynamic energy management."""

import cvxpy as cvx
import numpy as np

from dem.network import Device, Terminal


class Generator(Device):
    """Generator with quadratic cost, ramp constraints."""
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
        return self.alpha*cvx.sum_squares(p) - self.beta*cvx.sum_entries(p)

    @property
    def constraints(self):
        # TODO(mwytock): Add ramp constraints

        return [
            -self.terminals[0].power <= self.p_max,
            -self.terminals[0].power >= self.p_min
        ]


class FixedLoad(Device):
    """Fixed load."""
    def __init__(self, name=None, p=None):
        super(FixedLoad, self).__init__([Terminal()], name)
        self.p = p

    @property
    def constraints(self):
        return [self.terminals[0].power == self.p]


class ThermalLoad(Device):
    """Thermal load."""
    pass


class CurtailableLoad(Device):
    """Curtailable load."""
    def __init__(self, name=None, p=None, alpha=None):
        super(CurtailableLoad, self).__init__([Terminal()], name)
        self.p = p
        self.alpha = alpha

    @property
    def cost(self):
        return self.alpha*cvx.pos(self.p - self.terminals[0].power)


class DeferrableLoad(Device):
    """Deferrable load."""
    pass


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


class Storage(Device):
    """Storage device."""
    pass
