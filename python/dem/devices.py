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
    def __init__(self, name=None,
                 T_init=None, T_min=None, T_max=None, T_ambient=None,
                 p_max=None,
                 conduct_coeff=None,
                 efficiency=None,
                 capacity=None):
        super(ThermalLoad, self).__init__([Terminal()], name)
        self.T_init = T_init,
        self.T_min = T_min
        self.T_max = T_max
        self.T_ambient = T_ambient
        self.p_max = p_max
        self.conduct_coeff = conduct_coeff
        self.efficiency = efficiency
        self.capacity = capacity

    @property
    def constraints(self):
        alpha = self.conduct_coeff / self.capacity
        beta = self.efficiency / self.capacity
        N = self.terminals[0].power.size[0]
        self.T = cvx.Variable(N)

        constrs = [
            self.terminals[0].power <= self.p_max,
            self.terminals[0].power >= 0,
        ]

        if self.T_max is not None:
            constrs += [self.T <= self.T_max]
        if self.T_min is not None:
            constrs += [self.T >= self.T_min]

        for i in range(N):
            Tprev = self.T[i-1] if i else self.T_init
            constrs += [
                self.T[i] == (Tprev + alpha*(self.T_ambient[i] - Tprev) -
                              beta*self.terminals[0].power[i])
            ]

        return constrs


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
    def __init__(self, name=None, t_start=0, t_end=None, E=0, p_max=None):
        super(DeferrableLoad, self).__init__([Terminal()], name)
        self.t_start = t_start
        self.t_end = t_end
        self.E = E
        self.p_max = p_max

    @property
    def constraints(self):
        idx = slice(self.t_start, self.t_end)
        return [
            cvx.sum_entries(self.terminals[0].power[idx]) >= self.E,
            self.terminals[0].power >= 0,
            self.terminals[0].power <= self.p_max,
        ]


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
    def __init__(self, name=None, p_min=0, p_max=None, E_init=0, E_max=None):
        super(Storage, self).__init__([Terminal()], name)
        self.p_min = p_min
        self.p_max = p_max
        self.E_init = E_init
        self.E_max = E_max

    @property
    def constraints(self):
        N = self.terminals[0].power.size[0]
        cumsum = np.tril(np.ones((N,N)), 0)
        self.energy = self.E_init + cumsum*self.terminals[0].power
        return [
            self.terminals[0].power >= self.p_min,
            self.terminals[0].power <= self.p_max,
            self.energy <= self.E_max,
            self.energy >= 0,
        ]
