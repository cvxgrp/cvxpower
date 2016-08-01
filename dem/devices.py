r"""Device models for dynamic energy management.

Each device has one or more terminals each of which has associated with it a
vector representing power consumption or generation, :math:`p(\tau),
\tau = 1, \ldots, T`. The device models described below specify the objective
function and constraints imposed by each device over these power flows.

"""

import cvxpy as cvx
import numpy as np

from dem.network import Device, Terminal


class Generator(Device):
    r"""A generic generator model with quadratic cost and ramp constraints.

    The generator has range and ramp rate constraints

    .. math::

      P^{\min} \le -p \le P^{\max} \\
      R^{\min} \le -D p \le R^{\max}

    ..

    where :math:`D` is the forward difference operator. The cost function is
    separable across time periods

    .. math::

      \sum_{\tau=1}^T \phi(-p(\tau))

    ..

    and quadratic, :math:`\phi(x) = \alpha x^2 + \beta x`, parameterized by
    :math:`\alpha, \beta \in \mathbb{R}`.

    """
    def __init__(
            self,
            name=None,
            p_min=0,
            p_max=0,
            r_min=None,
            r_max=None,
            alpha=0,
            beta=0):
        super(Generator, self).__init__([Terminal()], name)
        self.p_min = p_min
        self.p_max = p_max
        self.alpha = alpha
        self.beta = beta

    @property
    def cost(self):
        p = self.terminals[0].power_var
        return self.alpha*cvx.sum_squares(p) - self.beta*cvx.sum_entries(p)

    @property
    def constraints(self):
        # TODO(mwytock): Add ramp constraints

        return [
            -self.terminals[0].power_var <= self.p_max,
            -self.terminals[0].power_var >= self.p_min
        ]


class FixedLoad(Device):
    """Fixed load."""
    def __init__(self, name=None, p=None):
        super(FixedLoad, self).__init__([Terminal()], name)
        self.p = p

    @property
    def constraints(self):
        return [self.terminals[0].power_var == self.p]


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
        N = self.terminals[0].power_var.size[0]
        self.T = cvx.Variable(N)

        constrs = [
            self.terminals[0].power_var <= self.p_max,
            self.terminals[0].power_var >= 0,
        ]

        if self.T_max is not None:
            constrs += [self.T <= self.T_max]
        if self.T_min is not None:
            constrs += [self.T >= self.T_min]

        for i in range(N):
            Tprev = self.T[i-1] if i else self.T_init
            constrs += [
                self.T[i] == (Tprev + alpha*(self.T_ambient[i] - Tprev) -
                              beta*self.terminals[0].power_var[i])
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
        return self.alpha*cvx.pos(self.p - self.terminals[0].power_var)


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
            cvx.sum_entries(self.terminals[0].power_var[idx]) >= self.E,
            self.terminals[0].power_var >= 0,
            self.terminals[0].power_var <= self.p_max,
        ]


class TransmissionLine(Device):
    """Transmission line."""
    def __init__(self, name=None, p_max=None):
        super(TransmissionLine, self).__init__([Terminal(), Terminal()], name)
        self.p_max = p_max

    @property
    def constraints(self):
        return [
            self.terminals[0].power_var + self.terminals[1].power_var == 0,
            (cvx.abs((self.terminals[0].power_var - self.terminals[1].power_var)/2)
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
        N = self.terminals[0].power_var.size[0]
        cumsum = np.tril(np.ones((N,N)), 0)
        self.energy = self.E_init + cumsum*self.terminals[0].power_var
        return [
            self.terminals[0].power_var >= self.p_min,
            self.terminals[0].power_var <= self.p_max,
            self.energy <= self.E_max,
            self.energy >= 0,
        ]
