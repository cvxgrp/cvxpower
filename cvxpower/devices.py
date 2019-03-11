"""
Copyright (C) Matt Wytock, Enzo Busseti, Nicholas Moehle, 2016-2019.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Code written by Nicholas Moehle before January 2019 is licensed
under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


__doc__ = r"""Device models for dynamic energy management.

Each device has one or more terminals each of which has associated with it a
vector representing power consumption or generation. Formally, we denote this
power schedule as :math:`p = (p(1), \ldots, p(T)) \in \mathbb{R}^T` and devices
specify the objective function and constraints imposed by each device over these
power flows.
"""

import cvxpy as cvx
import numpy as np

from .network import Device, Terminal


class Generator(Device):
    r"""Generator with quadratic cost and optional ramp constraints.

    A generator has range and ramp rate constraints defining the operating
    region, for :math:`\tau = 1,\ldots,T`

    .. math::

      P^{\min} \le -p(\tau) \le P^{\max} \\
      R^{\min} \le -(p(\tau) - p(\tau - 1)) \le R^{\max}

    ..

    where :math:`p(0)` is defined to be :math:`p^\mathrm{init}` is the initial
    power produced by the generator. If :math:`R^{\min}` or :math:`R^{\max}` are
    unspecified, the minimum or maximum ramp rate is unconstrained, respectively.

    The cost function is separable across time periods

    .. math::

      \sum_{\tau=1}^T \phi(-p(\tau))

    ..

    and quadratic, :math:`\phi(x) = \alpha x^2 + \beta x`, parameterized by
    :math:`\alpha, \beta \in \mathbb{R}`.

    :param power_min: Minimum power generation, :math:`P^\min`
    :param power_max: Maximum power generation, :math:`P^\max`
    :param alpha: Quadratic term in cost function, :math:`\alpha`
    :param beta: Linear term in cost function, :math:`\beta`
    :param ramp_min: (optional) Minimum ramp rate, :math:`R^\min`
    :param ramp_max: (optional) Maximum ramp rate, :math:`R^\max`
    :param power_init: (optional) Initial power generation for ramp constraints, :math:`p^\mathrm{init}`
    :param name: (optional) Display name for generator
    :type power_min: float or sequence of float
    :type power_max: float or sequence of float
    :type alpha: float or sequence of float
    :type beta: float or sequence of float
    :type ramp_min: float or sequence of float
    :type ramp_max: float or sequence of float
    :type power_init: float
    :type name: string
    """

    def __init__(
            self,
            power_min=None,
            power_max=None,
            ramp_min=None,
            ramp_max=None,
            power_init=0,
            alpha=0,
            beta=0,
            gamma=0,
            name=None,
            len_interval=1):
        super(Generator, self).__init__([Terminal()], name)
        self.power_min = power_min
        self.power_max = power_max
        self.ramp_min = ramp_min
        self.ramp_max = ramp_max
        self.power_init = power_init
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.len_interval = len_interval  # in hours

    @property
    def cost(self):
        p = self.terminals[0].power_var
        return self.alpha * cvx.square(p) - self.beta * p + self.gamma

    @property
    def constraints(self):
        p = -self.terminals[0].power_var

        constraints = []
        if self.power_min is not None:
            constraints += [p >= self.power_min]
        if self.power_max is not None:
            constraints += [p <= self.power_max]

        # TODO(mwytock): Add ramp constraints

        return constraints


class FixedLoad(Device):
    """Fixed load.

    A fixed load has a specified profile :math:`l \in \mathbb{R}^T`,
    enforced with constraint

    .. math::
      p = l.
    ..

    :param power: Load profile, :math:`l`
    :param name: (optional) Display name for load
    :type power: float or sequence of float
    :type name: string
    """

    def __init__(self, power=None, name=None):
        super(FixedLoad, self).__init__([Terminal()], name)
        self.power = power

    @property
    def constraints(self):
        if self.terminals[0].power_var.shape[1] == 1:
            p = self.power[:, 0] if (not np.isscalar(self.power) and
                                     self.power.ndim == 2) else self.power
            return [self.terminals[0].power_var[:, 0] == p]
        else:
            return [self.terminals[0].power_var == self.power]


class ThermalLoad(Device):
    r"""Thermal load.

    A thermal load consists of a heat store (e.g. building, refrigerator), with
    a temperature profile :math:`\theta \in \mathbb{R}^T` which must be
    maintained within :math:`[\theta^\min,\theta^\max]`. The temperature evolves
    as

    .. math::
      \theta(\tau + 1) = \theta(\tau) + (\mu/c)(\theta^\mathrm{amb}(\tau) -
      \theta(\tau)) - (\eta/c)p(\tau)
    ..

    for :math:`\tau = 1,\ldots,T-1` and :math:`\theta(1) =
    \theta^{\mathrm{init}}`, where :math:`\mu` is the ambient conduction
    coefficient, :math:`\eta` is the heating/cooling efficiency, :math:`c` is
    the heat capacity of the heat store, :math:`\theta \in \mathbb{R}^T` is the
    ambient temperature profile and :math:`\theta^{\mathrm{init}}` is the
    initial temperature of the heat store.

    The power consumption of is constrained by

    .. math::
      0 \le p \le P^\max.
    ..

    :param temp_init: Initial tempeature, :math:`\theta^\mathrm{init}`
    :param temp_min: Minimum temperature, :math:`\theta^\min`
    :param temp_max: Maximum temperature, :math:`\theta^\max`
    :param temp_amb: Ambient temperature, :math:`\theta^\mathrm{amb}`
    :param power_max: Maximum power consumption, :math:`P^\max`
    :param amb_conduct_coeff: Ambient conduction coefficient, :math:`\mu`
    :param efficiency: Heating/cooling efficiency, :math:`\eta`
    :param capacity: Heat capacity of the heat store, :math:`c`
    :type temp_init: float
    :type temp_min: float or sequence of floats
    :type temp_max: float or sequence of floats
    :type temp_amb: float or sequence of floats
    :type power_max: float or sequence of floats
    :type amb_conduct_coeff: float or sequence of floats
    :type efficiency: float or sequence of floats
    :type capacity: float or sequence of floats
    """

    def __init__(self,
                 temp_init=None, temp_min=None, temp_max=None, temp_amb=None,
                 power_max=None,
                 amb_conduct_coeff=None,
                 efficiency=None,
                 capacity=None,
                 name=None):
        super(ThermalLoad, self).__init__([Terminal()], name)
        self.temp_init = temp_init
        self.temp_min = temp_min
        self.temp_max = temp_max
        self.temp_amb = temp_amb
        self.power_max = power_max
        self.amb_conduct_coeff = amb_conduct_coeff
        self.efficiency = efficiency
        self.capacity = capacity

    @property
    def constraints(self):
        alpha = self.amb_conduct_coeff / self.capacity
        beta = self.efficiency / self.capacity
        N = self.terminals[0].power_var.shape[0]
        self.temp = cvx.Variable(shape=(N, 1))

        constrs = [
            self.terminals[0].power_var <= self.power_max,
            self.terminals[0].power_var >= 0,
        ]

        if self.temp_max is not None:
            constrs += [self.temp <= self.temp_max]
        if self.temp_min is not None:
            constrs += [self.temp >= self.temp_min]

        for i in range(N):
            temp_prev = self.temp[i - 1] if i else self.temp_init
            constrs += [
                self.temp[i] == (
                    temp_prev + alpha * (self.temp_amb[i] - temp_prev) -
                    beta * self.terminals[0].power_var[i])
            ]

        return constrs


class CurtailableLoad(Device):
    r"""Curtailable load.

    A curtailable load penalizes the shortfall between a desired load profile
    :math:`l \in \mathbb{R}^T` and delivered power with the linear penalty

    .. math::

      \alpha \sum_{\tau=1}^T (l(\tau) - p(\tau))_+

    ..

    where :math:`(z)_+ = \max(0, z)` and :math:`\alpha > 0` is curtailment
    penalty. Conceptually, the linear curtailment penalty represents the price
    paid for reducing load.

    :param power: Desired load profile, :math:`l`
    :param alpha: Linear curtailment penalty, :math:`\alpha`
    :param name: (optional) Display name for load
    :type power: float or sequence of float
    :type alpha: float or sequence of float
    :type name: string
    """

    def __init__(self, power=None, alpha=None, name=None):
        super(CurtailableLoad, self).__init__([Terminal()], name)
        self.power = power
        self.alpha = alpha

    @property
    def cost(self):
        return np.matrix(self.alpha) * cvx.pos(self.power - self.terminals[0].power_var)


class DeferrableLoad(Device):
    r"""Deferrable load.

    A deferrable load must consume a certain amount of energy :math:`E` over a
    flexible time horizon :math:`\tau = A,\ldots,D`. This is characterized by
    the constraint

    .. math::
      \sum_{\tau=A}^{D} p(\tau) \ge E
    ..

    In addition, the power consumption in any interval is bounded by

    .. math::
      p \le P^\max
    ..

    :param energy: Desired energy consumption, :math:`E`
    :param power_max: Maximum power consumption, :math:`P^\max`
    :param time_start: Starting interval, inclusive, :math:`A`
    :param time_end: Ending interval, inclusive, :math:`D`
    :param name: (optional) Display name for load
    :type energy: float
    :type power_max: float or sequence of float
    :type time_start: int
    :type time_end: int
    :type name: string
    """

    def __init__(self, energy=0, power_max=None, time_start=0, time_end=None,
                 name=None, len_interval=1.):
        super(DeferrableLoad, self).__init__([Terminal()], name)
        self.time_start = time_start
        self.time_end = time_end
        self.energy = energy
        self.power_max = power_max
        self.len_interval = len_interval

    @property
    def constraints(self):
        idx = slice(self.time_start, self.time_end)
        return [
            cvx.sum(self.terminals[0].power_var[idx]) *
            self.len_interval == self.energy,
            self.terminals[0].power_var >= 0,
            self.terminals[0].power_var <= self.power_max,
        ]


class TransmissionLine(Device):
    """Transmission line.

    A lossless transmission line has two terminals with power schedules
    :math:`p_1` and :math:`p_2`. Conservation of energy across the
    line is enforced with the constraint

    .. math::
      p_1 + p_2 = 0,
    ..

    and a maximum capacity of :math:`P^\max` with

    .. math::
      |p_1| \le P^\max.
    ..

    :param power_max: Maximum capacity of the transmission line
    :param name: (optional) Display name for transmission line
    :type power_max: float or sequence of floats
    :type name: string
    """

    def __init__(self, alpha=0., power_max=None, name=None):
        super(TransmissionLine, self).__init__([Terminal(), Terminal()], name)
        self.power_max = power_max
        self.alpha = alpha
        assert self.alpha >= 0

    @property
    def constraints(self):
        p1 = self.terminals[0].power_var
        p2 = self.terminals[1].power_var

        constrs = []
        if self.alpha > 0:
            constrs += [p1 + p2 >= self.alpha * cvx.square((p1 - p2) / 2)]
            if self.power_max is not None:
                constrs += [2 * self.alpha * self.power_max**2 >= p1 + p2]
        else:
            constrs += [p1 + p2 == 0]
            if self.power_max is not None:
                constrs += [cvx.abs((p1 - p2) / 2) <= self.power_max]

        return constrs


class Storage(Device):
    r"""Storage device.

    A storage device either takes or delivers power with charging and
    discharging rates specified by the constraints

    .. math::
      -D^\max \le p \le C^\max
    ..

    where :math:`C^\max` and :math:`D^\max` are the maximum charging and
    discharging rates. The charge level of the battery is given by

    .. math::
      q(\tau) = q^\mathrm{init} +  \sum_{t=1}^\tau p(t), \quad \tau = 1, \ldots, T,
    ..

    which is constrained according to the physical limits of the battery

    .. math::
      0 \le q \le Q^\max.
    ..

    :param discharge_max: Maximum discharge rate, :math:`D^\max`
    :param charge_max: Maximum charge rate, :math:`C^\max`
    :param energy_init: Initial charge, :math:`q^\mathrm{init}`
    :param energy_max: Maximum battery capacity, :math:`Q^\max`
    :param name: (optional) Display name of storage device
    :type discharge_max: float or sequence of floats
    :type charge_max: float or sequence of floats
    :type energy_init: float
    :type energy_max: float or sequence of floats
    :type name: string
    """

    def __init__(self, discharge_max=0, charge_max=None, energy_init=0,
                 energy_final=None, energy_max=None, name=None, len_interval=1.,
                 final_energy_price=None):
        super(Storage, self).__init__([Terminal()], name)
        self.discharge_max = discharge_max
        self.charge_max = charge_max
        self.energy_init = energy_init
        self.energy_max = energy_max
        self.energy_final = energy_final
        self.len_interval = len_interval  # in hours
        self.final_energy_price = final_energy_price
        self.energy = None

    @property
    def cost(self):
        T, S = self.terminals[0].power_var.shape
        if self.final_energy_price is not None:
            if self.energy is None:
                self.energy = cvx.Variable(self.terminals[0].power_var.shape)
            cost = np.zeros((T - 1, S))
            final_cost = cvx.reshape(
                self.energy[-1, :] * self.final_energy_price[0, 0], (1, S))
            cost = cvx.vstack([cost, final_cost])
        else:
            cost = np.zeros(T, S)
        return cost

    @property
    def constraints(self):
        P = self.terminals[0].power_var
        if self.energy is None:
            self.energy = cvx.Variable(self.terminals[0].power_var.shape)
        e_init = cvx.reshape(self.energy_init, ())
        constr = [
            cvx.diff(self.energy.T) == P[1:, :] * self.len_interval,
            self.energy[0, :] - e_init - P[0, :] * self.len_interval == 0,
            self.terminals[0].power_var >= -self.discharge_max,
            self.terminals[0].power_var <= self.charge_max,
            self.energy <= self.energy_max,
            self.energy >= 0,
        ]
        if self.energy_final is not None:
            constr += [self.energy[-1] >= self.energy_final]
        return constr
