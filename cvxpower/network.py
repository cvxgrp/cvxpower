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


__doc__ = """Network of power devices."""

import cvxpy as cvx
import numpy as np
import tqdm


def _get_all_terminals(device):
    """Gets all terminals, including those nested within child devices."""
    terms = device.terminals
    if hasattr(device, 'internal_terminal'):
        terms += [device.internal_terminal]
    if hasattr(device, 'devices'):
        terms += [t for d in device.devices for t in _get_all_terminals(d)]
    return terms


class Terminal(object):
    """Device terminal."""

    @property
    def power_var(self):
        return self._power

    @property
    def power(self):
        """Power consumed (positive value) or produced (negative value) at this
        terminal."""
        return self._power.value

    def _init_problem(self, time_horizon, num_scenarios):
        self._power = cvx.Variable(shape=(time_horizon, num_scenarios))

    def _set_payments(self, price):
        if price is not None and self._power.value is not None:
            self.payment = price * self._power.value
        else:
            self.payment = None


class Net(object):
    r"""Connection point for device terminals.

    A net defines the power balance constraint ensuring that power sums to zero
    across all connected terminals at each time point.

    :param terminals: The terminals connected to this net
    :param name: (optional) Display name of net
    :type terminals: list of :class:`Terminal`
    :type name: string
    """

    def __init__(self, terminals, name=None):
        self.name = "Net" if name is None else name
        self.terminals = terminals

    def _init_problem(self, time_horizon, num_scenarios):
        self.num_scenarios = num_scenarios
        # self.constraints = [sum(t._power[:, k] for t in self.terminals) == 0
        #                    for k in range(num_scenarios)]
        self.constraints = [
            sum(t._power for t in self.terminals) / num_scenarios == 0]

        self.problem = cvx.Problem(cvx.Minimize(0), self.constraints)

    def _set_payments(self):
        for t in self.terminals:
            t._set_payments(self.price)

    @property
    def results(self):
        return Results(price={self: self.price})

    @property
    def price(self):
        """Price associated with this net."""
        return self.constraints[0].dual_value
        #print([c.dual_value for c in self.constraints])
        # raise
        # if (len(self.constraints) == 1 and
        #        np.size(self.constraints[0].dual_value)) == 1:
        #    return self.constraints[0].dual_value
        # TODO(enzo) hardcoded 1/K probability
        # return np.sum(constr.dual_value
        # for constr in self.constraints)
        # if self.num_scenarios > 1:
        #    return np.matrix(np.sum([constr.dual_value[0]
        #                             for constr in self.constraints], 0))
        # return np.hstack(constr.dual_value.reshape(-1, 1)
        #                 for constr in self.constraints)


class Device(object):
    """Base class for network device.

    Subclasses are expected to override :attr:`constraints` and/or
    :attr:`cost` to define the device-specific cost function.

    :param terminals: The terminals of the device
    :param name: (optional) Display name of device
    :type terminals: list of :class:`Terminal`
    :type name: string
    """

    def __init__(self, terminals, name=None):
        self.name = type(self).__name__ if name is None else name
        self.terminals = terminals
        self.problem = None

    @property
    def cost(self):
        """Device objective, to be overriden by subclasses.

        :rtype: cvxpy expression of size :math:`T \times K`
        """
        return np.matrix(0.0)

    @property
    def constraints(self):
        """Device constraints, to be overriden by subclasses.

        :rtype: list of cvxpy constraints
        """
        return []

    @property
    def results(self):
        """Network optimization results.

        :rtype: :class:`Results`
        """
        status = self.problem.status if self.problem else None
        return Results(power={(self, i): t.power
                              for i, t in enumerate(self.terminals)},
                       payments={(self, i): t.payment
                                 for i, t in enumerate(self.terminals)},
                       status=status)

    def _init_problem(self, time_horizon, num_scenarios):
        self.problem = cvx.Problem(
            cvx.Minimize(cvx.sum(cvx.sum(self.cost, axis=1)) / num_scenarios),
            # TODO(enzo) we should weight by probs
            self.constraints +
            [terminal._power[0, k] == terminal._power[0, 0]
             for terminal in self.terminals
             for k in range(1, terminal._power.shape[1] if
                            len(terminal._power.shape) > 1 else 0)])

    def init_problem(self, time_horizon=1, num_scenarios=1):
        """Initialize the network optimization problem.

        :param time_horizon: The time horizon :math:`T` to optimize over.
        :param num_scenarios: The number of scenarios for robust MPC.
        :type time_horizon: int
        :type num_scenarios: int
        """
        for terminal in _get_all_terminals(self):
            terminal._init_problem(time_horizon, num_scenarios)

        self._init_problem(time_horizon, num_scenarios)

    def optimize(self, time_horizon=1, num_scenarios=1, **kwargs):
        self.init_problem(time_horizon, num_scenarios)
        self.problem.solve(**kwargs)
        return self.results


class Group(Device):
    """A single device composed of multiple devices and nets.


    The `Group` device allows for creating new devices composed of existing base
    devices or other groups.

    :param devices: Internal devices to be included.
    :param nets: Internal nets to be included.
    :param terminals: (optional) Terminals for new device.
    :param name: (optional) Display name of group device
    :type devices: list of :class:`Device`
    :type nets: list of :class:`Net`
    :type terminals: list of :class:`Terminal`
    :type name: string
    """

    def __init__(self, devices, nets, terminals=[], name=None):
        super(Group, self).__init__(terminals, name)
        self.devices = devices
        self.nets = nets

    @property
    def results(self):
        for n in self.nets:
            n._set_payments()
        results = sum(x.results for x in self.devices + self.nets)
        results.status = self.problem.status if self.problem else None
        return results

    def _init_problem(self, time_horizon, num_scenarios):
        for device in self.devices:
            device._init_problem(time_horizon, num_scenarios)

        for net in self.nets:
            net._init_problem(time_horizon, num_scenarios)

        self.problem = sum(x.problem for x in self.devices + self.nets)

    # def optimize(self, **kwargs):
    #    super(Group, self).optimize(**kwargs)
    #    for n in self.nets:
    #        n._set_payments
    #    raise


class Results(object):
    """Network optimization results."""

    def __init__(self, power=None, payments=None, price=None, status=None):
        self.power = power if power else {}
        self.payments = payments if payments else {}
        self.price = price if price else {}
        self.status = status

    def __radd__(self, other):
        return self.__add__(other)

    def __add__(self, other):
        if other == 0:
            return self
        power = self.power.copy()
        payments = self.payments.copy()
        price = self.price.copy()

        power.update(other.power)
        payments.update(other.payments)
        price.update(other.price)
        status = self.status if self.status is not None else other.status
        return Results(power, payments, price, status)

    def __str__(self):
        return self.summary()

    def __repr__(self):
        return self.summary()

    def summary(self):
        """Summary of results. Only works for single period optimization.

        :rtype: str
        """

        retval = "Status: " + self.status if self.status else "none"
        if not self.status in {cvx.OPTIMAL, cvx.OPTIMAL_INACCURATE}:
            return retval

        retval += "\n"
        retval += "%-20s %10s\n" % ("Terminal", "Power")
        retval += "%-20s %10s\n" % ("--------", "-----")
        averages = False
        for device_terminal, value in self.power.items():
            label = "%s[%d]" % (device_terminal[0].name, device_terminal[1])
            if isinstance(value, np.ndarray):
                value = np.mean(value)
                averages = True
            retval += "%-20s %10.2f\n" % (label, value)

        retval += "\n"
        retval += "%-20s %10s\n" % ("Net", "Price")
        retval += "%-20s %10s\n" % ("---", "-----")
        for net, value in self.price.items():
            if isinstance(value, np.ndarray):
                value = np.mean(value)
            retval += "%-20s %10.4f\n" % (net.name, value)

        retval += "\n"
        retval += "%-20s %10s\n" % ("Device", "Payment")
        retval += "%-20s %10s\n" % ("------", "-------")
        device_payments = {d[0][0]: 0 for d in self.payments.items()}
        for device_terminal, value in self.payments.items():
            if isinstance(value, np.ndarray):
                value = np.sum(value)
            device_payments[device_terminal[0]] += value
        for d in device_payments.keys():
            retval += "%-20s %10.2f\n" % (d.name, device_payments[d])

        if averages:
            retval += "\nPower and price are averages over the time horizon. Payment is total.\n"

        return retval

    def plot(self, index=None, **kwargs):  # , print_terminals=True):
        """Plot results."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(nrows=2, ncols=1, **kwargs)

        ax[0].set_ylabel("power")
        for device_terminal, value in self.power.items():
            label = "%s[%d]" % (device_terminal[0].name,
                                device_terminal[1])
            if index is None:
                ax[0].plot(value, label=label)
            else:
                ax[0].plot(index, value, label=label)
        ax[0].legend(loc="best")

        ax[1].set_ylabel("price")
        for net, value in self.price.items():
            if index is None:
                ax[1].plot(value, label=net.name)
            else:
                ax[1].plot(index, value, label=net.name)
        ax[1].legend(loc="best")

        return ax


def _update_mpc_results(t, time_steps, results_t, results_mpc):
    for key, val in results_t.power.items():
        results_mpc.power.setdefault(key, np.empty(time_steps))[t] = val[0, 0]
    for key, val in results_t.price.items():
        results_mpc.price.setdefault(key, np.empty(time_steps))[t] = val[0, 0]
    for key, val in results_t.payments.items():
        results_mpc.payments.setdefault(
            key, np.empty(time_steps))[t] = val[0, 0]


class OptimizationError(Exception):
    """Error due to infeasibility or numerical problems during optimization."""
    pass


def run_mpc(device, time_steps, predict, execute, **kwargs):
    """Execute model predictive control.

    This method executes the model predictive control loop, roughly:

    .. code:: python

      for t in time_steps:
        predict(t)
        device.problem.solve()
        execute(t)
    ..

    It is the responsibility of the provided `predict` and `execute` functions
    to update the device models with the desired predictions and execute the
    actions as appropriate.

    :param device: Device (or network of devices) to optimize
    :param time_steps: Time steps to optimize over
    :param predict: Prediction step
    :param execute: Execution step
    :type device: :class:`Device`
    :type time_steps: sequence
    :type predict: single argument function
    :type execute: single argument function
    :returns: Model predictive control results
    :rtype: :class:`Results`
    :raise: :class:`OptimizationError`

    """
    total_cost = 0.
    results = Results()
    #T_MPC = device
    for t in tqdm.trange(time_steps):
        predict(t)

        # device.init_problem(time_horizon=1)
        device.problem.solve(**kwargs)
        if device.problem.status != cvx.OPTIMAL:
            # temporary
            raise OptimizationError(
                "failed at iteration %d, %s" % (t, device.problem.status))
        stage_cost = sum([device.cost[0, 0]
                          for device in device.devices]).value
        #print('at time %s, adding cost %f' % (t, stage_cost))
        total_cost += stage_cost
        execute(t)
        _update_mpc_results(t, time_steps, device.results, results)
    return total_cost, results
