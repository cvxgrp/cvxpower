r"""Optimization over a network of power devices.

In particular, a problem of the form

.. math::

   \DeclareMathOperator{\minimize}{minimize}
   \DeclareMathOperator{\subjectto}{subject\;to}
   \minimize \;\; & \sum_{d\in\mathcal{D}} f_d(p_d) \\
   \subjectto \;\; & \sum_{t \in n} p_t(\tau) = 0, \tau = 1,\ldots,T, n \in \mathcal{N}

"""

import cvxpy as cvx
import numpy as np
import matplotlib.pyplot as plt
import tqdm

class OptimizationError(Exception):
    """Error due to infeasibility or numerical problems during optimziation."""
    pass


class Terminal(object):
    """Device terminal."""

    def power_var(self):
        return self._power

    @property
    def power(self):
        """Power consumed (positive value) or produced (negative value) at this
        terminal."""
        return self._power.value


class Net(object):
    r"""Connection point for device terminals.

    A net :math:`n` defines the power balance constraint :math:`\sum_{t \in n}
    p_t = 0` where the sum is over all terminals connected to the net denoted as
    :math:`t \in n`.
    """

    def __init__(self, terminals, name=None):
        self.name = "Net" if name is None else name
        self.terminals = terminals

    def init_problem(self, time_horizon=1, num_scenarios=1):
        self.constraint = sum(t._power for t in self.terminals) == 0

    @property
    def problem(self):
        return cvx.Problem(cvx.Minimize(0), [self.constraint])

    @property
    def results(self):
        return Results(price={self: self.price})

    @property
    def price(self):
        """Price associated with this net."""
        return self.constraint.dual_value


class Device(object):
    """Base class for network device.

    In order to add a new device model, subclass this and override
    `constraints` and `cost` to define the device-specific cost
    function,  :math:`f_d(p_d)`.

    :param terminals: The terminals of the device
    :type terminals: sequence of Terminal
    :param name: Display name of device
    :type name: str or None
    """

    def __init__(self, terminals, name=None):
        self.name = type(self).__name__ if name is None else name
        self.terminals = terminals

    @property
    def cost(self):
        """Device objective, to be overriden by subclasses.

        :rtype: cvxpy expression
        """
        return 0.0

    @property
    def constraints(self):
        """Device constraints, to be overriden by subclasses.

        :rtype: sequence of cvxpy expressions
        """
        return []

    @property
    def problem(self):
        """The network optimization problem.

        :rtype: cvxpy.Problem
        """
        return cvx.Problem(
            cvx.Minimize(self.cost),
            self.constraints +
            [terminal._power[0,k] == terminal._power[0,0]
             for terminal in self.terminals
             for k in xrange(1, terminal._power.size[1])])

    @property
    def results(self):
        """Network optimization results.

        :rtype: Results
        """
        return Results(power={(self, i): t.power
                              for i, t in enumerate(self.terminals)})

    def init_problem(self, time_horizon=1, num_scenarios=1):
        """Initialize the network optimization problem.

        :param time_horizon: The time horizon :math:`T` to optimize over.
        :param num_scenarios: The number of scenarios for robust MPC.
        :type time_horizon: int
        :type num_scenarios: int
        """
        for terminal in self.terminals:
            terminal._power = cvx.Variable(time_horizon, num_scenarios)


class Group(Device):
    """A single device composed of multiple devices and nets.


    The `Group` device allows for creating new devices composed of existing base
    devices or other groups.

    :param devices: Interal devices to be included
    :param nets: Internal nets to be included
    :param terminals: External terminals for new device
    :param name: Display name of group device
    :type devices: sequence of Device
    :type nets: sequence of Net
    :type terminals: sequence of Terminal
    :type name: str or None
    """
    def __init__(self, devices, nets, terminals=[], name=None):
        super(Group, self).__init__(terminals, name)
        self.devices = devices
        self.nets = nets

    @property
    def children(self):
        return self.devices + self.nets

    @property
    def problem(self):
        return sum(x.problem for x in self.children)

    @property
    def results(self):
        return sum(x.results for x in self.children)

    def init_problem(self, time_horizon=1, num_scenarios=1):
        for x in self.children:
            x.init_problem(time_horizon, num_scenarios)


class Results(object):
    """Network optimization results."""

    def __init__(self, power=None, price=None):
        self.power = power if power else {}
        self.price = price if price else {}

    def __radd__(self, other):
        return self.__add__(other)

    def __add__(self, other):
        if other == 0:
            return self
        power = self.power.copy()
        price = self.price.copy()
        power.update(other.power)
        price.update(other.price)
        return Results(power, price)

    def summary(self):
        """Summary of results.


        :rtype: str
        """
        retval = ""
        retval += "%-20s %10s\n" % ("Terminal", "Power")
        retval += "%-20s %10s\n" % ("--------", "-----")
        for device in self.devices:
            for i, terminal in enumerate(device.terminals):
                device_terminal = "%s[%d]" % (device.name, i)
                reval += "%-20s %10.4f\n" % (device_terminal, terminal.power.value)

        retval += "\n"
        retval += "%-20s %10s\n" % ("Net", "Price")
        retval += "%-20s %10s\n" % ("---", "-----")
        for net in self.nets:
            retval += "%-20s %10.4f\n" % (net.name, net.price)

        return retval

    def plot(self):
        """Plot results."""
        fig, ax = plt.subplots(nrows=2, ncols=1)

        ax[0].set_ylabel("power")
        for device_terminal, value in self.power.iteritems():
            label = "%s[%d]" % (device_terminal[0].name, device_terminal[1])
            ax[0].plot(value, label=label)
        ax[0].legend(loc="best")

        ax[1].set_ylabel("price")
        for net, value in self.price.iteritems():
            ax[1].plot(value, label=net.name)
        ax[1].legend(loc="best")

        return ax


def update_mpc_results(t, time_steps, results_t, results_mpc):
    for key, val in results_t.power.iteritems():
        results_mpc.power.setdefault(key, np.empty(time_steps))[t] = val[0]
    for key, val in results_t.price.iteritems():
        results_mpc.price.setdefault(key, np.empty(time_steps))[t] = val[0]


def run_mpc(device, time_steps, predict, execute):
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
    :type device: Device
    :type time_steps: sequence
    :type predict: single arg function
    :type execute: single arg function
    :returns: Model predictive control results
    :rtype: Results

    """
    problem = device.problem
    results = Results()
    for t in tqdm.trange(time_steps):
        predict(t)
        problem.solve(solver=cvx.SCS)
        if problem.status not in (cvx.OPTIMAL, cvx.OPTIMAL_INACCURATE):
            raise OptimizationError(
                "failed at iteration %d, %s" % (t, problem.status))
        execute(t)
        update_mpc_results(t, time_steps, device.results, results)
    return results
