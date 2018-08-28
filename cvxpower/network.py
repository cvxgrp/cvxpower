"""Network of power devices."""

import cvxpy as cvx
import numpy as np
import tqdm


def _get_all_terminals(device):
    """Gets all terminals, including those nested within child devices."""
    if isinstance(device, Group):
        return [t for d in device.devices for t in _get_all_terminals(d)]
    else:
        return device.terminals


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
        self.constraints = [sum(t._power[:, k] for t in self.terminals) == 0
                            for k in range(num_scenarios)]

        self.problem = cvx.Problem(cvx.Minimize(0), self.constraints)

    @property
    def results(self):
        return Results(price={self: self.price})

    @property
    def price(self):
        """Price associated with this net."""
        if (len(self.constraints) == 1 and
                np.size(self.constraints[0].dual_value)) == 1:
            return self.constraints[0].dual_value
        return np.hstack(constr.dual_value.reshape(-1, 1)
                         for constr in self.constraints)


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
        results = sum(x.results for x in self.devices + self.nets)
        results.status = self.problem.status if self.problem else None
        return results

    def _init_problem(self, time_horizon, num_scenarios):
        for device in self.devices:
            device._init_problem(time_horizon, num_scenarios)

        for net in self.nets:
            net._init_problem(time_horizon, num_scenarios)

        self.problem = sum(x.problem for x in self.devices + self.nets)


class Results(object):
    """Network optimization results."""

    def __init__(self, power=None, price=None, status=None):
        self.power = power if power else {}
        self.price = price if price else {}
        self.status = status

    def __radd__(self, other):
        return self.__add__(other)

    def __add__(self, other):
        if other == 0:
            return self
        power = self.power.copy()
        price = self.price.copy()
        power.update(other.power)
        price.update(other.price)
        status = self.status if self.status is not None else other.status
        return Results(power, price, status)

    def __str__(self):
        return self.summary()

    def __repr__(self):
        return self.summary()

    def summary(self):
        """Summary of results.

        :rtype: str
        """
        retval = "Status: " + self.status if self.status else "none"
        if self.status != cvx.OPTIMAL:
            return retval

        retval += "\n"
        retval += "%-20s %10s\n" % ("Terminal", "Power")
        retval += "%-20s %10s\n" % ("--------", "-----")
        for device_terminal, value in self.power.items():
            label = "%s[%d]" % (device_terminal[0].name, device_terminal[1])
            retval += "%-20s %10.4f\n" % (label, value)

        retval += "\n"
        retval += "%-20s %10s\n" % ("Net", "Price")
        retval += "%-20s %10s\n" % ("---", "-----")
        for net, value in self.price.items():
            retval += "%-20s %10.4f\n" % (net.name, value)

        return retval

    def plot(self, index=None, **kwargs):  # , print_terminals=True):
        """Plot results."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(nrows=2, ncols=1, **kwargs)

        ax[0].set_ylabel("power")
        for device_terminal, value in self.power.items():
            # if print_terminals:
            label = "%s[%d]" % (device_terminal[0].name,
                                device_terminal[1])
            # else:
            #    label = device_terminal[0].name
            # pd.Series(value.A1 if 'A1' in dir(value) else value,
            #          index=index).plot(ax=ax[0], label=label)
            if index is None:
                ax[0].plot(value, label=label)
            else:
                ax[0].plot(index, value, label=label)
        ax[0].legend(loc="best")
        # suppress xticks (enzo)
        # ax[0].xaxis.set_major_locator(plt.NullLocator())

        ax[1].set_ylabel("price")
        for net, value in self.price.items():
            # pd.Series(value.A1 if 'A1' in dir(value) else value,
            #          index=index).plot(ax=ax[1], label=net.name)
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


class OptimizationError(Exception):
    """Error due to infeasibility or numerical problems during optimziation."""
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
    problem = device.problem
    results = Results()
    for t in tqdm.trange(time_steps):
        predict(t)
        problem.solve(**kwargs)
        if problem.status != cvx.OPTIMAL:
            # temporary
            raise OptimizationError(
                "failed at iteration %d, %s" % (t, problem.status))
        stage_cost = sum([device.cost[0, 0]
                          for device in device.devices]).value
        #print('at time %s, adding cost %f' % (t, stage_cost))
        total_cost += stage_cost
        execute(t)
        _update_mpc_results(t, time_steps, device.results, results)
    return total_cost, results
