"""Network of devices."""

import cvxpy as cvx
import numpy as np
import matplotlib.pyplot as plt
import tqdm

class OptimizationError(Exception):
    pass


class Terminal(object):
    pass


class Net(object):
    def __init__(self, terminals, name=None):
        self.name = "Net" if name is None else name
        self.terminals = terminals

    def init(self, time_horizon=1, num_scenarios=1):
        self.constraint = sum(t.power for t in self.terminals) == 0

    def problem(self):
        return cvx.Problem(cvx.Minimize(0), [self.constraint])

    def results(self):
        return Results(price={self: self.price})

    @property
    def price(self):
        return self.constraint.dual_value


class Device(object):
    def __init__(self, terminals, name=None):
        self.name = type(self).__name__ if name is None else name
        self.terminals = terminals

    @property
    def cost(self):
        return 0.0

    @property
    def constraints(self):
        return []

    def init(self, time_horizon=1, num_scenarios=1):
        for terminal in self.terminals:
            terminal.power = cvx.Variable(time_horizon, num_scenarios)

    def problem(self):
        return cvx.Problem(
            cvx.Minimize(self.cost),
            self.constraints +
            [terminal.power[0,k] == terminal.power[0,0]
             for terminal in self.terminals
             for k in xrange(1, terminal.power.size[1])])

    def results(self):
        return Results(power={(self, i): t.power.value
                              for i, t in enumerate(self.terminals)})


class Group(Device):
    def __init__(self, devices, nets, terminals=[], name=None):
        super(Group, self).__init__(terminals, name)
        self.devices = devices
        self.nets = nets

    @property
    def children(self):
        return self.devices + self.nets

    def init(self, time_horizon=1, num_scenarios=1):
        for x in self.children:
            x.init(time_horizon, num_scenarios)

    def problem(self):
        return sum(x.problem() for x in self.children)

    def results(self):
        return sum(x.results() for x in self.children)


class Results(object):
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
    problem = device.problem()
    results = Results()
    for t in tqdm.trange(time_steps):
        predict(t)
        problem.solve(solver=cvx.SCS)
        if problem.status not in (cvx.OPTIMAL, cvx.OPTIMAL_INACCURATE):
            raise OptimizationError(
                "failed at iteration %d, %s" % (t, problem.status))
        execute(t)
        update_mpc_results(t, time_steps, device.results(), results)
    return results
