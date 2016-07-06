"""Network of devices."""

import cvxpy as cvx
import matplotlib.pyplot as plt

class NetworkError(Exception):
    pass


class Terminal(object):
    pass


class Net(object):
    def __init__(self, terminals, name=None):
        self.name = "Net" if name is None else name
        self.terminals = terminals

    def init_constraint(self):
        self.constraint = sum(t.power for t in self.terminals) == 0

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

    def init_variables(self, time_horizon):
        for terminal in self.terminals:
            terminal.power = cvx.Variable(time_horizon)

    def optimize(self, time_horizon=1):
        self.init_variables(time_horizon)
        prob = cvx.Problem(cvx.Minimize(self.cost), self.constraints)
        prob.solve()
        if prob.status != cvx.OPTIMAL:
            raise NetworkError("optimization failed: " + prob.status)


class Group(Device):
    def __init__(self, devices, nets, terminals=[], name=None):
        super(Group, self).__init__(terminals, name)
        self.devices = devices
        self.nets = nets

    @property
    def cost(self):
        return sum(d.cost for d in self.devices)

    @property
    def constraints(self):
        return ([constr for d in self.devices for constr in d.constraints] +
                [n.constraint for n in self.nets])

    def init_variables(self, time_horizon):
        for device in self.devices:
            device.init_variables(time_horizon)

        for net in self.nets:
            net.init_constraint()

    def print_results(self):
        print "%-20s %10s" % ("Terminal", "Power")
        print "%-20s %10s" % ("--------", "-----")
        for device in self.devices:
            for i, terminal in enumerate(device.terminals):
                device_terminal = "%s[%d]" % (device.name, i)
                print "%-20s %10.4f" % (device_terminal, terminal.power.value)

        print
        print "%-20s %10s" % ("Net", "Price")
        print "%-20s %10s" % ("---", "-----")
        for net in self.nets:
            print "%-20s %10.4f" % (net.name, net.price)

    def plot_results(self):
        fig, ax = plt.subplots(nrows=2, ncols=1)

        ax[0].set_ylabel("power")
        for device in self.devices:
            for i, terminal in enumerate(device.terminals):
                device_terminal = "%s[%d]" % (device.name, i)
                ax[0].plot(terminal.power.value, label=device_terminal)
        ax[0].legend()

        ax[1].set_ylabel("price")
        for net in self.nets:
            ax[1].plot(net.price, label=net.name)
        ax[1].legend()

        return ax
