"""Network of devices."""

import cvxpy as cvx

class Terminal(object):
    def __init__(self):
        self.power = cvx.Variable(1)


class Net(object):
    def __init__(self, terminals, name=None):
        self.name = "Net" if name is None else name
        self.terminals = terminals
        self.constraint = sum(t.power for t in terminals) == 0

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

    def optimize(self):
        prob = cvx.Problem(cvx.Minimize(self.cost), self.constraints)
        return prob.solve()

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
