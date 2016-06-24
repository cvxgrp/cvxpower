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
        self.cost = 0.0
        self.constraints = []


def get_problem(devices, nets):
    cost = sum(d.cost for d in devices)

    # Net and device constraints
    constrs = [n.constraint for n in nets]
    for device in devices:
        constrs.extend(device.constraints)
    for net in nets:
        constrs.append(net.constraint)

    return cost, constrs

def optimize(devices, nets):
    cost, constrs = get_problem(devices, nets)
    prob = cvx.Problem(cvx.Minimize(cost), constrs)
    return prob.solve()

def group(devices, nets, terminals):
    cost, constrs = get_problem(devices, nets)
    device = Device(terminals)
    device.cost = cost
    device.constrs = constrs
    return device

def print_results(devices, nets):
    print "%-20s %10s" % ("Terminal", "Power")
    print "%-20s %10s" % ("--------", "-----")
    for device in devices:
        for i, terminal in enumerate(device.terminals):
            device_terminal = "%s[%d]" % (device.name, i)
            print "%-20s %10.4f" % (device_terminal, terminal.power.value)

    print
    print "%-20s %10s" % ("Net", "Price")
    print "%-20s %10s" % ("---", "-----")
    for net in nets:
        print "%-20s %10.4f" % (net.name, net.price)
