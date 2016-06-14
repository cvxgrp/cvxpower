"""Network of devices.

See Section 2 of DEM paper for details.
"""

import cvxpy as cvx


class Terminal(object):
    def __init__(self):
        self.variable = cvx.Variable(1)


class Net(object):
    def __init__(self, terminals):
        self.terminals = terminals


class Device(object):
    def __init__(self, terminals):
        self.terminals = terminals
        self.cost = 0.0
        self.constraints = []


class Network(object):
    def __init__(self):
        self.devices = []
        self.nets = []

    def add_device(self, device):
        self.devices.append(device)

    def add_net(self, *args):
        net = Net(args)
        self.nets.append(net)
        return net

    def optimize(self):
        # Objective: minimize device cost
        obj = cvx.Minimize(sum(d.cost for d in self.devices))

        # Device constraints
        constrs = []
        for device in self.devices:
            constrs.extend(device.constraints)

        # Net constraints
        for net in self.nets:
            constrs.append(sum(t.variable for t in net.terminals) == 0)

        prob = cvx.Problem(obj, constrs)
        return prob.solve(verbose=True)

        # TODO(mwytock): Get dual variables
