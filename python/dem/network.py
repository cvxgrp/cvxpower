"""Network of devices.

See Section 2 of DEM paper for details.
"""

import cvxpy as cvx


class Terminal(object):
    def __init__(self):
        self.power = cvx.Variable(1)


class Net(object):
    def __init__(self, terminals):
        self.terminals = terminals


class Device(object):
    def __init__(self, terminals):
        self.terminals = terminals
        self.cost = 0.0
        self.constraints = []


def optimize(devices, nets):
    obj = cvx.Minimize(sum(d.cost for d in devices))

    # Device constraints
    constrs = []
    for device in devices:
        constrs.extend(device.constraints)

    # Net constraints
    for net in nets:
        constrs.append(sum(t.power for t in net.terminals) == 0)

    prob = cvx.Problem(obj, constrs)
    return prob.solve(verbose=True)
