"""Device models for dynamic energy management.

See Section 3 of DEM paper for details.
"""

import cvxpy as cvx

from dem.network import Device, Terminal, Network

class Generator(Device):
    """Generator with quadratic cost."""
    def __init__(
            self,
            p_min=0,
            p_max=0,
            alpha=0,
            beta=0.):
        super(Generator, self).__init__([Terminal()])

        self.cost = alpha*cvx.square(self.p) - beta*self.p
        self.constraints = [
            -self.p <= p_max,
            -self.p >= p_min
        ]

    @property
    def p(self):
        return self.terminals[0].variable


class Load(Device):
    """Fixed load."""
    def __init__(self, p):
        super(Load, self).__init__([Terminal()])
        self.constraints = [self.p == p]

    @property
    def p(self):
        return self.terminals[0].variable


class TransmissionLine(Device):
    """Transmission line."""
    def __init__(self, p_max=None):
        super(TransmissionLine, self).__init__([Terminal(), Terminal()])

        self.constraints = [
            self.loss == 0,
            cvx.abs(self.p) <= p_max
        ]

    @property
    def loss(self):
        """Loss in the line."""
        return self.terminals[0].variable + self.terminals[1].variable

    @property
    def p(self):
        """Power flow between terminals."""
        return (self.terminals[0].variable - self.terminals[1].variable)/2


def example_2_4():
    """Example with network from Section 2.4, excluding battery"""

    # Create devices
    load1 = Load(p=50)
    load2 = Load(p=100)
    gen1 = Generator(p_max=1000, alpha=0.01, beta=100)
    gen2 = Generator(p_max=100, alpha=0.1, beta=0.1)
    line1 = TransmissionLine(p_max=50)
    line2 = TransmissionLine(p_max=10)
    line3 = TransmissionLine(p_max=50)

    # Create the network, add devices
    network = Network()
    network.add_device(load1)
    network.add_device(load2)
    network.add_device(gen1)
    network.add_device(gen2)
    network.add_device(line1)
    network.add_device(line2)
    network.add_device(line3)

    # Connect terminals to nets
    net1 = network.add_net(
        load1.terminals[0],
        gen1.terminals[0],
        line1.terminals[0],
        line2.terminals[0])
    net2 = network.add_net(
        load2.terminals[0],
        line1.terminals[1],
        line3.terminals[0])
    net3 = network.add_net(
        gen2.terminals[0],
        line2.terminals[1],
        line3.terminals[1])


    # Optimize
    network.optimize()

    # Print results
    print "Load 1:", load1.p.value
    print "Load 2:", load2.p.value
    print "Generator 1:", gen1.p.value
    print "Generator 2:", gen2.p.value
    print "Line 1:", line1.p.value
    print "Line 2:", line2.p.value
    print "Line 3:", line3.p.value

if __name__ == "__main__":
    example_2_4()
