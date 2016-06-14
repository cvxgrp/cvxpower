"""Device models for dynamic energy management.

See Section 3 of DEM paper for details.
"""

import cvxpy as cvx

from dem.network import Device, Net, Terminal, optimize

class Generator(Device):
    """Generator with quadratic cost."""
    def __init__(
            self,
            p_min=0,
            p_max=0,
            alpha=0,
            beta=0):
        super(Generator, self).__init__([Terminal()])

        p = self.terminals[0].power
        self.cost = alpha*cvx.square(p) - beta*p
        self.constraints = [
            -p <= p_max,
            -p >= p_min
        ]


class Load(Device):
    """Fixed load."""
    def __init__(self, p):
        super(Load, self).__init__([Terminal()])
        self.constraints = [self.terminals[0].power == p]


class TransmissionLine(Device):
    """Transmission line."""
    def __init__(self, p_max=None):
        super(TransmissionLine, self).__init__([Terminal(), Terminal()])

        self.constraints = [
            self.loss == 0,
            cvx.abs(self.power_flow) <= p_max
        ]

    @property
    def loss(self):
        """Loss in the line."""
        return self.terminals[0].power + self.terminals[1].power

    @property
    def power_flow(self):
        """Power flow between terminals."""
        return (self.terminals[0].power - self.terminals[1].power)/2


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

    # Connect terminals to nets
    net1 = Net([
        load1.terminals[0],
        gen1.terminals[0],
        line1.terminals[0],
        line2.terminals[0]])
    net2 = Net([
        load2.terminals[0],
        line1.terminals[1],
        line3.terminals[0]])
    net3 = Net([
        gen2.terminals[0],
        line2.terminals[1],
        line3.terminals[1]])

    optimize([load1, load2, gen1, gen2, line1, line2, line3],
             [net1, net2,  net3])

    # Print results
    print "Load 1:", load1.terminals[0].power.value
    print "Load 2:", load2.terminals[0].power.value
    print "Generator 1:", gen1.terminals[0].power.value
    print "Generator 2:", gen2.terminals[0].power.value
    print "Line 1:", line1.power_flow.value
    print "Line 2:", line2.power_flow.value
    print "Line 3:", line3.power_flow.value

if __name__ == "__main__":
    example_2_4()
