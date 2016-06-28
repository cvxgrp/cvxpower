
from dem import *

import numpy as np


def test_hello_world():
    load = FixedLoad(p=100)
    gen = Generator(p_max=1000, alpha=0.1, beta=100)
    net = Net([load.terminals[0], gen.terminals[0]])
    network = Group([load, gen], [net])
    network.optimize()

    np.testing.assert_allclose(load.terminals[0].power.value, 100)
    np.testing.assert_allclose(gen.terminals[0].power.value, -100)
    np.testing.assert_allclose(net.price, 120, rtol=1e-4)


def test_curtailable_load():
    load = CurtailableLoad(p=1000, alpha=150)
    gen = Generator(p_max=1000, alpha=1, beta=100)
    net = Net([load.terminals[0], gen.terminals[0]])
    network = Group([load, gen], [net])
    network.optimize()

    np.testing.assert_allclose(load.terminals[0].power.value, 25.00, rtol=1e-3)
    np.testing.assert_allclose(gen.terminals[0].power.value, -25, rtol=1e-3)
    np.testing.assert_allclose(net.price, 150, rtol=1e-4)


def test_two_generators_with_transmission():
    load = FixedLoad(p=100)
    gen1 = Generator(p_max=1000, alpha=0.01, beta=100, name="Gen1")
    gen2 = Generator(p_max=100, alpha=0.1, beta=0.1, name="Gen2")
    line = TransmissionLine(p_max=50)

    net1 = Net([load.terminals[0], gen1.terminals[0], line.terminals[0]])
    net2 = Net([gen2.terminals[0], line.terminals[1]])

    network = Group([load, gen1, gen2, line], [net1, net2])
    network.optimize()

    np.testing.assert_allclose(load.terminals[0].power.value, 100, rtol=1e-4)
    np.testing.assert_allclose(gen1.terminals[0].power.value, -50, rtol=1e-4)
    np.testing.assert_allclose(gen2.terminals[0].power.value, -50, rtol=1e-4)
    np.testing.assert_allclose(line.terminals[0].power.value, -50, rtol=1e-4)
    np.testing.assert_allclose(line.terminals[1].power.value,  50, rtol=1e-4)

    np.testing.assert_allclose(net1.price, 101,  rtol=1e-4)
    np.testing.assert_allclose(net2.price, 10.1, rtol=1e-4)


def test_three_buses():
    load1 = FixedLoad(p=50, name="Load1")
    load2 = FixedLoad(p=100, name="Load2")
    gen1 = Generator(p_max=1000, alpha=0.01, beta=100, name="Gen1")
    gen2 = Generator(p_max=100, alpha=0.1, beta=0.1, name="Gen2")
    line1 = TransmissionLine(p_max=50)
    line2 = TransmissionLine(p_max=10)
    line3 = TransmissionLine(p_max=50)

    net1 = Net([load1.terminals[0], gen1.terminals[0], line1.terminals[0], line2.terminals[0]])
    net2 = Net([load2.terminals[0], line1.terminals[1], line3.terminals[0]])
    net3 = Net([gen2.terminals[0], line2.terminals[1], line3.terminals[1]])

    network = Group([load1, load2, gen1, gen2, line1, line2, line3], [net1, net2, net3])
    network.optimize()

    np.testing.assert_allclose(load1.terminals[0].power.value,  50, rtol=1e-4)
    np.testing.assert_allclose(load2.terminals[0].power.value, 100, rtol=1e-4)
    np.testing.assert_allclose( gen1.terminals[0].power.value, -90, rtol=1e-4)
    np.testing.assert_allclose( gen2.terminals[0].power.value, -60, rtol=1e-4)
    np.testing.assert_allclose(line1.terminals[0].power.value,  50, rtol=1e-4)
    np.testing.assert_allclose(line1.terminals[1].power.value, -50, rtol=1e-4)
    np.testing.assert_allclose(line2.terminals[0].power.value, -10, rtol=1e-4)
    np.testing.assert_allclose(line2.terminals[1].power.value,  10, rtol=1e-4)
    np.testing.assert_allclose(line3.terminals[0].power.value, -50, rtol=1e-4)
    np.testing.assert_allclose(line3.terminals[1].power.value,  50, rtol=1e-4)

    np.testing.assert_allclose(net1.price, 101.8008,  rtol=1e-4)
    np.testing.assert_allclose(net2.price, 196.1916, rtol=1e-4)
    np.testing.assert_allclose(net3.price,  12.0975, rtol=1e-4)


def test_group():
    solar = Generator(p_max=10, alpha=0, beta=0, name="Solar")
    load = FixedLoad(p=13)
    line = TransmissionLine(p_max=25)
    net = Net([load.terminals[0], solar.terminals[0], line.terminals[0]])
    home = Group([solar, load, line], [net], [line.terminals[1]], name="Home")

    grid = Generator(p_max=1e6, alpha=0.05, beta=100, name="Grid")
    meter = Net([line.terminals[1], grid.terminals[0]], name="Meter")

    network = Group([home, grid], [meter])
    network.optimize()

    np.testing.assert_allclose(home.terminals[0].power.value,  3)
    np.testing.assert_allclose(grid.terminals[0].power.value, -3)

    np.testing.assert_allclose(net.price, 100.3, rtol=1e-4)


def test_vary_parameters():
    load1 = FixedLoad(p=50, name="Load1")
    load2 = FixedLoad(p=100, name="Load2")
    gen1 = Generator(p_max=100, alpha=1, beta=10, name="Gen1")
    gen2 = Generator(p_max=1000, alpha=0.01, beta=0, name="Gen2")
    line1 = TransmissionLine(p_max=100)
    line2 = TransmissionLine(p_max=10)
    line3 = TransmissionLine()

    net1 = Net([load1.terminals[0], gen1.terminals[0], line1.terminals[0], line2.terminals[0]])
    net2 = Net([load2.terminals[0], line1.terminals[1], line3.terminals[0]])
    net3 = Net([gen2.terminals[0], line2.terminals[1], line3.terminals[1]])
    network = Group([load1, load2, gen1, gen2, line1, line2, line3], [net1, net2, net3])

    line3.p_max = 50
    network.optimize()
    np.testing.assert_allclose(load1.terminals[0].power.value,  50, rtol=1e-4)
    np.testing.assert_allclose(load2.terminals[0].power.value, 100, rtol=1e-4)
    np.testing.assert_allclose( gen1.terminals[0].power.value, -90, rtol=1e-4)
    np.testing.assert_allclose( gen2.terminals[0].power.value, -60, rtol=1e-4)
    np.testing.assert_allclose(line1.terminals[0].power.value,  50, rtol=1e-4)
    np.testing.assert_allclose(line1.terminals[1].power.value, -50, rtol=1e-4)
    np.testing.assert_allclose(line2.terminals[0].power.value, -10, rtol=1e-4)
    np.testing.assert_allclose(line2.terminals[1].power.value,  10, rtol=1e-4)
    np.testing.assert_allclose(line3.terminals[0].power.value, -50, rtol=1e-4)
    np.testing.assert_allclose(line3.terminals[1].power.value,  50, rtol=1e-4)

    np.testing.assert_allclose(net1.price, 190.0136, rtol=1e-4)
    np.testing.assert_allclose(net2.price, 190.0136, rtol=1e-4)
    np.testing.assert_allclose(net3.price,   1.2002, rtol=1e-4)


    line3.p_max = 100
    network.optimize()
    np.testing.assert_allclose(load1.terminals[0].power.value,   50, rtol=1e-4)
    np.testing.assert_allclose(load2.terminals[0].power.value,  100, rtol=1e-4)
    np.testing.assert_allclose( gen1.terminals[0].power.value,  -40, rtol=1e-4)
    np.testing.assert_allclose( gen2.terminals[0].power.value, -110, rtol=1e-4)
    np.testing.assert_allclose(line1.terminals[0].power.value,    0, atol=1e-4)
    np.testing.assert_allclose(line1.terminals[1].power.value,    0, atol=1e-4)
    np.testing.assert_allclose(line2.terminals[0].power.value,  -10, rtol=1e-4)
    np.testing.assert_allclose(line2.terminals[1].power.value,   10, rtol=1e-4)
    np.testing.assert_allclose(line3.terminals[0].power.value, -100, rtol=1e-4)
    np.testing.assert_allclose(line3.terminals[1].power.value,  100, rtol=1e-4)

    np.testing.assert_allclose(net1.price, 89.9965, rtol=1e-4)
    np.testing.assert_allclose(net2.price, 89.9965, rtol=1e-4)
    np.testing.assert_allclose(net3.price,  2.2009, rtol=1e-4)
