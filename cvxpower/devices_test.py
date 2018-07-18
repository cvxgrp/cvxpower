
import cvxpy as cvx
import numpy as np
import unittest

from cvxpower import *


class StaticTest(unittest.TestCase):

    def test_hello_world(self):
        load = FixedLoad(power=100)
        gen = Generator(power_max=1000, alpha=0.1, beta=100)
        net = Net([load.terminals[0], gen.terminals[0]])
        network = Group([load, gen], [net])
        network.init_problem()
        network.problem.solve()

        np.testing.assert_allclose(load.terminals[0].power, 100)
        np.testing.assert_allclose(gen.terminals[0].power, -100)
        np.testing.assert_allclose(net.price, 120, rtol=1e-2)

    def test_curtailable_load(self):
        load = CurtailableLoad(power=1000, alpha=150)
        gen = Generator(power_max=1000, alpha=1, beta=100)
        net = Net([load.terminals[0], gen.terminals[0]])
        network = Group([load, gen], [net])
        network.init_problem()
        network.problem.solve()

        np.testing.assert_allclose(load.terminals[0].power, 25.00, rtol=1e-2)
        np.testing.assert_allclose(gen.terminals[0].power, -25, rtol=1e-2)
        np.testing.assert_allclose(net.price, 150, rtol=1e-2)

    def test_two_generators_with_transmission(self):
        load = FixedLoad(power=100)
        gen1 = Generator(power_max=1000, alpha=0.01, beta=100, name="Gen1")
        gen2 = Generator(power_max=100, alpha=0.1, beta=0.1, name="Gen2")
        line = TransmissionLine(power_max=50)

        net1 = Net([load.terminals[0], gen1.terminals[0], line.terminals[0]])
        net2 = Net([gen2.terminals[0], line.terminals[1]])

        network = Group([load, gen1, gen2, line], [net1, net2])
        network.init_problem()
        network.problem.solve()

        np.testing.assert_allclose(load.terminals[0].power, 100, rtol=1e-2)
        np.testing.assert_allclose(gen1.terminals[0].power, -50, rtol=1e-2)
        np.testing.assert_allclose(gen2.terminals[0].power, -50, rtol=1e-2)
        np.testing.assert_allclose(line.terminals[0].power, -50, rtol=1e-2)
        np.testing.assert_allclose(line.terminals[1].power,  50, rtol=1e-2)

        np.testing.assert_allclose(net1.price, 101,  rtol=1e-2)
        np.testing.assert_allclose(net2.price, 10.1, rtol=1e-2)

    def test_three_buses(self):
        load1 = FixedLoad(power=50, name="Load1")
        load2 = FixedLoad(power=100, name="Load2")
        gen1 = Generator(power_max=1000, alpha=0.01, beta=100, name="Gen1")
        gen2 = Generator(power_max=100, alpha=0.1, beta=0.1, name="Gen2")
        line1 = TransmissionLine(power_max=50)
        line2 = TransmissionLine(power_max=10)
        line3 = TransmissionLine(power_max=55)

        net1 = Net([load1.terminals[0], gen1.terminals[0],
                    line1.terminals[0], line2.terminals[0]])
        net2 = Net([load2.terminals[0], line1.terminals[1], line3.terminals[0]])
        net3 = Net([gen2.terminals[0], line2.terminals[1], line3.terminals[1]])

        network = Group([load1, load2, gen1, gen2, line1,
                         line2, line3], [net1, net2, net3])
        network.init_problem()
        network.problem.solve()

        np.testing.assert_allclose(load1.terminals[0].power,  50, rtol=1e-2)
        np.testing.assert_allclose(load2.terminals[0].power, 100, rtol=1e-2)
        np.testing.assert_allclose(gen1.terminals[0].power, -85, rtol=1e-2)
        np.testing.assert_allclose(gen2.terminals[0].power, -65, rtol=1e-2)
        np.testing.assert_allclose(line1.terminals[0].power,  45, rtol=1e-2)
        np.testing.assert_allclose(line1.terminals[1].power, -45, rtol=1e-2)
        np.testing.assert_allclose(line2.terminals[0].power, -10, rtol=1e-2)
        np.testing.assert_allclose(line2.terminals[1].power,  10, rtol=1e-2)
        np.testing.assert_allclose(line3.terminals[0].power, -55, rtol=1e-2)
        np.testing.assert_allclose(line3.terminals[1].power,  55, rtol=1e-2)

        np.testing.assert_allclose(net1.price, 101.7, rtol=1e-2)
        np.testing.assert_allclose(net2.price, 101.7, rtol=1e-2)
        np.testing.assert_allclose(net3.price,  13.1, rtol=1e-2)

    def test_group(self):
        solar = Generator(power_max=10, alpha=0, beta=0, name="Solar")
        load = FixedLoad(power=13)
        line = TransmissionLine(power_max=25)
        net = Net([load.terminals[0], solar.terminals[0], line.terminals[0]])
        home = Group([solar, load, line], [net], [
                     line.terminals[1]], name="Home")

        grid = Generator(power_max=1e6, alpha=0.05, beta=100, name="Grid")
        meter = Net([line.terminals[1], grid.terminals[0]], name="Meter")

        network = Group([home, grid], [meter])
        network.init_problem()
        network.problem.solve()

        np.testing.assert_allclose(home.terminals[0].power,  3)
        np.testing.assert_allclose(grid.terminals[0].power, -3)

        np.testing.assert_allclose(net.price, 100.3, rtol=1e-2)

    def test_vary_parameters(self):
        load1 = FixedLoad(power=50, name="Load1")
        load2 = FixedLoad(power=100, name="Load2")
        gen1 = Generator(power_max=100, alpha=1, beta=10, name="Gen1")
        gen2 = Generator(power_max=1000, alpha=0.01, beta=0, name="Gen2")
        line1 = TransmissionLine(power_max=100)
        line2 = TransmissionLine(power_max=10)
        line3 = TransmissionLine(power_max=Parameter(1))

        net1 = Net([load1.terminals[0], gen1.terminals[0],
                    line1.terminals[0], line2.terminals[0]])
        net2 = Net([load2.terminals[0], line1.terminals[1], line3.terminals[0]])
        net3 = Net([gen2.terminals[0], line2.terminals[1], line3.terminals[1]])
        network = Group([load1, load2, gen1, gen2, line1,
                         line2, line3], [net1, net2, net3])
        network.init_problem()
        prob = network.problem

        line3.power_max.value = [50]
        prob.solve()
        np.testing.assert_allclose(load1.terminals[0].power,  50, rtol=1e-2)
        np.testing.assert_allclose(load2.terminals[0].power, 100, rtol=1e-2)
        np.testing.assert_allclose(gen1.terminals[0].power, -90, rtol=1e-2)
        np.testing.assert_allclose(gen2.terminals[0].power, -60, rtol=1e-2)
        np.testing.assert_allclose(line1.terminals[0].power,  50, rtol=1e-2)
        np.testing.assert_allclose(line1.terminals[1].power, -50, rtol=1e-2)
        np.testing.assert_allclose(line2.terminals[0].power, -10, rtol=1e-2)
        np.testing.assert_allclose(line2.terminals[1].power,  10, rtol=1e-2)
        np.testing.assert_allclose(line3.terminals[0].power, -50, rtol=1e-2)
        np.testing.assert_allclose(line3.terminals[1].power,  50, rtol=1e-2)

        np.testing.assert_allclose(net1.price, 190.0136, rtol=1e-2)
        np.testing.assert_allclose(net2.price, 190.0136, rtol=1e-2)
        np.testing.assert_allclose(net3.price,   1.2000, rtol=1e-2)

        line3.power_max.value = [100]
        prob.solve()
        np.testing.assert_allclose(load1.terminals[0].power,   50, rtol=1e-2)
        np.testing.assert_allclose(load2.terminals[0].power,  100, rtol=1e-2)
        np.testing.assert_allclose(gen1.terminals[0].power,  -40, rtol=1e-2)
        np.testing.assert_allclose(gen2.terminals[0].power, -110, rtol=1e-2)
        np.testing.assert_allclose(line1.terminals[0].power,    0, atol=1e-4)
        np.testing.assert_allclose(line1.terminals[1].power,    0, atol=1e-4)
        np.testing.assert_allclose(line2.terminals[0].power,  -10, rtol=1e-2)
        np.testing.assert_allclose(line2.terminals[1].power,   10, rtol=1e-2)
        np.testing.assert_allclose(line3.terminals[0].power, -100, rtol=1e-2)
        np.testing.assert_allclose(line3.terminals[1].power,  100, rtol=1e-2)

        np.testing.assert_allclose(net1.price, 89.9965, rtol=1e-2)
        np.testing.assert_allclose(net2.price, 89.9965, rtol=1e-2)
        np.testing.assert_allclose(net3.price,  2.2009, rtol=1e-2)


T = 10
p_load = (np.sin(np.pi * np.arange(T) / T) + 1e-2).reshape(-1, 1)


class DynamicTest(unittest.TestCase):

    def test_dynamic_load(self):
        load = FixedLoad(power=p_load)
        gen = Generator(power_max=2, power_min=-0.01, alpha=100, beta=100)
        net = Net([load.terminals[0], gen.terminals[0]])
        network = Group([load, gen], [net])

        network.init_problem(time_horizon=T)
        network.problem.solve()
        np.testing.assert_allclose(load.terminals[0].power,  p_load, atol=1e-4)
        np.testing.assert_allclose(gen.terminals[0].power, -p_load, atol=1e-4)
        np.testing.assert_allclose(net.price, p_load * 200 + 100, rtol=1e-2)

    def test_storage(self):
        load = FixedLoad(power=p_load)
        gen = Generator(power_max=2, alpha=100, beta=100)
        storage = Storage(charge_max=0.1, discharge_max=0.1, energy_max=0.5)

        net = Net([load.terminals[0], gen.terminals[0], storage.terminals[0]])
        network = Group([load, gen, storage], [net])
        network.init_problem(time_horizon=T)
        network.problem.solve()

    def test_deferrable_load(self):
        load = FixedLoad(power=p_load)
        gen = Generator(power_max=2, alpha=100, beta=100)
        deferrable = DeferrableLoad(time_start=5, energy=0.5, power_max=0.1)

        net = Net([load.terminals[0], gen.terminals[
                  0], deferrable.terminals[0]])
        network = Group([load, gen, deferrable], [net])
        network.init_problem(time_horizon=T)
        network.problem.solve()

    def test_thermal_load(self):
        temp_amb = (np.sin(np.pi * np.arange(T) / T) +
                    1e-2).reshape(-1, 1)**2 * 50 + 50

        load = FixedLoad(power=p_load)
        gen = Generator(power_max=2, alpha=100, beta=100)
        thermal = ThermalLoad(
            temp_init=60, temp_amb=temp_amb, temp_max=90,
            power_max=0.1, amb_conduct_coeff=0.1, efficiency=0.95, capacity=1)

        net = Net([load.terminals[0], gen.terminals[0], thermal.terminals[0]])
        network = Group([load, gen, thermal], [net])
        network.init_problem(time_horizon=T)
        network.problem.solve()

#
# TODO(mwytock): Robust test cases
#


#
# TODO(mwytock): MPC test cases
#

if __name__ == "__main__":
    unittest.main()
