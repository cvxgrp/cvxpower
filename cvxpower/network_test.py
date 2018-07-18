
import unittest

from cvxpower import *


class ResultsTest(unittest.TestCase):

    def test_summary_normal(self):
        load = FixedLoad(power=10)
        gen = Generator(power_max=10)
        net = Net([load.terminals[0], gen.terminals[0]])
        network = Group([load, gen], [net])
        network.init_problem()
        network.problem.solve()
        assert "optimal" in network.results.summary()

    def test_summary_infeasible(self):
        load = FixedLoad(power=10)
        gen = Generator(power_max=1)
        net = Net([load.terminals[0], gen.terminals[0]])
        network = Group([load, gen], [net])
        network.init_problem()
        network.problem.solve()
        assert "infeasible" in network.results.summary()


class DeviceTest(unittest.TestCase):

    def test_optimize(self):
        load = FixedLoad(power=1)
        gen = Generator(power_max=10)
        net = Net([load.terminals[0], gen.terminals[0]])
        network = Group([load, gen], [net])
        network.optimize()


if __name__ == "__main__":
    unittest.main()
