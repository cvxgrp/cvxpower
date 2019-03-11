"""
Copyright (C) Matt Wytock, Enzo Busseti, Nicholas Moehle, 2016-2019.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Code written by Nicholas Moehle before January 2019 is licensed
under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


import unittest

from cvxpower import *


class ResultsTest(unittest.TestCase):

    def test_summary_normal(self):
        load = FixedLoad(power=10)
        gen = Generator(power_max=10)
        net = Net([load.terminals[0], gen.terminals[0]])
        network = Group([load, gen], [net])
        network.init_problem()
        network.problem.solve(solver='ECOS')
        assert "optimal" in network.results.summary()

    def test_summary_infeasible(self):
        load = FixedLoad(power=10)
        gen = Generator(power_max=1)
        net = Net([load.terminals[0], gen.terminals[0]])
        network = Group([load, gen], [net])
        network.init_problem()
        network.problem.solve(solver='ECOS')
        assert "infeasible" in network.results.summary()


class DeviceTest(unittest.TestCase):

    def test_optimize(self):
        load = FixedLoad(power=1)
        gen = Generator(power_max=10)
        net = Net([load.terminals[0], gen.terminals[0]])
        network = Group([load, gen], [net])
        network.optimize(solver='ECOS')


if __name__ == "__main__":
    unittest.main()
