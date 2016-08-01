Network Model
=============

.. automodule:: dem.network

The classes below define the abstractions used by the dynamic network energy
optimization framework. For concrete devices which subclass
`Device <#dem.network.Device>`_, see `here <devices.html>`_.

Base
----

.. autoclass:: dem.network.Terminal
   :members:

.. autoclass:: dem.network.Device
   :members:

.. autoclass:: dem.network.Net
   :members:

Groups
------

Groups allow composing device networks in a hierarchical fashion.

.. autoclass:: dem.network.Group
   :members:


Optimization and MPC
--------------------

Utilities for performing optimization, model predictive control and
visualizing the results.

.. autoclass:: OptimizationError

.. autoclass:: dem.network.Results
  :members:

.. automethod:: dem.network.run_mpc
