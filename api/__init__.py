"""An API of DoWhy.

Detailed documentation and user guides are available at
[https://github.com/TejuOye/CausalFast](https://github.com/TejuOye/CausalFast).
"""
__version__ = "0.2.9"
from causalfast.causalfastapi import functions
simulator = functions.simulator
makegraph = functions.makegraph


import dowhy
from dowhy import CausalModel
