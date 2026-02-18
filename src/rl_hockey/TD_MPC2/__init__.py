"""TD-MPC2 package.

Main exports:
- TDMPC2: Main agent class

Model components (from model_definition subpackage):
- Encoder, DynamicsSimple, DynamicsOpponent, Reward, Termination
- QFunction, QEnsemble, Policy, OpponentCloning
- MPPIPlannerSimplePaper

For backward compatibility, import directly from this package or from tdmpc2 module.
"""

from .tdmpc2 import TDMPC2

__all__ = ["TDMPC2"]
