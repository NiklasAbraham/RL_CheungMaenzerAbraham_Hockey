"""TD-MPC2 package.

Main exports:
- TDMPC2: Main agent class

Model components (from model_definition subpackage):
- Encoder, DynamicsSimple, DynamicsOpponent, Reward, Termination
- QFunction, QEnsemble, Policy, OpponentCloning
- MPPIPlannerSimplePaper

For backward compatibility, import directly from this package or from tdmpc2 module.
"""

from .core.agent import TDMPC2
from .model_definition import (
    DynamicsOpponent,
    DynamicsSimple,
    Encoder,
    MPPIPlannerSimplePaper,
    Policy,
    QEnsemble,
    Reward,
)

__all__ = [
    "TDMPC2",
    "Encoder",
    "DynamicsSimple",
    "DynamicsOpponent",
    "Reward",
    "QEnsemble",
    "Policy",
    "MPPIPlannerSimplePaper",
]
