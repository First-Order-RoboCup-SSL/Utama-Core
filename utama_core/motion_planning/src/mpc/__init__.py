"""
MPC (Model Predictive Control) Local Planner

Provides optimal local navigation with obstacle avoidance for SSL robots.
"""

from .omni_mpc import OmnidirectionalMPC, OmniMPCConfig

__all__ = [
    "OmnidirectionalMPC",
    "OmniMPCConfig",
]
