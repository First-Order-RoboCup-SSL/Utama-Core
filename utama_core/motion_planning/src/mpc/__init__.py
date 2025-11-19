"""
MPC (Model Predictive Control) Local Planner

Provides optimal local navigation with obstacle avoidance for SSL robots.
"""

from .mpc_config import MPCConfig, get_default_sim_config, get_real_robot_config
from .mpc_planner import MPCPlanner, RobotState, Obstacle
from .omni_mpc import OmnidirectionalMPC, OmniMPCConfig

__all__ = [
    'MPCConfig',
    'MPCPlanner',
    'RobotState',
    'Obstacle',
    'get_default_sim_config',
    'get_real_robot_config',
    'OmnidirectionalMPC',
    'OmniMPCConfig',
]
