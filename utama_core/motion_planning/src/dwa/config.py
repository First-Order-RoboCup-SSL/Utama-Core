from dataclasses import dataclass

from utama_core.config.enums import Mode
from utama_core.config.physical_constants import ROBOT_RADIUS
from utama_core.config.robot_params.grsim import MAX_VEL
from utama_core.config.robot_params.real import MAX_VEL as REAL_MAX_VEL
from utama_core.config.robot_params.rsim import MAX_VEL as RSIM_MAX_VEL


@dataclass(slots=True)
class DynamicWindowConfig:
    """Configuration shared by the Dynamic Window planner and controller."""

    simulate_frames: int = 3
    max_acceleration: float = 8
    max_safety_radius: float = ROBOT_RADIUS * 2.5
    safety_penalty_distance_sq: float = 0.3
    max_speed_for_full_bubble: float = 1.0
    target_tolerance: float = 0.01
    max_speed: float = MAX_VEL
    n_directions: int = 8
    v_resolution = 0.05
    weight_goal = 1.0
    weight_obstacle = 1.0
    weight_speed = 0.1

def get_dwa_config(mode: Mode) -> DynamicWindowConfig:
    """Returns the Dynamic Window configuration."""
    if mode == Mode.RSIM:
        return DynamicWindowConfig(max_speed=RSIM_MAX_VEL)
    elif mode == Mode.GRSIM:
        return DynamicWindowConfig()
    elif mode == Mode.REAL:
        return DynamicWindowConfig(max_speed=REAL_MAX_VEL, max_acceleration=0.3, target_tolerance=0.015)
