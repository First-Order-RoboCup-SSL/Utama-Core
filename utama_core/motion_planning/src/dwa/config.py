from dataclasses import dataclass

from utama_core.config.enums import Mode
from utama_core.config.physical_constants import ROBOT_RADIUS
from utama_core.config.robot_params import GRSIM_PARAMS, REAL_PARAMS, RSIM_PARAMS


@dataclass(slots=True)
class DynamicWindowConfig:
    """Configuration shared by the Dynamic Window planner and controller."""

    max_speed: float
    max_acceleration: float
    simulate_frames: int = 3
    max_safety_radius: float = ROBOT_RADIUS * 2.5
    safety_penalty_distance_sq: float = 0.3
    max_speed_for_full_bubble: float = 1.0
    target_tolerance: float = 0.01
    n_directions: int = 8
    v_resolution = 0.05
    weight_goal = 50.0
    weight_obstacle = 1.0  # Increased to match new (0-10) obstacle cost scale
    weight_speed = 1.0
    show_debug_rectangles = False  # Enable visualization of bounding boxes for testing


def get_dwa_config(mode: Mode) -> DynamicWindowConfig:
    """Returns the Dynamic Window configuration."""
    if mode == Mode.RSIM:
        return DynamicWindowConfig(max_speed=RSIM_PARAMS.MAX_VEL, max_acceleration=RSIM_PARAMS.MAX_ACCELERATION)
    elif mode == Mode.GRSIM:
        return DynamicWindowConfig(max_speed=GRSIM_PARAMS.MAX_VEL, max_acceleration=GRSIM_PARAMS.MAX_ACCELERATION)
    elif mode == Mode.REAL:
        return DynamicWindowConfig(
            max_speed=REAL_PARAMS.MAX_VEL, max_acceleration=REAL_PARAMS.MAX_ACCELERATION, target_tolerance=0.015
        )
    else:
        raise ValueError(f"Unknown mode for DWA config: {mode}")
