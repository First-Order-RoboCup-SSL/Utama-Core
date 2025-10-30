from dataclasses import dataclass

from utama_core.config.physical_constants import ROBOT_RADIUS
from utama_core.config.robot_params.grsim import MAX_VEL


@dataclass(slots=True)
class DynamicWindowConfig:
    """Configuration shared by the Dynamic Window planner and controller."""

    simulate_frames: float = 3.0
    max_acceleration: float = 50
    max_safety_radius: float = ROBOT_RADIUS * 2.5
    safety_penalty_distance_sq: float = 0.3
    max_speed_for_full_bubble: float = 1.0
    target_tolerance: float = 0.01
    max_speed: float = MAX_VEL  # this gets overridden in the DWAController based on mode
    n_directions: int = 8
