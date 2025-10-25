from abc import ABC, abstractmethod
from typing import Tuple

from utama_core.config.modes import Mode
from utama_core.entities.game import Game


class MotionController(ABC):
    def __init__(self, mode_str: str):
        self.mode: Mode = mode_str

    @abstractmethod
    def path_to(
        self,
        game: Game,
        robot_id: int,
        target_pos: Tuple[float, float],
        target_oren: float,
    ) -> Tuple[Tuple[float, float], float]:
        """
        Calculate the next motion command for the robot to reach the target.
        returns a tuple of (tuple(global_x_velocity, global_y_velocity), angular_velocity).
        """
        ...

    def reset(self, robot_id: int) -> None:
        """
        Reset the internal state of the motion controller for the specified robot.
        """
        pass
