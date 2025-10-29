from abc import ABC, abstractmethod
from typing import Tuple

from utama_core.config.modes import Mode
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game
from utama_core.rsoccer_simulator.src.ssl.envs import SSLStandardEnv


class MotionController(ABC):
    def __init__(self, mode: Mode, n_friendly: int, rsim_env: SSLStandardEnv | None = None):
        self.mode = mode
        self.n_friendly = n_friendly
        self.rsim_env: SSLStandardEnv | None = rsim_env

    @abstractmethod
    def path_to(
        self,
        game: Game,
        robot_id: int,
        target_pos: Vector2D,
        target_oren: float,
    ) -> Tuple[Vector2D, float]:
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
