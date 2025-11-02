from abc import ABC, abstractmethod

from utama_core.config.enums import Mode
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game
from utama_core.rsoccer_simulator.src.ssl.envs import SSLStandardEnv


class MotionController(ABC):
    def __init__(self, mode: Mode, rsim_env: SSLStandardEnv | None = None):
        self.mode = mode
        self.rsim_env: SSLStandardEnv | None = rsim_env

    @abstractmethod
    def calculate(
        self,
        game: Game,
        robot_id: int,
        target_pos: Vector2D,
        target_oren: float,
    ) -> tuple[Vector2D, float]:
        """
        Calculate the next motion command for the robot to reach the target.
        returns a tuple of (Vector2D global velocity, angular_velocity).
        """
        ...

    def reset(self, robot_id: int) -> None:
        """
        Reset the internal state of the motion controller for the specified robot.
        """
        pass
