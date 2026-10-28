from abc import ABC, abstractmethod

from utama_core.config.enums import Mode
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game
from utama_core.motion_planning.src.common.pose_ahead import (
    _PARAMS_BY_MODE,
    PoseAhead,
    control_latency_s,
)
from utama_core.rsoccer_simulator.src.ssl.envs import SSLStandardEnv


class MotionController(ABC):
    def __init__(self, mode: Mode, rsim_env: SSLStandardEnv | None = None):
        self.mode = mode
        self.rsim_env: SSLStandardEnv | None = rsim_env
        # `move()` plans from where our robots will be when its command takes effect.
        self.pose_ahead = PoseAhead(control_latency_s(mode))
        # `move()` holds a robot with the ball to these (`RobotParams.CARRY_SPEED`/`CARRY_TURN`).
        params = _PARAMS_BY_MODE[Mode(mode)]
        self.carry_speed: float | None = params.CARRY_SPEED
        self.carry_turn: float | None = params.CARRY_TURN
        # Each robot's kick bend, rad per metre the ball sits left of centre on the dribbler
        # (`RobotProfile.kick_bend_deg_per_cm`); set by `StrategyRunner` on real robots.
        self.kick_bend: dict[int, float] = {}

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

    def sent(self, robot_id: int, velocity: Vector2D, angular_vel: float) -> None:
        """The command `move()` actually sent, when it slowed `calculate()`'s down (a robot with the
        ball, `carry_speed`/`carry_turn`). A controller that ramps from its last output ramps from
        this instead: ramping from the faster one lurched a real robot 0.9 m/s sideways when it
        lost the ball mid-dribble (dribble_and_shoot, 2026-10-28)."""

    def reset(self, robot_id: int) -> None:
        """
        Reset the internal state of the motion controller for the specified robot.
        """
        pass
