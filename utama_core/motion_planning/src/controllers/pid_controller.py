from typing import Tuple

from utama_core.config.enums import Mode
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game
from utama_core.motion_planning.src.common.motion_controller import MotionController
from utama_core.motion_planning.src.pid.pid import get_pids
from utama_core.rsoccer_simulator.src.ssl.envs import SSLStandardEnv


class PIDController(MotionController):
    def __init__(self, mode: Mode, rsim_env: SSLStandardEnv | None = None):
        super().__init__(mode, rsim_env)
        self.pid_oren, self.pid_trans = get_pids(mode)

    def calculate(
        self,
        game: Game,
        robot_id: int,
        target_pos: Vector2D,
        target_oren: float,
    ) -> Tuple[Vector2D, float]:
        robot = game.friendly_robots[robot_id]
        return self.pid_trans.calculate(target_pos, robot.p, robot_id), self.pid_oren.calculate(
            target_oren, robot.orientation, robot_id
        )

    def reset(self, robot_id):
        self.pid_oren.reset(robot_id)
        self.pid_trans.reset(robot_id)
