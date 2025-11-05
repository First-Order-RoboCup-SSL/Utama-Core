from utama_core.config.enums import Mode
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game
from utama_core.motion_planning.src.common.motion_controller import MotionController
from utama_core.motion_planning.src.dwa.config import get_dwa_config
from utama_core.motion_planning.src.dwa.translation_controller import (
    DWATranslationController,
)
from utama_core.motion_planning.src.pid.pid import PID, get_pids
from utama_core.rsoccer_simulator.src.ssl.envs.standard_ssl import SSLStandardEnv


class DWAController(MotionController):
    def __init__(self, mode: Mode, rsim_env: SSLStandardEnv | None):
        super().__init__(mode, rsim_env)
        self._dwa_oren, self._dwa_trans = self._initialize_dwa(mode, rsim_env)

    def _initialize_dwa(self, mode: Mode, env: SSLStandardEnv | None) -> tuple[PID, DWATranslationController]:

        pid_oren, _ = get_pids(mode)
        dwa_config = get_dwa_config(mode)

        trans = DWATranslationController(
            config=dwa_config,
            env=env,
        )
        return pid_oren, trans

    def calculate(
        self,
        game: Game,
        robot_id: int,
        target_pos: Vector2D,
        target_oren: float,
    ) -> tuple[Vector2D, float]:
        robot = game.friendly_robots[robot_id]
        global_vel = self._dwa_trans.calculate(game, target_pos, robot_id)
        global_oren = self._dwa_oren.calculate(target_oren, robot.orientation, robot_id)
        return global_vel, global_oren

    def reset(self, robot_id):
        self._dwa_oren.reset(robot_id)
        self._dwa_trans.reset(robot_id)
