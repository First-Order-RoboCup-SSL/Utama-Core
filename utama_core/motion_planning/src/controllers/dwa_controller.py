from utama_core.config.enums import Mode
from utama_core.config.robot_params.grsim import MAX_VEL
from utama_core.config.robot_params.real import MAX_VEL as REAL_MAX_VEL
from utama_core.config.robot_params.rsim import MAX_VEL as RSIM_MAX_VEL
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game
from utama_core.motion_planning.src.common.motion_controller import MotionController
from utama_core.motion_planning.src.dwa.config import DynamicWindowConfig
from utama_core.motion_planning.src.dwa.planner import DWATranslationController
from utama_core.motion_planning.src.pid.pid import (
    PID,
    get_grsim_pids,
    get_real_pids,
    get_rsim_pids,
)
from utama_core.rsoccer_simulator.src.ssl.envs.standard_ssl import SSLStandardEnv


class DWAController(MotionController):
    def __init__(self, mode: Mode, n_friendly: int, rsim_env: SSLStandardEnv | None):
        super().__init__(mode, n_friendly, rsim_env)
        self._dwa_oren, self._dwa_trans = self._initialize_dwa(mode, n_friendly, rsim_env)

    def _initialize_dwa(
        self, mode: Mode, n_friendly: int, env: SSLStandardEnv | None
    ) -> tuple[PID, DWATranslationController]:
        defaults = DynamicWindowConfig()
        max_acceleration = defaults.max_acceleration
        target_tolerance = defaults.target_tolerance

        if mode == Mode.RSIM:
            pid_oren, _ = get_rsim_pids()
            max_speed = RSIM_MAX_VEL
        elif mode == Mode.GRSIM:
            pid_oren, _ = get_grsim_pids()
            max_speed = MAX_VEL
        elif mode == Mode.REAL:
            pid_oren, _ = get_real_pids()
            max_speed = REAL_MAX_VEL
            max_acceleration = 0.3
            target_tolerance = 0.015

        else:
            raise ValueError(f"Unknown mode enum: {mode}.")

        trans = DWATranslationController(
            DynamicWindowConfig(
                max_speed=max_speed,
                max_acceleration=max_acceleration,
                target_tolerance=target_tolerance,
            ),
            num_robots=n_friendly,
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
