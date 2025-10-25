from typing import Tuple

from utama_core.config.modes import Mode
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game
from utama_core.motion_planning.src.common.motion_controller import MotionController
from utama_core.motion_planning.src.dwa.planner import DynamicWindowPlanner
from utama_core.motion_planning.src.pid.pid import (
    PID,
    TwoDPID,
    get_grsim_pids,
    get_real_pids,
    get_rsim_pids,
)


class DWAController(MotionController):
    def __init__(self, mode: Mode):
        super().__init__(mode)
        self.pid_oren, _ = self._initialize_pids(self.mode)
        self._planner = DynamicWindowPlanner()

    def _initialize_pids(self, mode: Mode) -> tuple[PID, TwoDPID]:
        if mode == Mode.RSIM:
            return get_rsim_pids()
        elif mode == Mode.GRSIM:
            return get_grsim_pids()
        elif mode == Mode.REAL:
            return get_real_pids()
        else:
            raise ValueError(f"Unknown mode enum: {mode}.")

    def path_to(
        self,
        game: Game,
        robot_id: int,
        target_pos: Vector2D,
        target_oren: float,
    ) -> Tuple[Vector2D, float]:
        robot = game.friendly_robots[robot_id]
        global_vel, score = self._planner.path_to(game, robot_id, target_pos)
        return global_vel, self.pid_oren.calculate(target_oren, robot.orientation, robot_id)

    def reset(self, robot_id):
        self.pid_oren.reset(robot_id)
