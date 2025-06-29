from motion_planning.src.pid.pid import (
    PID,
    TwoDPID,
    get_grsim_pids,
    get_real_pids,
    get_rsim_pids,
)
from motion_planning.src.planning.planner import DynamicWindowPlanner
from entities.game import Game
from shapely.geometry import LineString
from typing import List, Tuple


class MotionController:
    def __init__(self, mode: str):
        self._pid_oren, self._pid_trans = self._initialize_pids(mode)
        self._planner = DynamicWindowPlanner()

    def _initialize_pids(self, mode: str) -> tuple[PID, TwoDPID]:
        if mode == "rsim":
            return get_rsim_pids()
        elif mode == "grsim":
            return get_grsim_pids()
        elif mode == "real":
            return get_real_pids()
        else:
            raise ValueError(
                f"Unknown mode: {mode}. Choose from 'rsim', 'grsim', or 'real'."
            )

    def path_to(
        self,
        game: Game,
        robot_id: int,
        target: Tuple[float, float],
        temporary_obstacles: List[LineString] = [],
    ) -> Tuple[Tuple[float, float], float]:
        return self._planner.path_to(game, robot_id, target, temporary_obstacles)

    @property
    def pid_oren(self) -> PID:
        return self._pid_oren

    @property
    def pid_trans(self) -> TwoDPID:
        return self._pid_trans
