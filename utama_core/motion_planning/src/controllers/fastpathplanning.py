from abc import ABC, abstractmethod

from utama_core.config.enums import Mode
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game
from utama_core.rsoccer_simulator.src.ssl.envs import SSLStandardEnv
from utama_core.motion_planning.src.common.motion_controller import MotionController
from utama_core.motion_planning.src.fastpathplanning.planner import FastPathPlanner
from utama_core.motion_planning.src.pid.pid import get_pids

class FastPathPlanningController(MotionController):
    def __init__(self, mode: Mode, rsim_env: SSLStandardEnv | None = None):
        self.mode = mode
        self.rsim_env: SSLStandardEnv | None = rsim_env
        self.pid_oren, self.pid_trans = get_pids(mode)
        self.fpp = FastPathPlanner(env = self.rsim_env)
       
    def calculate(
        self,
        game: Game,
        robot_id: int,
        target_pos: Vector2D,
        target_oren: float,
    ) -> tuple[Vector2D, float]:
        robot = game.friendly_robots[robot_id]
 
        
        oren = self.pid_oren.calculate(target_oren, robot.orientation, robot_id)
        
        pos = self.fpp._path_to(game, robot_id, target_pos)[0][1]
 
        vel  = self.pid_trans.calculate(pos, robot.p, robot_id)
        

        return vel, oren
    def reset(self, robot_id):
        self.pid_oren.reset(robot_id)

