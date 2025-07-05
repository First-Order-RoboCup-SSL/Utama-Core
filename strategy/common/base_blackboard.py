import py_trees
from strategy.common.tactics import Tactic
from team_controller.src.controllers import AbstractRobotController
from motion_planning.src.motion_controller import MotionController
from entities.game import Game
from entities.data.command import RobotCommand
from rsoccer_simulator.src.ssl.ssl_gym_base import SSLBaseEnv
from typing import Dict, Union
from strategy.common import Role


class BaseBlackboard(py_trees.blackboard.Client):
    @property
    def game(self) -> Game:
        return self.get("game")

    @property
    def robot_controller(self) -> AbstractRobotController:
        return self.get("robot_controller")

    @property
    def motion_controller(self) -> MotionController:
        return self.get("motion_controller")

    @property
    def rsim_env(self) -> SSLBaseEnv:
        return self.get("rsim_env")

    @property
    def cmd_map(self) -> Dict[int, Union[None, RobotCommand]]:
        return self.get("cmd_map")

    @property
    def role_map(self) -> Dict[int, Role]:
        return self.get("role_map")
    
    @property
    def tactic(self) -> Tactic:
        return self.get("tactic")
