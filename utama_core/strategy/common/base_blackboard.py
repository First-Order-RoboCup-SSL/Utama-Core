from typing import Dict, Union

import py_trees

from utama_core.config.enums import Role, Tactic
from utama_core.entities.data.command import RobotCommand
from utama_core.entities.game import Game
from utama_core.motion_planning.src.common.motion_controller import MotionController
from utama_core.rsoccer_simulator.src.ssl.ssl_gym_base import SSLBaseEnv
from utama_core.team_controller.src.controllers import AbstractRobotController


class BaseBlackboard(py_trees.blackboard.Client):
    @classmethod
    def base_keys(cls) -> set[str]:
        """Return the canonical blackboard keys registered by the base client."""
        return {name for name, value in vars(cls).items() if isinstance(value, property)}

    @classmethod
    def base_client_names(cls) -> set[str]:
        """Return the default client names used by the base blackboard setup."""
        return {"GlobalBlackboard"}

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
