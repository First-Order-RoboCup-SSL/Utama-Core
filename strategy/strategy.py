from abc import abstractmethod, ABC
from typing import Dict, List

from entities.data.command import RobotCommand
from entities.game.present_future_game import PresentFutureGame
from team_controller.src.controllers.common.robot_controller_abstract import AbstractRobotController

class Strategy(ABC):

    def __init__(self, robot_controller: AbstractRobotController):
        self.robot_controller = robot_controller

    @abstractmethod
    def step(self, present_future_game: PresentFutureGame):
        ...

