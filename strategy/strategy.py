from abc import abstractmethod, ABC
from enum import Enum
from rsoccer_simulator.src.ssl.ssl_gym_base import SSLBaseEnv

from entities.data.command import RobotCommand
from entities.game.present_future_game import PresentFutureGame
from team_controller.src.controllers.common.robot_controller_abstract import (
    AbstractRobotController,
)
from motion_planning.src.pid.pid import PID, TwoDPID, get_grsim_pids


# output from step function to report progress of strategy
class StrategyStatus(Enum):
    TERMINAL_END = 0
    ITERATION_END = 1
    RUNNING = 2


class Strategy(ABC):

    def __init__(self):
        self.robot_controller: AbstractRobotController = None
        self.env: SSLBaseEnv = None
        self.pid_oren: PID = None
        self.pid_trans: TwoDPID = None

    def load_rsim_env(self, env: SSLBaseEnv):
        self.env = env

    def load_robot_controller(self, robot_controller: AbstractRobotController):
        self.robot_controller = robot_controller

    def load_pids(self, pid_oren: PID, pid_trans: TwoDPID):
        self.pid_oren = pid_oren
        self.pid_trans = pid_trans

    @abstractmethod
    def step(self, present_future_game: PresentFutureGame) -> StrategyStatus: ...

    @abstractmethod
    def assert_exp_robots(self, n_runtime_friendly: int, n_runtime_enemy: int):
        """
        Called on initial run to make sure that the expected robots on runtime
        match the possible robots in this strategy
        """
        ...
