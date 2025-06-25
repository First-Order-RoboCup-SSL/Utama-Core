import py_trees
from team_controller.src.controllers import AbstractRobotController
from motion_planning.src.pid import PID, TwoDPID
from entities.game import PresentFutureGame
from rsoccer_simulator.src.ssl.ssl_gym_base import SSLBaseEnv


class BaseBlackboard(py_trees.blackboard.Client):
    @property
    def present_future_game(self) -> PresentFutureGame:
        return self.get("present_future_game")

    @property
    def robot_controller(self) -> AbstractRobotController:
        return self.get("robot_controller")

    @property
    def pid_oren(self) -> PID:
        return self.get("pid_oren")

    @property
    def pid_trans(self) -> TwoDPID:
        return self.get("pid_trans")

    @property
    def rsim_env(self) -> SSLBaseEnv:
        return self.get("rsim_env")
