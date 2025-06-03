import py_trees

from entities.game.present_future_game import PresentFutureGame
from motion_planning.src.pid.pid import PID, TwoDPID
from team_controller.src.controllers.common.robot_controller_abstract import (
    AbstractRobotController,
)


class RobocupBehaviour(py_trees.behaviour.Behaviour):
    def __init__(self, name: str):
        super().__init__(name=self.__class__.__name__)
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(
            key="robot_controller", access=py_trees.common.Access.READ
        )
        self.blackboard.register_key(
            key="present_future_game", access=py_trees.common.Access.READ
        )
        self.blackboard.register_key(
            key="pid_trans", access=py_trees.common.Access.READ
        )
        self.blackboard.register_key(key="pid_oren", access=py_trees.common.Access.READ)

    # @property
    # def present_future_game(self) -> PresentFutureGame:
    #     return self.blackboard.present_future_game

    # @property
    # def robot_controller(self) -> AbstractRobotController:
    #     return self.blackboard.robot_controller

    # @property
    # def pid_trans(self) -> TwoDPID:
    #     return self.blackboard.pid_trans

    # @property
    # def pid_oren(self) -> PID:
    #     return self.blackboard.pid_oren
