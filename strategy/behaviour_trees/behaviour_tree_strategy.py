import logging
import time
import py_trees
from entities.game.present_future_game import PresentFutureGame
from strategy.abstract_strategy import AbstractStrategy
from team_controller.src.controllers.common.robot_controller_abstract import (
    AbstractRobotController,
)

logger = logging.getLogger(__name__)


class BehaviourTreeStrategy(AbstractStrategy):
    def __init__(
        self,
        robot_controller: AbstractRobotController,
        behaviour: py_trees.behaviour.Behaviour,
    ):
        super().__init__(robot_controller)
        self.blackboard = py_trees.blackboard.Client(name="GlobalConfig")
        self.blackboard.register_key(
            key="robot_controller", access=py_trees.common.Access.EXCLUSIVE_WRITE
        )
        self.blackboard.register_key(
            key="present_future_game", access=py_trees.common.Access.EXCLUSIVE_WRITE
        )
        self.blackboard.robot_controller = robot_controller
        self.behaviour_tree = py_trees.trees.BehaviourTree(behaviour)

    def step(self, present_future_game: PresentFutureGame):
        start_time = time.time()
        self.blackboard.present_future_game = present_future_game
        self.behaviour_tree.tick_once()
        end_time = time.time()
        logger.info(
            "Behaviour Tree %s executed in %f secs",
            self.behaviour_tree.__class__.__name__,
            end_time - start_time,
        )
        self.robot_controller.send_robot_commands()
