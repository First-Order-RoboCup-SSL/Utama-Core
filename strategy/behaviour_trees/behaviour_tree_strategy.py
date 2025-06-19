import logging
import time
import py_trees
from entities.game.present_future_game import PresentFutureGame
from motion_planning.src.pid.pid import PID, TwoDPID
from strategy.abstract_strategy import AbstractStrategy
from team_controller.src.controllers.common.robot_controller_abstract import (
    AbstractRobotController,
)

logger = logging.getLogger(__name__)


class BehaviourTreeStrategy(AbstractStrategy):
    def __init__(
        self,
        behaviour: py_trees.behaviour.Behaviour,
    ):
        super().__init__()
        self.blackboard = py_trees.blackboard.Client(name="GlobalConfig")
        self.blackboard.register_key(
            key="robot_controller", access=py_trees.common.Access.WRITE
        )
        self.blackboard.register_key(
            key="present_future_game", access=py_trees.common.Access.WRITE
        )
        self.blackboard.register_key(
            key="pid_oren", access=py_trees.common.Access.WRITE
        )
        self.blackboard.register_key(
            key="pid_trans", access=py_trees.common.Access.WRITE
        )

        self.behaviour_tree = py_trees.trees.BehaviourTree(behaviour)

    def load_robot_controller(self, robot_controller: AbstractRobotController):
        """Overrides the parent method to update the blackboard when the robot controller is loaded."""
        # Update the blackboard with the robot controller provided by the StrategyRunner
        self.robot_controller = robot_controller
        self.blackboard.set(name="robot_controller", value=robot_controller)
    
    def load_pids(self, pid_oren: PID, pid_trans: TwoDPID):
        """Overrides the parent method to update the blackboard when PIDs are loaded."""
        # Update the blackboard with the PIDs provided by the StrategyRunner
        self.blackboard.set(name="pid_oren", value=pid_oren)
        self.blackboard.set(name="pid_trans", value=pid_trans)

    def assert_exp_robots(self, n_runtime_friendly: int, n_runtime_enemy: int):
        if 1 <= n_runtime_friendly <= 3 and 1 <= n_runtime_enemy <= 3:
            return True
        return False
    
    def step(self, present_future_game: PresentFutureGame):
        start_time = time.time()
        self.blackboard.set(name="present_future_game", value=present_future_game)
        self.behaviour_tree.tick()
        
        end_time = time.time()
        logger.info(
            "Behaviour Tree %s executed in %f secs",
            self.behaviour_tree.__class__.__name__,
            end_time - start_time,
        )
        
        self.robot_controller.send_robot_commands()