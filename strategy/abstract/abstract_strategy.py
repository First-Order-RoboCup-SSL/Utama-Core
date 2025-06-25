from abc import abstractmethod, ABC
from rsoccer_simulator.src.ssl.ssl_gym_base import SSLBaseEnv
import logging
import time
import py_trees
from entities.game.present_future_game import PresentFutureGame
from motion_planning.src.pid.pid import PID, TwoDPID
from team_controller.src.controllers.common.robot_controller_abstract import (
    AbstractRobotController,
)
from strategy.abstract.abstract_behaviour import AbstractBehaviour

logger = logging.getLogger(__name__)


class AbstractStrategy(ABC):
    def __init__(
        self,
    ):
        self.robot_controller = None  # Will be set by StrategyRunner
        self.blackboard = self._setup_blackboard()
        self.behaviour_tree = py_trees.trees.BehaviourTree(self.create_behaviour())

    ### START OF FUNCTIONS TO BE IMPLEMENTED BY YOUR STRATEGY ###

    @abstractmethod
    def create_behaviour(self): ...

    @abstractmethod
    def assert_exp_robots(self, n_runtime_friendly: int, n_runtime_enemy: int):
        """
        Called on initial run to make sure that the expected robots on runtime.
        match the possible robots in this strategy.
        By default, 1 <= n_robots <= 6 is already asserted, so does not need to be checked here.
        """
        ...

    ### END OF STRATEGY IMPLEMENTATION ###

    def load_rsim_env(self, env: SSLBaseEnv):
        """
        Called by StrategyRunner: Load the RSim environment into the blackboard.
        """
        self.blackboard.set(name="rsim_env", value=env)

    def load_robot_controller(self, robot_controller: AbstractRobotController):
        """
        Called by StrategyRunner: Load the robot controller into the blackboard.
        """
        self.robot_controller = robot_controller
        self.blackboard.set(name="robot_controller", value=robot_controller)

    def load_pids(self, pid_oren: PID, pid_trans: TwoDPID):
        """
        Called by StrategyRunner: Load the PIDs for orientation and translation control into the blackboard.
        """
        self.blackboard.set(name="pid_oren", value=pid_oren)
        self.blackboard.set(name="pid_trans", value=pid_trans)

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

    def _setup_blackboard(self):
        """Sets up the blackboard with the necessary keys for the strategy."""
        blackboard = py_trees.blackboard.Client(name="GlobalConfig")
        blackboard.register_key(
            key="robot_controller", access=py_trees.common.Access.WRITE
        )
        blackboard.register_key(
            key="present_future_game", access=py_trees.common.Access.WRITE
        )
        blackboard.register_key(key="pid_oren", access=py_trees.common.Access.WRITE)
        blackboard.register_key(key="pid_trans", access=py_trees.common.Access.WRITE)
        blackboard.register_key(key="rsim_env", access=py_trees.common.Access.WRITE)
        return blackboard
