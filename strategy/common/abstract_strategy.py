from rsoccer_simulator.src.ssl.ssl_gym_base import SSLBaseEnv
from entities.game import Game
from entities.data.command import RobotCommand
from motion_planning.src.motion_controller import MotionController
from team_controller.src.controllers.common.robot_controller_abstract import (
    AbstractRobotController,
)
from config.settings import BLACKBOARD_NAMESPACE_MAP
from skills.src.utils.move_utils import empty_command
from config.roles import Role
from abc import abstractmethod, ABC
import logging
import py_trees
from strategy.common.base_blackboard import BaseBlackboard
from typing import cast

logger = logging.getLogger(__name__)


class AbstractStrategy(ABC):
    def __init__(self):
        self.behaviour_tree = py_trees.trees.BehaviourTree(self.create_behaviour_tree())

        ### These attributes are set by the StrategyRunner before the strategy is run. ###
        self.robot_controller: AbstractRobotController = None
        self.blackboard: BaseBlackboard = None

    ### START OF FUNCTIONS TO BE IMPLEMENTED BY YOUR STRATEGY ###

    @abstractmethod
    def create_behaviour_tree(self) -> py_trees.behaviour.Behaviour:
        """
        Create the behaviour tree for this strategy.
        This method should return a py_trees tree that defines the behaviour of the strategy.
        Behaviour trees must follow the following structure:
        1. **Game Analysis and Role Assignment**: Analyse game state and assign roles via `role_map`.
        2. **Tactical Execution**: Execute tactic and set commands for robots involved in the play via `cmd_map`.
        """
        ...

    @abstractmethod
    def assert_exp_robots(self, n_runtime_friendly: int, n_runtime_enemy: int):
        """
        Called on initial run to make sure that the expected robots on runtime.
        match the possible robots in this strategy.
        By default, 1 <= n_robots <= 6 is already asserted, so does not need to be checked here.
        """
        ...

    def execute_default_action(
        self, game: Game, role: Role, robot_id: int
    ) -> RobotCommand:
        """
        Called on each unassigned robot to execute the default action.
        This is used when no specific command is set in the blackboard after the coach tree for this robot.
        """
        return empty_command(True)

    ### END OF STRATEGY IMPLEMENTATION ###

    def load_rsim_env(self, env: SSLBaseEnv):
        """
        Called by StrategyRunner: Load the RSim environment into the blackboard.
        """
        self.blackboard.set("rsim_env", env, overwrite=True)
        self.blackboard.register_key(key="rsim_env", access=py_trees.common.Access.READ)

    def load_robot_controller(self, robot_controller: AbstractRobotController):
        """
        Called by StrategyRunner: Load the robot controller into the blackboard.
        """
        self.robot_controller = robot_controller

    def load_motion_controller(self, motion_controller: MotionController):
        """
        Called by StrategyRunner: Load the Motion Controller into the blackboard.
        """
        self.blackboard.set("motion_controller", motion_controller, overwrite=True)
        self.blackboard.register_key(
            key="motion_controller", access=py_trees.common.Access.READ
        )

    def setup_behaviour_tree(self, is_opp_strat: bool):
        """
        Must be called before strategy can be run.
        Setups the tree and blackboard based on if is_opp_strat
        """
        self.blackboard = self._setup_blackboard(is_opp_strat)
        self.behaviour_tree.setup(is_opp_strat=is_opp_strat)

    def step(self, game: Game):
        # start_time = time.time()
        self.blackboard.game = game
        self.blackboard.cmd_map = {robot_id: None for robot_id in game.friendly_robots}

        self.behaviour_tree.tick()

        for robot_id, values in self.blackboard.cmd_map.items():
            if values is not None:
                self.robot_controller.add_robot_commands(values, robot_id)

            # if the robot is not assigned a command, execute the default action
            else:
                if robot_id not in self.blackboard.role_map:
                    role = Role.UNASSIGNED
                else:
                    role = self.blackboard.role_map[robot_id]
                cmd = self.execute_default_action(game, role, robot_id)
                self.robot_controller.add_robot_commands(cmd, robot_id)
        self.robot_controller.send_robot_commands()

        # end_time = time.time()
        # logger.info(
        #     "Behaviour Tree %s executed in %f secs",
        #     self.behaviour_tree.__class__.__name__,
        #     end_time - start_time,
        # )

    def _setup_blackboard(self, is_opp_strat: bool) -> BaseBlackboard:
        """Sets up the blackboard with the necessary keys for the strategy."""

        blackboard = py_trees.blackboard.Client(
            name="GlobalBlackboard", namespace=BLACKBOARD_NAMESPACE_MAP[is_opp_strat]
        )
        blackboard.register_key(key="game", access=py_trees.common.Access.WRITE)
        blackboard.register_key(key="cmd_map", access=py_trees.common.Access.WRITE)

        blackboard.register_key(key="role_map", access=py_trees.common.Access.WRITE)
        blackboard.register_key(key="tactic", access=py_trees.common.Access.WRITE)
        blackboard.role_map = {}

        blackboard.register_key(key="rsim_env", access=py_trees.common.Access.WRITE)
        blackboard.register_key(
            key="motion_controller", access=py_trees.common.Access.WRITE
        )

        blackboard: BaseBlackboard = cast(BaseBlackboard, blackboard)
        return blackboard
