from rsoccer_simulator.src.ssl.ssl_gym_base import SSLBaseEnv
from entities.game import PresentFutureGame, Robot, Game
from entities.data.command import RobotCommand
from motion_planning.src.pid.pid import PID, TwoDPID
from team_controller.src.controllers.common.robot_controller_abstract import (
    AbstractRobotController,
)
from strategy.common.roles import Role
from abc import abstractmethod, ABC
from typing import Dict, Union
import logging
import time
import py_trees

logger = logging.getLogger(__name__)


class AbstractStrategy(ABC):
    def __init__(
        self,
    ):
        self.robot_controller = None  # Will be set by StrategyRunner
        self.blackboard = self._setup_blackboard()
        self.behaviour_tree = py_trees.trees.BehaviourTree(self.create_behaviour_tree())

    ### START OF FUNCTIONS TO BE IMPLEMENTED BY YOUR STRATEGY ###

    @abstractmethod
    def create_behaviour_tree(self):
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
        pass

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
        self.blackboard.set(
            name="cmd_map",
            value=self._reset_cmd_map(present_future_game.current.friendly_robots),
        )
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
                cmd = self.execute_default_action(
                    present_future_game.current, role, robot_id
                )
                self.robot_controller.add_robot_commands(cmd, robot_id)
        self.robot_controller.send_robot_commands()

        end_time = time.time()
        logger.info(
            "Behaviour Tree %s executed in %f secs",
            self.behaviour_tree.__class__.__name__,
            end_time - start_time,
        )

    def _reset_cmd_map(
        self, friendly_robots: Dict[int, Robot]
    ) -> Dict[int, Union[None, RobotCommand]]:
        """Resets the command map to be set in the blackboard."""
        if not hasattr(self, "_cmd_map_cache"):
            self._cmd_map_cache = {k: None for k in friendly_robots}
        return self._cmd_map_cache.copy()

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
        blackboard.register_key(key="cmd_map", access=py_trees.common.Access.WRITE)
        blackboard.register_key(key="role_map", access=py_trees.common.Access.WRITE)

        # initialize role_map
        blackboard.set(name="role_map", value={})
        return blackboard
