import logging
from abc import ABC, abstractmethod
from typing import Optional, cast

import py_trees
import pydot
from py_trees import utilities as py_trees_utilities

from utama_core.config.roles import Role
from utama_core.config.settings import BLACKBOARD_NAMESPACE_MAP, RENDER_BASE_PATH
from utama_core.entities.data.command import RobotCommand
from utama_core.entities.game import Game
from utama_core.motion_planning.src.common.motion_controller import MotionController
from utama_core.rsoccer_simulator.src.ssl.ssl_gym_base import SSLBaseEnv
from utama_core.skills.src.utils.move_utils import empty_command
from utama_core.strategy.common.base_blackboard import BaseBlackboard
from utama_core.team_controller.src.controllers.common.robot_controller_abstract import (
    AbstractRobotController,
)

logger = logging.getLogger(__name__)


def _prune_base_blackboard_elements(graph: pydot.Dot) -> None:
    """Strip BaseBlackboard artifacts from the rendered DOT graph."""
    prunable_names = BaseBlackboard.base_keys() | BaseBlackboard.base_client_names()
    if not prunable_names:
        return

    name_cache: dict[str, str] = {}

    def should_prune(name: str) -> bool:
        if name in name_cache:
            normalised = name_cache[name]
        else:
            normalised = name.strip('"')
            if normalised.startswith("/"):
                normalised = normalised.rsplit("/", 1)[-1]
            name_cache[name] = normalised
        return normalised in prunable_names

    for edge in list(graph.get_edges()):
        if should_prune(edge.get_source()) or should_prune(edge.get_destination()):
            graph.del_edge(edge.get_source(), edge.get_destination())

    def prune_nodes(container: pydot.Dot) -> None:
        for node in list(container.get_nodes()):
            if should_prune(node.get_name()):
                container.del_node(node)
        for subgraph in container.get_subgraphs():
            prune_nodes(subgraph)

    prune_nodes(graph)


class AbstractStrategy(ABC):
    """
    Base class for team strategies backed by behaviour trees.
    """

    def __init__(self):
        self.behaviour_tree = py_trees.trees.BehaviourTree(self.create_behaviour_tree())

        ### These attributes are set by the StrategyRunner before the strategy is run. ###
        self.robot_controller: AbstractRobotController = None
        self.blackboard: BaseBlackboard = None

    ### START OF FUNCTIONS TO BE IMPLEMENTED BY YOUR STRATEGY ###

    @abstractmethod
    def create_behaviour_tree(self) -> py_trees.behaviour.Behaviour:
        """
        Create and return the root behaviour for this strategy.

        The tree should be structured in two conceptual phases:
        1. Game analysis and role assignment (populate `blackboard.role_map`).
        2. Tactical execution (set per-robot commands in `blackboard.cmd_map`).
        """
        ...

    @abstractmethod
    def assert_exp_robots(self, n_runtime_friendly: int, n_runtime_enemy: int):
        """
        Validate the number of robots for which the strategy is designed.

        Called once on initial run. Implementations can assert constraints on
        the number of friendly and enemy robots that the strategy expects.
        By default an external guard ensures 1 <= robots <= 6, so only add
        additional checks if needed.

        Args:
            n_runtime_friendly: Number of friendly robots in the match.
            n_runtime_enemy: Number of opponent robots in the match.
        """
        ...

    def execute_default_action(self, game: Game, role: Role, robot_id: int) -> RobotCommand:
        """
        Provide a fallback command for robots without assignments.

        Invoked after the tree tick for any robot that did not receive a
        command via `blackboard.cmd_map`. Override to implement a safer or
        more appropriate default behaviour.

        Args:
            game: Current game state snapshot.
            role: Role assigned to the robot, or `Role.UNASSIGNED`.
            robot_id: Identifier of the robot.

        Returns:
            A `RobotCommand` to send to the controller.
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
        self.blackboard.register_key(key="motion_controller", access=py_trees.common.Access.READ)

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

        motion_controller: MotionController = getattr(self.blackboard, "motion_controller", None)
        if motion_controller is not None:
            motion_controller.update_game(game)

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
        blackboard.rsim_env = None  # set to None by default
        blackboard.register_key(key="motion_controller", access=py_trees.common.Access.WRITE)

        blackboard: BaseBlackboard = cast(BaseBlackboard, blackboard)
        return blackboard

    def render(
        self,
        name: Optional[str] = None,
        visibility_level: py_trees.common.VisibilityLevel = py_trees.common.VisibilityLevel.DETAIL,
        with_blackboard_variables: bool = True,
        with_qualified_names: bool = False,
    ):
        """
        Renders a dot, png, and svg file of the behaviour tree in the directory specified by `RENDER_BASE_PATH`.
        - `name` (str, optional): The name of the output files. If None, uses the class name.
        - `visibility_level` (py_trees.common.VisibilityLevel): The visibility level for the rendering. Default is DETAIL.
        - `with_blackboard_variables` (bool): Whether to include blackboard variables in the rendering. Default is True.
        - `with_qualified_names` (bool): Whether to use qualified names in the rendering. Default is False.
        """
        RENDER_BASE_PATH.mkdir(parents=True, exist_ok=True)
        name = self.__class__.__name__ if name is None else name

        graph = py_trees.display.dot_tree(
            root=self.behaviour_tree.root,
            visibility_level=visibility_level,
            with_blackboard_variables=with_blackboard_variables,
            with_qualified_names=with_qualified_names,
        )

        if with_blackboard_variables:
            _prune_base_blackboard_elements(graph)

        filename_wo_extension = py_trees_utilities.get_valid_filename(name)

        for extension, writer in {
            "dot": graph.write_dot,
            "png": graph.write_png,
            "svg": graph.write_svg,
        }.items():
            output_path = RENDER_BASE_PATH / f"{filename_wo_extension}.{extension}"
            try:
                writer(output_path.as_posix())
            except (AssertionError, OSError, FileNotFoundError):
                logger.warning("skipping %s export; Graphviz 'dot' executable not available", extension)
