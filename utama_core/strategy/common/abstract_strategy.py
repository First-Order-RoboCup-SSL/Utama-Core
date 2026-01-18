import logging
from abc import ABC, abstractmethod
from typing import Optional, cast

import py_trees
import pydot
from py_trees import utilities as py_trees_utilities

from utama_core.config.enums import Role
from utama_core.config.settings import BLACKBOARD_NAMESPACE_MAP, RENDER_BASE_PATH
from utama_core.entities.data.command import RobotCommand
from utama_core.entities.game import Game
from utama_core.entities.game.field import Field, FieldBounds
from utama_core.global_utils.math_utils import assert_valid_bounding_box
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
    def assert_exp_robots(self, n_runtime_friendly: int, n_runtime_enemy: int) -> bool:
        """
        Validate that the number of friendly and enemy robots matches the strategy's expectations.

        This method is called once during initialization. Implementations can enforce
        specific constraints on the number of robots the strategy supports.
        An external guard already ensures that 1 ≤ robots ≤ 6, so only apply
        additional checks if needed.

        Args:
            n_runtime_friendly: Number of friendly robots available during the match.
            n_runtime_enemy: Number of opponent robots available during the match.

        Returns:
            bool: True if the robot counts are as expected, False otherwise.
        """
        ...

    @abstractmethod
    def assert_exp_goals(self, includes_my_goal_line: bool, includes_opp_goal_line: bool) -> bool:
        """
        Validate that the field configuration includes the expected goals.

        Implementations should verify that the strategy can operate correctly
        given whether our own and the opponent’s goal lines are present.

        Args:
            includes_my_goal_line: True if the field includes our own goal line.
            includes_opp_goal_line: True if the field includes the opponent’s goal line.

        Returns:
            bool: True if the field configuration matches the strategy’s expectations, False otherwise.
        """
        ...

    @abstractmethod
    def get_min_bounding_zone(self) -> Optional[FieldBounds]:
        """
        Return the minimum field region required by the strategy.

        If the strategy only operates within a subset of the full field, return
        a `FieldBounds` object defining that region. Otherwise, return `None`
        to indicate no restriction.

        This method is called during `load_game()`, when the blackboard is already initialized,
        so `game` is available.

        Note:
            The bounding zone should be defined in field coordinates (i.e., absolute positions).

        Returns:
            Optional[FieldBounds]: A `FieldBounds` specifying the minimum bounding region, or `None`.
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
        return empty_command(False)

    ### END OF STRATEGY IMPLEMENTATION ###

    def setup_behaviour_tree(self, is_opp_strat: bool):
        """
        Must be called before strategy can be run.

        Setups the tree and blackboard based on if is_opp_strat
        """
        self.blackboard = self._setup_blackboard(is_opp_strat)
        self.behaviour_tree.setup(is_opp_strat=is_opp_strat)

    def load_rsim_env(self, env: SSLBaseEnv):
        """
        Called by StrategyRunner: Load the RSim environment into the blackboard.
        """
        self.blackboard.set("rsim_env", env, overwrite=True)
        self.blackboard.register_key(key="rsim_env", access=py_trees.common.Access.READ)

    def load_robot_controller(self, robot_controller: AbstractRobotController):
        """
        Called by StrategyRunner: Load the robot controller into the class.
        """
        self.robot_controller = robot_controller

    def load_motion_controller(self, motion_controller: MotionController):
        """
        Called by StrategyRunner: Load the Motion Controller into the blackboard.
        """
        self.blackboard.set("motion_controller", motion_controller, overwrite=True)
        self.blackboard.register_key(key="motion_controller", access=py_trees.common.Access.READ)

    def assert_field_requirements(self):
        """
        Assert that the actual field size meets the strategy's requirements,
        that both actual field and min_bounding_zone are within the full field,
        and that bounding boxes are well-formed (top-left above/left of bottom-right).
        """
        actual_field_size = self.blackboard.game.field.field_bounds
        min_bounding_zone = self.get_min_bounding_zone()

        # --- Validate min bounding zone ---
        if min_bounding_zone is not None:
            assert_valid_bounding_box(min_bounding_zone)

            # --- Check that actual field contains min_bounding_zone ---
            ax0, ay0 = actual_field_size.top_left
            ax1, ay1 = actual_field_size.bottom_right
            mx0, my0 = min_bounding_zone.top_left
            mx1, my1 = min_bounding_zone.bottom_right

            assert ax0 <= mx0, f"Field top-left x {ax0} smaller than required {mx0}"
            assert ay0 >= my0, f"Field top-left y {ay0} smaller than required {my0}"
            assert ax1 >= mx1, f"Field bottom-right x {ax1} smaller than required {mx1}"
            assert ay1 <= my1, f"Field bottom-right y {ay1} smaller than required {my1}"

    def load_game(self, game: Game):
        """
        Called by StrategyRunner: Load the game object into the blackboard.

        We do not set to READ after, as we TestManager may reset the game object for the new episode.
        """
        self.blackboard.set("game", game, overwrite=True)
        self.assert_field_requirements()

    def step(self):
        # start_time = time.time()
        game = self.blackboard.game
        
        # Dict[int, Union[None, RobotCommand]]. Initialise empty Dict.
        self.blackboard.cmd_map = {robot_id: None for robot_id in game.friendly_robots}

        self.behaviour_tree.tick()

        for robot_id, values in self.blackboard.cmd_map.items():
            if values is not None:
                self.robot_controller.add_robot_commands(values, robot_id)

            # if the robot is not assigned a command, execute the default action
            else:
                # Dict[int, Role]
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
                logger.warning(
                    "skipping %s export; Graphviz 'dot' executable not available",
                    extension,
                )
