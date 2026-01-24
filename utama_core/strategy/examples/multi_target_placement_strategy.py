"""Placement strategy that moves one robot through a list of targets."""

from typing import Optional

import py_trees
from py_trees.composites import Sequence

from utama_core.config.settings import TIMESTEP
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game.field import FieldBounds
from utama_core.global_utils.math_utils import compute_bounding_zone_from_points
from utama_core.skills.src.utils.move_utils import move
from utama_core.strategy.common.abstract_behaviour import AbstractBehaviour
from utama_core.strategy.common.abstract_strategy import AbstractStrategy
from utama_core.strategy.examples.utils import SetBlackboardVariable


class MultiTargetPlacementStep(AbstractBehaviour):
    """
    Command a single robot to visit a list of target coordinates in order.

    Args:
        rd_robot_id: Blackboard key for the robot ID.
        targets: List of (x, y) target coordinates to visit.
        target_orientation: Orientation to hold while moving (radians).
        reach_tolerance: Distance to consider a target reached (meters).
        loop_targets: If True, wrap back to the first target after the last.
    """

    def __init__(
        self,
        rd_robot_id: str,
        targets: list[tuple[float, float]],
        target_orientation: float = 0.0,
        reach_tolerance: float = 0.05,
        loop_targets: bool = False,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.robot_id_key = rd_robot_id
        self.targets = [Vector2D(*target) for target in targets]
        self.target_orientation = target_orientation
        self.reach_tolerance = reach_tolerance
        self.loop_targets = loop_targets
        self._target_index = 0

    def setup_(self):
        self.blackboard.register_key(key=self.robot_id_key, access=py_trees.common.Access.READ)

    def update(self) -> py_trees.common.Status:
        if not self.targets:
            return py_trees.common.Status.FAILURE

        game = self.blackboard.game
        rsim_env = self.blackboard.rsim_env
        robot_id = self.blackboard.get(self.robot_id_key)

        if not game.friendly_robots or robot_id not in game.friendly_robots:
            return py_trees.common.Status.RUNNING

        current = game.friendly_robots[robot_id].p
        target = self.targets[self._target_index]

        if current.distance_to(target) <= self.reach_tolerance:
            if self._target_index + 1 < len(self.targets):
                self._target_index += 1
            elif self.loop_targets:
                self._target_index = 0
            target = self.targets[self._target_index]

        if rsim_env:
            rsim_env.draw_point(target.x, target.y, color="red")
            v = game.friendly_robots[robot_id].v
            p = game.friendly_robots[robot_id].p
            rsim_env.draw_point(p.x + v.x * TIMESTEP * 5, p.y + v.y * TIMESTEP * 5, color="green")

        cmd = move(
            game,
            self.blackboard.motion_controller,
            robot_id,
            target,
            self.target_orientation,
        )
        self.blackboard.cmd_map[robot_id] = cmd
        return py_trees.common.Status.RUNNING


class MultiTargetPlacementStrategy(AbstractStrategy):
    """
    Strategy that drives a single robot through a list of target positions.

    Args:
        robot_id: The ID of the robot to control.
        targets: List of (x, y) target coordinates to visit in order.
        target_orientation: Orientation to hold while moving (radians).
        reach_tolerance: Distance to consider a target reached (meters).
        loop_targets: If True, wrap back to the first target after the last.
    """

    def __init__(
        self,
        robot_id: int,
        targets: list[tuple[float, float]],
        target_orientation: float = 0.0,
        reach_tolerance: float = 0.05,
        loop_targets: bool = False,
    ):
        self.robot_id = robot_id
        self.targets = targets
        self.target_orientation = target_orientation
        self.reach_tolerance = reach_tolerance
        self.loop_targets = loop_targets
        self.robot_id_key = "target_robot_id"
        super().__init__()

    def assert_exp_robots(self, n_runtime_friendly: int, n_runtime_enemy: int):
        return n_runtime_friendly >= 1

    def assert_exp_goals(self, includes_my_goal_line: bool, includes_opp_goal_line: bool):
        return True

    def get_min_bounding_zone(self) -> Optional[FieldBounds]:
        if not self.targets:
            return None
        return compute_bounding_zone_from_points(self.targets)

    def create_behaviour_tree(self) -> py_trees.behaviour.Behaviour:
        """Factory function to create a behaviour tree for multi-target placement."""
        coach_root = Sequence(name="CoachRoot", memory=True)

        set_robot_id = SetBlackboardVariable(
            name="SetTargetRobotID",
            variable_name=self.robot_id_key,
            value=self.robot_id,
        )

        move_targets = MultiTargetPlacementStep(
            rd_robot_id=self.robot_id_key,
            targets=self.targets,
            target_orientation=self.target_orientation,
            reach_tolerance=self.reach_tolerance,
            loop_targets=self.loop_targets,
        )

        coach_root.add_children([set_robot_id, move_targets])
        return coach_root
