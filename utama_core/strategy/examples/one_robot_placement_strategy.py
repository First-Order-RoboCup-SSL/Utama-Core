import math

import numpy as np
import py_trees
from py_trees.composites import Sequence

from utama_core.config.settings import TIMESTEP
from utama_core.global_utils.math_utils import Vector2D
from utama_core.skills.src.utils.move_utils import move
from utama_core.strategy.common.abstract_behaviour import AbstractBehaviour
from utama_core.strategy.common.abstract_strategy import (
    AbstractStrategy,
    SpaceRequirements,
)
from utama_core.strategy.examples.utils import (
    CalculateFieldCenter,
    SetBlackboardVariable,
)

_ARRIVE_TOL = 0.15  # metres
_MARGIN = 0.2  # metres inset from field bounds edges for waypoints


class RobotPlacementStep(AbstractBehaviour):
    """
    Cycles a robot through a 3x3 grid of waypoints covering the field bounds,
    facing the ball at each step.

    Blackboard reads:
        - robot_id_key (int): robot to control
        - field_center_key (tuple): center of the active field bounds
    """

    def __init__(self, rd_robot_id: str, field_center_key: str = "FieldCenter"):
        super().__init__()
        self.field_center_key = field_center_key
        self.robot_id_key = rd_robot_id
        self._waypoints: list[Vector2D] = []
        self._wp_idx = 0

    def setup_(self):
        self.blackboard.register_key(key=self.robot_id_key, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=self.field_center_key, access=py_trees.common.Access.READ)

    def _build_waypoints(self) -> list[Vector2D]:
        """3x3 grid inset from the active field bounds, in a snake pattern."""
        bounds = self.blackboard.game.field.field_bounds
        x_min = bounds.top_left[0] + _MARGIN
        x_max = bounds.bottom_right[0] - _MARGIN
        y_min = bounds.bottom_right[1] + _MARGIN
        y_max = bounds.top_left[1] - _MARGIN

        xs = [x_min, (x_min + x_max) / 2, x_max]
        ys = [y_min, (y_min + y_max) / 2, y_max]

        # Snake order: alternate y direction per column to minimise travel
        points = []
        for i, x in enumerate(xs):
            col_ys = ys if i % 2 == 0 else reversed(ys)
            for y in col_ys:
                points.append(Vector2D(x, y))
        return points

    def update(self) -> py_trees.common.Status:
        game = self.blackboard.game
        rsim_env = self.blackboard.rsim_env
        robot_id: int = self.blackboard.get(self.robot_id_key)

        if not self._waypoints:
            try:
                self.blackboard.get(self.field_center_key)  # wait until center is ready
            except KeyError:
                return py_trees.common.Status.FAILURE
            self._waypoints = self._build_waypoints()

        target = self._waypoints[self._wp_idx]
        robot = game.friendly_robots[robot_id]
        ball = game.ball

        bx, by = ball.p.x, ball.p.y
        cx, cy = robot.p.x, robot.p.y
        oren = np.atan2(by - cy, bx - cx)

        cmd = move(game, self.blackboard.motion_controller, robot_id, target, oren)

        if rsim_env:
            rsim_env.draw_point(target.x, target.y, color="red")
            v = robot.v
            rsim_env.draw_point(cx + v.x * TIMESTEP * 5, cy + v.y * TIMESTEP * 5, color="green")

        if math.dist((cx, cy), (target.x, target.y)) < _ARRIVE_TOL:
            self._wp_idx = (self._wp_idx + 1) % len(self._waypoints)

        self.blackboard.cmd_map[robot_id] = cmd
        return py_trees.common.Status.RUNNING


class RobotPlacementStrategy(AbstractStrategy):
    def __init__(self, robot_id: int):
        """
        Initializes the RobotPlacementStrategy with a specific robot ID.
        Robot placement oscillates the specified robot between two points centered around the middle of the field bounds.

        :param robot_id: The ID of the robot this strategy will control.
        :param field_bounds: The bounds of the field to operate within.
        """
        self.robot_id = robot_id
        super().__init__()

    def assert_exp_robots(self, n_runtime_friendly: int, n_runtime_enemy: int):
        if n_runtime_friendly == 1 and n_runtime_enemy == 0:
            return True
        return False

    def assert_exp_goals(self, includes_my_goal_line: bool, includes_opp_goal_line: bool):
        return True  # No specific goal line requirements

    def get_min_bounding_req(self):
        return SpaceRequirements(min_length=1.0, min_width=1.0)  # Require at least a 1x1 area to allow for oscillation

    def create_behaviour_tree(self) -> py_trees.behaviour.Behaviour:
        """Factory function to create a complete behaviour tree."""

        robot_id_key = "target_robot_id"
        field_center_key = "FieldCenter"

        coach_root = Sequence(name="CoachRoot", memory=False)

        set_rbt_id = SetBlackboardVariable(
            name="SetTargetRobotID",
            variable_name=robot_id_key,
            value=self.robot_id,
        )

        ### Assemble the tree ###

        # Calculate Field Center from custom field_bounds
        calc_center = CalculateFieldCenter(output_key=field_center_key)

        coach_root.add_children(
            [
                set_rbt_id,
                calc_center,
                RobotPlacementStep(rd_robot_id=robot_id_key, field_center_key=field_center_key),
            ]
        )

        return coach_root
