import math
import random
from typing import Any, Optional

import numpy as np
import py_trees
from py_trees.composites import Sequence

from utama_core.config.settings import TIMESTEP
from utama_core.entities.game.field import Field, FieldBounds
from utama_core.global_utils.math_utils import Vector2D
from utama_core.skills.src.utils.move_utils import move
from utama_core.strategy.common.abstract_behaviour import AbstractBehaviour

# from robot_control.src.tests.utils import one_robot_placement
from utama_core.strategy.common.abstract_strategy import AbstractStrategy
from utama_core.strategy.common.field_behaviours import CalculateFieldCenter


class RobotPlacementStep(AbstractBehaviour):
    """
    A behaviour that commands a robot to move between two specific positions on the field.

    **Args:**
        invert (bool): Whether to invert the robot's movement direction.
    **Blackboard Interaction:**
        Reads:
            - `rd_robot_id` (int): The ID of the robot to check for ball possession. Typically from the `SetBlackboardVariable` node.

    **Returns:**
        - `py_trees.common.Status.RUNNING`: The behaviour is actively commanding the robot to move.
    """

    def __init__(self, rd_robot_id: str, field_center_key: str = "FieldCenter"):
        super().__init__()
        self.field_center_key = field_center_key
        self.robot_id_key = rd_robot_id
        self.initialized = False
        self.center_x = 0.0
        self.center_y = 0.0
        self.tx = 0.0
        self.ty = 0.0

    def setup_(self):
        self.blackboard.register_key(key=self.robot_id_key, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=self.field_center_key, access=py_trees.common.Access.READ)

    def update(self) -> py_trees.common.Status:
        """Closure which advances the simulation by one step."""

        # Initialize targets if not ready
        if not self.initialized:
            try:
                center = self.blackboard.get(self.field_center_key)
                if center:
                    self.center_x, self.center_y = center
                    self.tx = self.center_x
                    self.ty = self.center_y + 0.5
                    self.initialized = True
            except KeyError:
                # Center not yet available
                return py_trees.common.Status.FAILURE

        if not self.initialized:
            return py_trees.common.Status.FAILURE

        game = self.blackboard.game
        rsim_env = self.blackboard.rsim_env
        id: int = self.blackboard.get(self.robot_id_key)

        friendly_robots = game.friendly_robots
        bx, by = game.ball.p.x, game.ball.p.y
        rp = friendly_robots[id].p
        cx, cy = rp.x, rp.y
        error = math.dist((self.tx, self.ty), (cx, cy))

        if game.friendly_robots and game.ball is not None:
            friendly_robots = game.friendly_robots
            bx, by = game.ball.p.x, game.ball.p.y
            rp = friendly_robots[id].p
            cx, cy, _ = rp.x, rp.y, friendly_robots[id].orientation
            error = math.dist((self.tx, self.ty), (cx, cy))

            # Ensure target x is always the center x
            self.tx = self.center_x

            switch = error < 0.1
            if switch:
                upper_target = self.center_y + 0.5
                lower_target = self.center_y - 0.5

                if math.isclose(self.ty, lower_target, abs_tol=0.1):
                    self.ty = upper_target
                else:
                    self.ty = lower_target

            # changed so the robot tracks the ball while moving
            oren = np.atan2(by - cy, bx - cx)
            cmd = move(
                game,
                self.blackboard.motion_controller,
                id,
                Vector2D(self.tx, self.ty),
                oren,
            )
            if rsim_env:
                rsim_env.draw_point(self.tx, self.ty, color="red")
                v = game.friendly_robots[id].v
                p = game.friendly_robots[id].p
                rsim_env.draw_point(p.x + v.x * TIMESTEP * 5, p.y + v.y * TIMESTEP * 5, color="green")

        self.blackboard.cmd_map[id] = cmd
        return py_trees.common.Status.RUNNING


class SetBlackboardVariable(AbstractBehaviour):
    """A generic behaviour to set a variable on the blackboard."""

    def __init__(self, name: str, variable_name: str, value: Any):
        super().__init__(name=name)
        self.variable_name = variable_name
        self.value = value

    def setup_(self):
        self.blackboard.register_key(key=self.variable_name, access=py_trees.common.Access.WRITE)

    def update(self) -> py_trees.common.Status:
        # print(f"Setting {self.variable_name} to {self.value} on the blackboard.")
        self.blackboard.set(self.variable_name, self.value, overwrite=True)
        return py_trees.common.Status.SUCCESS


class RobotPlacementStrategy(AbstractStrategy):
    def __init__(self, robot_id: int, field_bounds: Optional[FieldBounds] = None):
        """
        Initializes the RobotPlacementStrategy with a specific robot ID.

        :param robot_id: The ID of the robot this strategy will control.
        :param field_bounds: The bounds of the field to operate within.
        """
        self.robot_id = robot_id
        self.field_bounds = field_bounds if field_bounds else Field.FULL_FIELD_BOUNDS
        super().__init__()

    def assert_exp_robots(self, n_runtime_friendly: int, n_runtime_enemy: int):
        if 1 <= n_runtime_friendly <= 6:
            return True
        return False

    def assert_exp_goals(self, includes_my_goal_line: bool, includes_opp_goal_line: bool):
        return True  # No specific goal line requirements

    def get_min_bounding_zone(self) -> Optional[FieldBounds]:
        # toggles robot between (1, -1) and (1, 1)
        return FieldBounds(top_left=(-1, 1), bottom_right=(1, -1))

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
        calc_center = CalculateFieldCenter(field_bounds=self.field_bounds, output_key=field_center_key)

        coach_root.add_children(
            [
                set_rbt_id,
                calc_center,
                RobotPlacementStep(rd_robot_id=robot_id_key, field_center_key=field_center_key),
            ]
        )

        return coach_root
