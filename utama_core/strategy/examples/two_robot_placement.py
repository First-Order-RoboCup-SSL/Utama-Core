import math
import random
from typing import Any, Optional

import numpy as np
import py_trees
from py_trees.composites import Parallel, Sequence

from utama_core.config.settings import TIMESTEP
from utama_core.entities.game.field import Field, FieldBounds
from utama_core.global_utils.math_utils import Vector2D
from utama_core.skills.src.utils.move_utils import move
from utama_core.strategy.common.abstract_behaviour import AbstractBehaviour

# from robot_control.src.tests.utils import one_robot_placement
from utama_core.strategy.common.abstract_strategy import AbstractStrategy


class CalculateFieldCenter(AbstractBehaviour):
    """
    Calculates the center of the provided field bounds and writes it to the blackboard.
    """

    def __init__(self, field_bounds: FieldBounds, output_key: str = "FieldCenter"):
        super().__init__(name="CalculateFieldCenter")
        self.output_key = output_key
        self.field_bounds = field_bounds
        self.calculated = False

    def setup_(self):
        self.blackboard.register_key(key=self.output_key, access=py_trees.common.Access.WRITE)

    def update(self) -> py_trees.common.Status:
        if self.calculated:
            return py_trees.common.Status.SUCCESS

        center = self.field_bounds.center
        self.blackboard.set(self.output_key, center, overwrite=True)
        self.calculated = True
        return py_trees.common.Status.SUCCESS


class RobotPlacementStep(AbstractBehaviour):
    """
    A behaviour that commands a robot to move between two specific positions on the field.

    **Args:**
        rd_robot_id (str): Blackboard key for the robot ID.
        start_point (tuple): Starting coordinate (x, y).
        end_point (tuple): Ending coordinate (x, y).
        turn_key (str): Blackboard key for synchronization turn.
        my_turn_idx (int): The turn index for this robot.
        next_turn_idx (int): The turn index to set when movement is complete.
    """

    def __init__(
        self,
        rd_robot_id: str,
        role: str,
        turn_key: str,
        my_turn_idx: int,
        next_turn_idx: int,
        field_center_key: str = "FieldCenter",
    ):
        super().__init__()
        self.robot_id_key = rd_robot_id
        self.role = role
        self.turn_key = turn_key
        self.my_turn_idx = my_turn_idx
        self.next_turn_idx = next_turn_idx
        self.field_center_key = field_center_key

        # State management
        self.current_target = None
        self.initialized = False
        self.start_point = None
        self.end_point = None

    def setup_(self):
        self.blackboard.register_key(key=self.robot_id_key, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=self.turn_key, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=self.turn_key, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=self.field_center_key, access=py_trees.common.Access.READ)

    def update(self) -> py_trees.common.Status:
        """Closure which advances the simulation by one step."""
        game = self.blackboard.game
        rsim_env = self.blackboard.rsim_env
        id: int = self.blackboard.get(self.robot_id_key)

        # Initialize targets if not ready
        if not self.initialized:
            try:
                center = self.blackboard.get(self.field_center_key)
                if center:
                    cx, cy = center
                    if self.role == "horizontal":
                        self.start_point = (cx - 0.5, cy)
                        self.end_point = (cx + 0.5, cy)
                    elif self.role == "vertical":
                        self.start_point = (cx, cy - 0.5)
                        self.end_point = (cx, cy + 0.5)
                    else:
                        return py_trees.common.Status.FAILURE

                    self.current_target = self.end_point
                    self.initialized = True
            except KeyError:
                # Center not yet available
                return py_trees.common.Status.FAILURE

        if not self.initialized:
            return py_trees.common.Status.FAILURE

        # Check whose turn it is
        current_turn = self.blackboard.get(self.turn_key)

        friendly_robots = game.friendly_robots
        if not friendly_robots or id not in friendly_robots:
            return py_trees.common.Status.FAILURE

        rp = friendly_robots[id].p
        cx, cy = rp.x, rp.y

        cmd = None

        # Only move if it is my turn
        if current_turn == self.my_turn_idx:
            tx, ty = self.current_target
            dist_error = math.dist((tx, ty), (cx, cy))

            # Check if reached target
            if dist_error < 0.05:
                # Switch target for next time
                if self.current_target == self.start_point:
                    self.current_target = self.end_point
                else:
                    self.current_target = self.start_point

                # Pass turn to the other robot
                self.blackboard.set(self.turn_key, self.next_turn_idx, overwrite=True)

            else:
                # Move towards target
                bx, by = game.ball.p.x, game.ball.p.y
                oren = np.atan2(by - cy, bx - cx)
                cmd = move(
                    game,
                    self.blackboard.motion_controller,
                    id,
                    Vector2D(tx, ty),
                    oren,
                )

                if rsim_env:
                    rsim_env.draw_point(tx, ty, color="red")
                    v = game.friendly_robots[id].v
                    p = game.friendly_robots[id].p
                    rsim_env.draw_point(p.x + v.x * TIMESTEP * 5, p.y + v.y * TIMESTEP * 5, color="green")

        if cmd:
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


class TwoRobotPlacementStrategy(AbstractStrategy):
    def __init__(
        self,
        first_robot_id: int,
        second_robot_id: int,
        field_bounds: Optional[FieldBounds] = None,
    ):
        """
        Initializes the TwoRobotPlacementStrategy with a specific robot ID (though this handles two robots).
        """
        self.first_robot_id = first_robot_id
        self.second_robot_id = second_robot_id
        self.field_bounds = field_bounds if field_bounds else Field.FULL_FIELD_BOUNDS
        super().__init__()

    def assert_exp_robots(self, n_runtime_friendly: int, n_runtime_enemy: int):
        if 2 == n_runtime_friendly:
            return True
        return False

    def assert_exp_goals(self, includes_my_goal_line: bool, includes_opp_goal_line: bool):
        return True  # No specific goal line requirements

    def get_min_bounding_zone(self) -> Optional[FieldBounds]:
        # toggles robot between (1, -1) and (1, 1)
        # Using full field bounds logic now, but maybe should return specific bounds if needed.
        # For now, keeping consistent with previous simple bounds or updating if needed.
        return FieldBounds(top_left=(-1, 1), bottom_right=(1, -1))

    def create_behaviour_tree(self) -> py_trees.behaviour.Behaviour:
        """Factory function to create a complete behaviour tree."""

        first_robot_key = "first_robot_id"
        second_robot_key = "second_robot_id"
        turn_key = "CrossMovementTurn"
        field_center_key = "FieldCenter"

        coach_root = Sequence(name="CoachRoot", memory=True)

        set_first_robot = SetBlackboardVariable(name="SetR1", variable_name=first_robot_key, value=self.first_robot_id)
        set_second_robot = SetBlackboardVariable(
            name="SetR2", variable_name=second_robot_key, value=self.second_robot_id
        )

        # Initialize turn to 0 (Robot 1 starts)
        set_turn = SetBlackboardVariable(name="InitTurn", variable_name=turn_key, value=0)

        # Calculate Field Center from custom field_bounds
        calc_center = CalculateFieldCenter(field_bounds=self.field_bounds, output_key=field_center_key)

        # Robot 1 (X-mover): Centered at (center_x, center_y), move range +/- 0.5 in X
        move_robot1 = RobotPlacementStep(
            rd_robot_id=first_robot_key,
            role="horizontal",
            turn_key=turn_key,
            my_turn_idx=0,
            next_turn_idx=1,
            field_center_key=field_center_key,
        )

        # Robot 2 (Y-mover): Centered at (center_x, center_y), move range +/- 0.5 in Y
        move_robot2 = RobotPlacementStep(
            rd_robot_id=second_robot_key,
            role="vertical",
            turn_key=turn_key,
            my_turn_idx=1,
            next_turn_idx=0,
            field_center_key=field_center_key,
        )

        # Use Parallel to allow both robots to be ticked
        action_parallel = Parallel(
            name="RobotActions", policy=py_trees.common.ParallelPolicy.SuccessOnAll(synchronise=False)
        )
        action_parallel.add_children([move_robot1, move_robot2])

        coach_root.add_children([set_first_robot, set_second_robot, set_turn, calc_center, action_parallel])

        return coach_root
