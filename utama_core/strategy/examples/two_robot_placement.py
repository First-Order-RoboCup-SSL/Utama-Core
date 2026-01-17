import math
import random
from typing import Any, Optional

import numpy as np
import py_trees
from py_trees.composites import Parallel, Sequence

from utama_core.config.settings import TIMESTEP
from utama_core.entities.game.field import FieldBounds
from utama_core.global_utils.math_utils import Vector2D
from utama_core.skills.src.utils.move_utils import move
from utama_core.strategy.common.abstract_behaviour import AbstractBehaviour

# from robot_control.src.tests.utils import one_robot_placement
from utama_core.strategy.common.abstract_strategy import AbstractStrategy


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
        start_point: tuple[float, float],
        end_point: tuple[float, float],
        turn_key: str,
        my_turn_idx: int,
        next_turn_idx: int,
    ):
        super().__init__()
        self.robot_id_key = rd_robot_id
        self.start_point = start_point
        self.end_point = end_point
        self.current_target = end_point  # Start moving towards end point

        self.turn_key = turn_key
        self.my_turn_idx = my_turn_idx
        self.next_turn_idx = next_turn_idx

    def setup_(self):
        self.blackboard.register_key(key=self.robot_id_key, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=self.turn_key, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=self.turn_key, access=py_trees.common.Access.WRITE)

    def update(self) -> py_trees.common.Status:
        """Closure which advances the simulation by one step."""
        game = self.blackboard.game
        rsim_env = self.blackboard.rsim_env
        id: int = self.blackboard.get(self.robot_id_key)

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
    def __init__(self, robot_id: int):
        """
        Initializes the TwoRobotPlacementStrategy with a specific robot ID (though this handles two robots).
        """
        self.robot_id = robot_id
        super().__init__()

    def assert_exp_robots(self, n_runtime_friendly: int, n_runtime_enemy: int):
        if 1 <= n_runtime_friendly <= 6:
            return True
        return False

    def assert_exp_goals(self, includes_my_goal_line: bool, includes_opp_goal_line: bool):
        return True  # No specific goal line requirements

    def get_min_bounding_zone(self) -> Optional[FieldBounds]:
        return FieldBounds(top_left=(-1, 1), bottom_right=(1, -1))

    def create_behaviour_tree(self) -> py_trees.behaviour.Behaviour:
        """Factory function to create a complete behaviour tree."""

        robot1_key = "robot1_id"
        robot2_key = "robot2_id"
        turn_key = "CrossMovementTurn"

        coach_root = Sequence(name="CoachRoot", memory=True)

        # We assume the user wants to use the first available robots or distinct ones.
        # Since the strategy is init with one robot_id from main, we might need to assume
        # specific logic for two robots. Ideally main.py or strategy runner assigns IDs.
        # However, typically 'robot_id' passed here is just one.
        # For this specific "TwoRobot" request, we'll hardcode using robot 0 and 1,
        # or use the passed robot_id and robot_id + 1 if available.
        # Let's assume 0 and 1 for simplicity of this standalone example in `examples/`.

        set_robot1 = SetBlackboardVariable(name="SetR1", variable_name=robot1_key, value=0)
        set_robot2 = SetBlackboardVariable(name="SetR2", variable_name=robot2_key, value=1)

        # Initialize turn to 0 (Robot 1 starts)
        set_turn = SetBlackboardVariable(name="InitTurn", variable_name=turn_key, value=0)

        # Robot 1 (X-mover): Center (3.375, 0), Length 1m -> X range [2.875, 3.875]
        # Starts moving towards end point.
        move_robot1 = RobotPlacementStep(
            rd_robot_id=robot1_key,
            start_point=(2.875, 0.0),
            end_point=(3.875, 0.0),
            turn_key=turn_key,
            my_turn_idx=0,
            next_turn_idx=1,
        )

        # Robot 2 (Y-mover): Center (3.375, 0), Length 1m -> Y range [-0.5, 0.5]
        move_robot2 = RobotPlacementStep(
            rd_robot_id=robot2_key,
            start_point=(3.375, -0.5),
            end_point=(3.375, 0.5),
            turn_key=turn_key,
            my_turn_idx=1,
            next_turn_idx=0,
        )

        # Use Parallel to allow both robots to be ticked
        action_parallel = Parallel(
            name="RobotActions", policy=py_trees.common.ParallelPolicy.SuccessOnAll(synchronise=False)
        )
        action_parallel.add_children([move_robot1, move_robot2])

        coach_root.add_children([set_robot1, set_robot2, set_turn, action_parallel])

        return coach_root
