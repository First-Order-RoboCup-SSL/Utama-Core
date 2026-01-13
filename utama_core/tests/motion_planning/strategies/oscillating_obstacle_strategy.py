"""Strategy for controlling moving obstacles that oscillate back and forth."""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, List, Optional

import py_trees

if TYPE_CHECKING:
    from utama_core.tests.motion_planning.single_robot_moving_obstacle_test import (
        MovingObstacleConfig,
    )

from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game.field import FieldBounds
from utama_core.skills.src.utils.move_utils import move
from utama_core.strategy.common.abstract_behaviour import AbstractBehaviour
from utama_core.strategy.common.abstract_strategy import AbstractStrategy


class OscillatingObstacleBehaviour(AbstractBehaviour):
    """
    Behaviour that makes robots oscillate back and forth.

    Args:
        obstacle_id: The robot ID to control
        center_position: Center point of oscillation (x, y)
        oscillation_axis: 'x' or 'y' - which axis to oscillate along
        amplitude: How far to move from center
        direction_up_or_right: True for going up/right first, False for going down/left first
        speed: Speed of oscillation (radians per second in sin wave)
    """

    def __init__(
        self,
        obstacle_id: int,
        center_position: tuple[float, float],
        oscillation_axis: str,
        amplitude: float,
        direction_up_or_right: bool,
        speed: float,
    ):
        super().__init__(name=f"OscillatingObstacle_{obstacle_id}")
        self.obstacle_id = obstacle_id
        self.center_x, self.center_y = center_position
        self.oscillation_axis = oscillation_axis.lower()
        self.amplitude = amplitude
        self.direction_up_or_right = direction_up_or_right
        self.speed = speed
        self.start_time = None

    def initialise(self):
        """Record start time when behaviour is initialized."""
        self.start_time = time.time()

    def update(self) -> py_trees.common.Status:
        """Command robot to oscillate along specified axis."""
        game = self.blackboard.game
        rsim_env = self.blackboard.rsim_env

        if not game.friendly_robots or self.obstacle_id not in game.friendly_robots:
            return py_trees.common.Status.RUNNING

        # Calculate oscillation based on elapsed time
        elapsed_time = time.time() - self.start_time
        if self.direction_up_or_right:
            offset = self.amplitude * math.sin(self.speed * elapsed_time)
        else:
            offset = self.amplitude * math.cos(self.speed * elapsed_time)

        # Calculate target position based on oscillation axis
        if self.oscillation_axis == "x":
            target_x = self.center_x + offset
            target_y = self.center_y
        elif self.oscillation_axis == "y":
            target_x = self.center_x
            target_y = self.center_y + offset
        else:
            # Default to no movement if invalid axis
            target_x = self.center_x
            target_y = self.center_y

        target_position = Vector2D(target_x, target_y)

        # Visualize oscillation path
        if rsim_env:
            # Draw center point
            rsim_env.draw_point(self.center_x, self.center_y, color="blue")

            # Draw current target
            rsim_env.draw_point(target_x, target_y, color="yellow")

            # Draw oscillation range
            if self.oscillation_axis == "x":
                rsim_env.draw_line(
                    [
                        (self.center_x - self.amplitude, self.center_y),
                        (self.center_x + self.amplitude, self.center_y),
                    ],
                    color="green",
                    width=1,
                )
            else:
                rsim_env.draw_line(
                    [
                        (self.center_x, self.center_y - self.amplitude),
                        (self.center_x, self.center_y + self.amplitude),
                    ],
                    color="green",
                    width=1,
                )

        # Use motion controller to generate command
        # This ensures proper obstacle avoidance and velocity calculation
        cmd = move(
            game,
            self.blackboard.motion_controller,
            self.obstacle_id,
            target_position,
            0.0,  # Face forward
        )

        self.blackboard.cmd_map[self.obstacle_id] = cmd
        return py_trees.common.Status.RUNNING


class OscillatingObstacleStrategy(AbstractStrategy):
    """
    Strategy that controls multiple robots to oscillate as moving obstacles.

    Args:
        obstacle_configs: List of MovingObstacleConfig objects defining each obstacle's behavior
    """

    def __init__(self, obstacle_configs: List["MovingObstacleConfig"]):
        self.obstacle_configs = obstacle_configs
        super().__init__()

    def assert_exp_robots(self, n_runtime_friendly: int, n_runtime_enemy: int):
        """Requires number of friendly robots to match obstacle count."""
        return n_runtime_friendly >= len(self.obstacle_configs)

    def assert_exp_goals(self, includes_my_goal_line: bool, includes_opp_goal_line: bool):
        return True

    def get_min_bounding_zone(self) -> Optional[FieldBounds]:
        """Calculate bounding box for all oscillating obstacles."""
        if not self.obstacle_configs:
            return None

        all_points = []
        for config in self.obstacle_configs:
            cx, cy = config.center_position
            amplitude = config.amplitude

            # Add extreme points of oscillation
            if config.oscillation_axis.lower() == "x":
                all_points.append((cx - amplitude, cy))
                all_points.append((cx + amplitude, cy))
            else:
                all_points.append((cx, cy - amplitude))
                all_points.append((cx, cy + amplitude))

        # Calculate bounding box
        xs = [p[0] for p in all_points]
        ys = [p[1] for p in all_points]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        # Add some padding
        padding = 0.5
        return FieldBounds(
            top_left=(min_x - padding, max_y + padding),
            bottom_right=(max_x + padding, min_y - padding),
        )

    def create_behaviour_tree(self) -> py_trees.behaviour.Behaviour:
        """Create parallel behaviour tree with all oscillating obstacles."""
        if len(self.obstacle_configs) == 1:
            # Single obstacle - just return one behaviour
            config = self.obstacle_configs[0]
            return OscillatingObstacleBehaviour(
                obstacle_id=0,
                center_position=config.center_position,
                oscillation_axis=config.oscillation_axis,
                amplitude=config.amplitude,
                direction_up_or_right=config.direction_up_or_right,
                speed=config.speed,
            )

        # Multiple obstacles - create parallel behaviours
        behaviours = []
        for i, config in enumerate(self.obstacle_configs):
            behaviour = OscillatingObstacleBehaviour(
                obstacle_id=i,
                center_position=config.center_position,
                oscillation_axis=config.oscillation_axis,
                amplitude=config.amplitude,
                direction_up_or_right=config.direction_up_or_right,
                speed=config.speed,
            )
            behaviours.append(behaviour)

        # Run all obstacle behaviours in parallel
        return py_trees.composites.Parallel(
            name="OscillatingObstacles",
            policy=py_trees.common.ParallelPolicy.SuccessOnAll(),
            children=behaviours,
        )
