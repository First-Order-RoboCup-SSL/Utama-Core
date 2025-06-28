from typing import List, Tuple

from shapely import Polygon
from entities.game.game_frame import GameFrame
from entities.game.robot import Robot
from motion_planning.src.planning.exit_strategies import ExitStrategy
from motion_planning.src.planning.path_planner import (
    BisectorPlanner,
    RRTPlanner,
    DynamicWindowPlanner,
    target_inside_robot_radius,
)
from math import dist
import time
from entities.game.field import Field
from enum import Enum


class TempObstacleType(Enum):
    NONE = []
    FIELD = [Field.full_field]
    DEFENCE_ZONES = [Field.left_defense_area, Field.right_defense_area]
    ALL = [Field.left_defense_area, Field.right_defense_area, Field.full_field]


class TimedSwitchController:
    """Takes two planners, one run per frame and one run per N frames,
    idea is that the slower planner gives more accurate global guidance
    """

    DEFAULT_RUN = 60  # SLow planner is invoked once every DEFAULT_RUN frames

    def __init__(
        self,
        num_robots: int,
        game: GameFrame,
        exit_strategy: ExitStrategy,
        friendly_colour,
        env,
    ):
        self.num_robots = num_robots
        self._exit_strategy = exit_strategy
        self._slow_planner = BisectorPlanner(game, friendly_colour, env)
        self._game = game
        self._fast_planner = DynamicWindowPlanner(game)
        self._real_targets = [None for _ in range(num_robots)]
        self._intermediate_target = [None for _ in range(num_robots)]
        self._exit_points = [None for _ in range(num_robots)]
        self._last_slow_frame = [0 for _ in range(num_robots)]

        # DEBUG ONLY
        self._env = env

    def path_to(
        self,
        target: Tuple[float, float],
        robot_id: int,
        temporary_obstacles_enum: TempObstacleType,
    ) -> Tuple[float, float]:
        """
        Computes the path to the given target for the specified robot, considering temporary obstacles such as defence zones, field or None


        Args:
            target (Tuple[float, float]): The target coordinates (x, y) to which the robot should navigate.
            robot_id (int): The identifier of the robot for which the path is being computed.
            temporary_obstacles_enum (TempObstacleType): An enumeration indicating the type of temporary obstacles to consider.

        Returns:
            Tuple[float, float]: The next coordinates (x, y) in the path to the target.
        """
        robot_position = self._game.friendly_robots[robot_id].p

        if self._exit_points[robot_id] is None:
            required_exit_point = self._exit_strategy.get_exit_point(
                (robot_position.x, robot_position.y), temporary_obstacles_enum.value
            )
            if required_exit_point is not None:
                # Should not be too far inside the obstacle, use the fast planner with no temporary obstacles
                # to give a safe path to the edge of the obstacle
                self._exit_points[robot_id] = required_exit_point

        if self._exit_points[robot_id] is not None:
            print("ALREADY trying to exit", self._exit_points[robot_id])
            if ExitStrategy.is_close_enough_to_exit_point(
                (robot_position.x, robot_position.y), self._exit_points[robot_id]
            ):
                self._exit_points[robot_id] = None
            else:
                return self._fast_planner.path_to(
                    robot_id,
                    self._exit_points[robot_id],
                    temporary_obstacles=TempObstacleType.NONE.value,
                )[0]

        if target == self._real_targets[robot_id]:
            if self._last_slow_frame[robot_id] == 0:
                # Invoke the slow planner and reset the counter
                self._intermediate_target[robot_id] = self._slow_planner.path_to(
                    robot_id, target, temporary_obstacles=temporary_obstacles_enum.value
                )
                self._last_slow_frame[robot_id] = self.DEFAULT_RUN
            else:
                # Count down until the next slow frame
                self._last_slow_frame[robot_id] -= 1
        else:
            self._real_targets[robot_id] = target
            self._intermediate_target[robot_id] = self._slow_planner.path_to(
                robot_id, target, temporary_obstacles=temporary_obstacles_enum.value
            )
            self._last_slow_frame[robot_id] = self.DEFAULT_RUN

        target = self._fast_planner.path_to(
            robot_id,
            self._intermediate_target[robot_id],
            temporary_obstacles=temporary_obstacles_enum.value,
        )[0]
        return target
