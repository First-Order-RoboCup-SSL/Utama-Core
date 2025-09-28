# File to contain strategies for when the robot is inside a temporary obstacle and needs to exit
# As per the RoboCup rules, we must make best effort to stay outside defence zones

from abc import ABC, abstractmethod
from math import hypot
from typing import Iterable, Optional, Tuple

from utama_core.motion_planning.src.planning.geometry import AxisAlignedRectangle
from utama_core.motion_planning.src.planning.obstacles import ObstacleRegion


class ExitStrategy(ABC):
    """Base class for exit strategies, which determine how a robot should exit a temporary obstacle, if no action is
    required return None."""

    @abstractmethod
    def get_exit_point(
        self, robot_position: Tuple[float, float], obstacles: Iterable[ObstacleRegion]
    ) -> Optional[Tuple[float, float]]: ...

    EXIT_POINT_BUFFER = 0.12  # Distance from the obstacle boundary to exit point, ensuring robot is completely outside
    OBSTACLE_SAFE_BUFFER = 0.05  # Attempt to exit the obstacle when the robot is closer than this distance to the boundary or inside the polygon
    CLOSE_ENOUGH_TO_EXIT_POINT = 0.03  # Distance from the exit point to consider it reached

    def _is_too_close(self, robot_position: Tuple[float, float], obstacles: Iterable[ObstacleRegion]) -> bool:
        """Returns True if the robot is within OBSTACLE_SAFE_BUFFER or inside any obstacle."""
        for obstacle in obstacles:
            rect = obstacle.rect
            if rect.contains(robot_position) or rect.distance_to_boundary(robot_position) < self.OBSTACLE_SAFE_BUFFER:
                return True
        return False

    @staticmethod
    def _closest_point_with_buffer(rect: AxisAlignedRectangle, robot_pos: Tuple[float, float]) -> Tuple[float, float]:
        return rect.exit_point_with_buffer(robot_pos, ExitStrategy.EXIT_POINT_BUFFER)

    @staticmethod
    def is_close_enough_to_exit_point(robot_pos: Tuple[float, float], exit_point: Tuple[float, float]) -> bool:
        dx = robot_pos[0] - exit_point[0]
        dy = robot_pos[1] - exit_point[1]
        return hypot(dx, dy) < ExitStrategy.CLOSE_ENOUGH_TO_EXIT_POINT


class ClosestPointExit(ExitStrategy):
    def get_exit_point(
        self, robot_position: Tuple[float, float], obstacles: Iterable[ObstacleRegion]
    ) -> Optional[Tuple[float, float]]:
        """Returns the closest point to the robot that is outside the obstacles if the robot position is inside any of
        the obstacles.

        We assume that the obstacles do not overlap so the robot can be in at most one obstacle at a time.
        """

        for obstacle in obstacles:
            if self._is_too_close(robot_position, [obstacle]):
                return self._closest_point_with_buffer(obstacle.rect, robot_position)
        return None
