# File to contain strategies for when the robot is inside a temporary obstacle and needs to exit
# As per the RoboCup rules, we must make best effort to stay outside defence zones

from abc import ABC, abstractmethod
from typing import Tuple, Optional, List

from shapely import Polygon, Point
from shapely.ops import nearest_points


class ExitStrategy(ABC):
    """Base class for exit strategies, which determine how a robot should exit a temporary obstacle, if no action is required return None"""

    @abstractmethod
    def get_exit_point(
        self, robot_position: Tuple[float, float], obstacles: List[Polygon]
    ) -> Optional[Tuple[float, float]]: ...

    EXIT_POINT_BUFFER = 0.12  # Distance from the obstacle boundary to exit point, ensuring robot is completely outside
    OBSTACLE_SAFE_BUFFER = 0.05  # Attempt to exit the obstacle when the robot is closer than this distance to the boundary or inside the polygon
    CLOSE_ENOUGH_TO_EXIT_POINT = (
        0.03  # Distance from the exit point to consider it reached
    )

    def _is_too_close(
        self, robot_position: Tuple[float, float], obstacles: List[Polygon]
    ) -> bool:
        """Returns True if the robot is within OBSTACLE_SAFE_BUFFER or inside any of the obstacles, False otherwise."""
        robot_point = Point(robot_position)
        for obstacle in obstacles:
            if (
                obstacle.contains(robot_point)
                or obstacle.distance(robot_point) < self.OBSTACLE_SAFE_BUFFER
            ):
                return True
        return False

    @staticmethod
    def closest_point_with_buffer(polygon: Polygon, robot_pos: Point) -> Point:
        """
        Moves the robot outward from the closest point on the polygon if it's too close.

        :param polygon: A Shapely Polygon representing the boundary.
        :param robot_pos: A Shapely Point representing the robot's position.
        :param safety_buffer: The minimum required distance from the polygon.
        :return: A Shapely Point representing the offset position.
        """
        # Find the closest point on the polygon boundary
        closest_point = polygon.exterior.interpolate(
            polygon.exterior.project(robot_pos)
        )

        # Check if robot is inside or outside the polygon
        if polygon.contains(closest_point):
            # Inside case: Move outward (closest point → robot)
            print("INSIDE")
            direction_vec = (
                robot_pos.x - closest_point.x,
                robot_pos.y - closest_point.y,
            )

        else:
            print("OUTSIDE", robot_pos, closest_point)
            # Outside case: Move further out (robot → closest point, reversed)
            direction_vec = (
                closest_point.x - robot_pos.x,
                closest_point.y - robot_pos.y,
            )

        # Normalize the direction vector
        direction_length = (direction_vec[0] ** 2 + direction_vec[1] ** 2) ** 0.5
        if direction_length > 0:
            unit_vec = (
                direction_vec[0] / direction_length,
                direction_vec[1] / direction_length,
            )
        else:
            # If the robot is exactly at the closest point, pick an arbitrary outward direction
            unit_vec = (1, 0)

        # Compute the target point at exactly the safety buffer distance away
        target_x = closest_point.x + ExitStrategy.EXIT_POINT_BUFFER * unit_vec[0]
        target_y = closest_point.y + ExitStrategy.EXIT_POINT_BUFFER * unit_vec[1]

        return Point(target_x, target_y)

    @staticmethod
    def is_close_enough_to_exit_point(
        robot_pos: Tuple[float, float], exit_point: Tuple[float, float]
    ) -> bool:
        """
        Returns True if the robot is close enough to the exit point to consider it reached.

        :param robot_pos: A Shapely Point representing the robot's position.
        :param exit_point: A Shapely Point representing the exit point.
        :return: True if the robot is close enough to the exit point, False otherwise.
        """
        return (
            Point(robot_pos).distance(Point(exit_point))
            < ExitStrategy.CLOSE_ENOUGH_TO_EXIT_POINT
        )


class ClosestPointExit(ExitStrategy):
    def get_exit_point(
        self, robot_position: Tuple[float, float], obstacles: List[Polygon]
    ) -> Optional[Tuple[float, float]]:
        """Returns the closest point to the robot that is outside the obstacles if the robot position is inside
        any of the obstacles. We assume that the obstacles do not overlap so the robot can be in at most one obstacle at a time.
        """

        robot_point = Point(robot_position)
        for obstacle in obstacles:
            if self._is_too_close(robot_position, [obstacle]):
                return self.closest_point_with_buffer(obstacle, robot_point).coords[0]
        return None
