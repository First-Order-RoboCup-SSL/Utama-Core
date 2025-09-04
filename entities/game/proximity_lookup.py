import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np

from entities.data.object import ObjectKey, ObjectType, TeamType
from entities.game.ball import Ball
from entities.game.robot import Robot


class ProximityLookup:
    """A proximity map that tracks the distance between robots and the ball."""

    def __init__(
        self,
        friendly_robots: Optional[Dict[int, Robot]],
        enemy_robots: Optional[Dict[int, Robot]],
        ball: Optional[Ball],
    ):
        """Initialize the proximity map with a set of points.

        :param point_array: A 2D numpy array where each row is a point in the format [x, y].
        """
        self.friendly_end_idx = len(friendly_robots) if friendly_robots else 0
        self.enemy_end_idx = self.friendly_end_idx + len(enemy_robots) if enemy_robots else self.friendly_end_idx
        self.object_keys, self.point_array = self._get_object_keys_and_point_array(friendly_robots, enemy_robots, ball)
        self.key_index_map = {key: i for i, key in enumerate(self.object_keys)}
        self.proximity_matrix = self._build_proximity_matrix(self.point_array)

    def _get_object_keys_and_point_array(
        self,
        friendly_robots: Optional[Dict[int, Robot]],
        enemy_robots: Optional[Dict[int, Robot]],
        ball: Optional[Ball],
    ) -> Tuple[List[ObjectKey], np.ndarray]:
        object_keys: List[ObjectKey] = []
        point_array: List[np.ndarray] = []

        if friendly_robots:
            for robot in friendly_robots.values():
                object_keys.append(ObjectKey(TeamType.FRIENDLY, ObjectType.ROBOT, robot.id))
                point_array.append(robot.p.to_array())

        if enemy_robots:
            for robot in enemy_robots.values():
                object_keys.append(ObjectKey(TeamType.ENEMY, ObjectType.ROBOT, robot.id))
                point_array.append(robot.p.to_array())

        if ball:
            object_keys.append(ObjectKey(TeamType.NEUTRAL, ObjectType.BALL, 0))
            point_array.append(ball.p.to_2d().to_array())

        return object_keys, np.array(point_array)

    # Optimisation: could potentially store np.dot for ranking purposes,
    # then sqrt only when querying (profile to see if this is worth it)
    def _build_proximity_matrix(self, point_array: np.ndarray) -> np.ndarray:
        """Build the pairwise Euclidean distance matrix between all objects."""
        if point_array.size < 2:
            return None

        diffs = point_array[:, np.newaxis, :] - point_array[np.newaxis, :, :]  # (n, n, 2)
        dist_matrix = np.linalg.norm(diffs, axis=-1)  # (n, n)
        np.fill_diagonal(dist_matrix, np.inf)  # Exclude self-comparison
        return dist_matrix

    def _closest_to_index(
        self,
        source_index: int,
        team_type_filter: Optional[TeamType] = None,
    ) -> Tuple[Optional[ObjectKey], float]:
        """Generalized helper to find the closest object to a given index, filtered by team type."""
        if self.proximity_matrix is None:
            warnings.warn("Proximity matrix is empty, cannot compute closest object.")
            return (None, np.inf)

        distances = self.proximity_matrix[source_index]

        # Determine slice based on team
        if team_type_filter == TeamType.FRIENDLY:
            if self.friendly_end_idx == 0:
                return (None, np.inf)
            sub_distances = distances[: self.friendly_end_idx]
            offset = 0
        elif team_type_filter == TeamType.ENEMY:
            if self.enemy_end_idx == self.friendly_end_idx:
                return (None, np.inf)
            sub_distances = distances[self.friendly_end_idx : self.enemy_end_idx]
            offset = self.friendly_end_idx
        else:
            if self.enemy_end_idx == 0:
                return (None, np.inf)
            sub_distances = distances[: self.enemy_end_idx]
            offset = 0

        closest_relative_index = np.argmin(sub_distances)
        closest_absolute_index = offset + closest_relative_index
        closest_distance = sub_distances[closest_relative_index]

        return self.object_keys[closest_absolute_index], closest_distance

    def closest_to_ball(self, team_type_filter: Optional[TeamType] = None) -> Tuple[Optional[ObjectKey], float]:
        if not self.object_keys or self.object_keys[-1].object_type != ObjectType.BALL:
            warnings.warn("Invalid closest_to_ball query: cannot find ball in proximity lookup.")
            return (None, np.inf)

        ball_index = len(self.object_keys) - 1
        return self._closest_to_index(ball_index, team_type_filter)

    def closest_to_robot(
        self, robot_key: ObjectKey, team_type_filter: Optional[TeamType] = None
    ) -> Tuple[Optional[ObjectKey], float]:
        if robot_key not in self.key_index_map:
            warnings.warn(f"Robot {robot_key} not found in proximity lookup.")
            return (None, np.inf)

        robot_index = self.key_index_map[robot_key]
        return self._closest_to_index(robot_index, team_type_filter)
