import numpy as np
from typing import List, Optional, Union, Dict, Tuple
from entities.data.object import ObjectKey, TeamType, ObjectType
from entities.game.robot import Robot
from entities.game.ball import Ball
import warnings


class ProximityLookup:
    """
    A proximity map that tracks the distance between robots and the ball.
    """

    def __init__(
        self,
        friendly_robots: Optional[Dict[int, Robot]],
        enemy_robots: Optional[Dict[int, Robot]],
        ball: Optional[Ball],
    ):
        """
        Initialize the proximity map with a set of points.
        :param point_array: A 2D numpy array where each row is a point in the format [x, y].
        """
        self.friendly_end_idx = len(friendly_robots) if friendly_robots else 0
        self.enemy_end_idx = (
            self.friendly_end_idx + len(enemy_robots)
            if enemy_robots
            else self.friendly_end_idx
        )
        self.object_keys, self.point_array = self._get_object_keys_and_point_array(
            friendly_robots, enemy_robots, ball
        )
        self.key_index_map = {key: i for i, key in enumerate(self.object_keys)}
        self.proximity_matrix = self._build_proximity_matrix(self.point_array)

    def _get_object_keys_and_point_array(
        self,
        friendly_robots: Optional[Dict[int, Robot]],
        enemy_robots: Optional[Dict[int, Robot]],
        ball: Optional[Ball],
    ) -> Tuple[List[ObjectKey], np.ndarray]:
        object_keys = []
        point_array = []
        for robot in friendly_robots.values():
            object_keys.append(ObjectKey(TeamType.FRIENDLY, ObjectType.ROBOT, robot.id))
            point_array.append(robot.p.to_array())
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
        """
        Build the pairwise Euclidean distance matrix between all objects.
        """
        if point_array.size < 2:
            return None

        diffs = (
            point_array[:, np.newaxis, :] - point_array[np.newaxis, :, :]
        )  # (n, n, 2)
        dist_matrix = np.linalg.norm(diffs, axis=-1)  # (n, n)
        np.fill_diagonal(dist_matrix, np.inf)  # Exclude self-comparison
        return dist_matrix

    def closest_to_ball(
        self, team_type: Optional[TeamType] = None
    ) -> Tuple[Optional[ObjectKey], float]:
        """
        Find the closest robot to the ball.
        :param team_type: Optional team type to filter results.
        :return: Tuple of the closest ObjectKey and its distance to the ball.
        If no robots are found, returns (None, np.inf).
        """
        if self.proximity_matrix is None:
            warnings.warn("Proximity matrix is empty, cannot find closest to ball.")
            return (None, np.inf)

        if self.object_keys[-1].object_type != ObjectType.BALL:
            warnings.warn(
                "Invalid closest_to_ball query: cannot find ball in proximity lookup."
            )
            return (None, np.inf)
        ball_index = len(self.object_keys) - 1  # Last in order
        distances = self.proximity_matrix[ball_index]

        # Determine slice based on team
        if team_type == TeamType.FRIENDLY:
            if self.friendly_end_idx == 0:
                return (None, np.inf)
            sub_distances = distances[: self.friendly_end_idx]
            offset = 0
        elif team_type == TeamType.ENEMY:
            if self.enemy_end_idx == self.friendly_end_idx:
                return (None, np.inf)
            sub_distances = distances[self.friendly_end_idx : self.enemy_end_idx]
            offset = self.friendly_end_idx
        else:  # all robots
            if self.enemy_end_idx == 0:
                return (None, np.inf)
            sub_distances = distances[: self.enemy_end_idx]
            offset = 0

        closest_relative_index = np.argmin(sub_distances)
        closest_absolute_index = offset + closest_relative_index
        closest_distance = sub_distances[closest_relative_index]
        return self.object_keys[closest_absolute_index], closest_distance
