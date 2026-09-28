import logging
from collections import deque
from enum import Enum, auto
from itertools import islice
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from utama_core.entities.data.object import ObjectKey, ObjectType, TeamType
from utama_core.entities.data.vector import Vector2D, Vector3D
from utama_core.entities.game.game_frame import Ball, GameFrame, Robot

logger = logging.getLogger(__name__)


# --- Enums (keep as is) ---
class AttributeType(Enum):
    POSITION = auto()
    VELOCITY = auto()
    # ACCELERATION = auto() # If you decide to store pre-calculated acceleration


def get_structured_object_key(obj: Any, team: TeamType) -> Optional[ObjectKey]:
    if isinstance(obj, Robot) and hasattr(obj, "id") and isinstance(obj.id, int):
        return ObjectKey(team, ObjectType.ROBOT, obj.id)
    elif isinstance(obj, Ball):
        return ObjectKey(TeamType.NEUTRAL, ObjectType.BALL, 0)
    logger.warning(f"Could not determine ObjectKey for object of type {type(obj)} with team {team}")
    return None


# Helper to convert a Vector to the stored form: a tuple of Python floats, the
# values a float64 array of it would hold (and `np.array` of a list of them is
# that array), without building an array per object per frame.
def _vector_to_floats(vector: Union[Vector2D, Vector3D]) -> Tuple[float, ...]:
    if isinstance(vector, Vector2D):
        return (float(vector.x), float(vector.y))
    elif isinstance(vector, Vector3D):
        return (float(vector.x), float(vector.y), float(vector.z))
    raise TypeError(f"Unsupported vector type for NumPy conversion: {type(vector)}")


class GameHistory:
    def __init__(self, max_history: int):
        self.max_history = max_history
        self.raw_games_history: deque[GameFrame] = deque(maxlen=max_history)

        # Generic historical data storage:
        # ObjectKey -> AttributeType -> deque[(timestamp: float, value: tuple of floats)]
        self.historical_data: Dict[ObjectKey, Dict[AttributeType, deque[Tuple[float, Tuple[float, ...]]]]] = {}

    def _ensure_attribute_deque_exists(self, object_key: ObjectKey, attribute_type: AttributeType):
        """Ensures a deque exists for the given object_key and attribute_type."""
        if object_key not in self.historical_data:
            self.historical_data[object_key] = {}
        if attribute_type not in self.historical_data[object_key]:
            self.historical_data[object_key][attribute_type] = deque(maxlen=self.max_history)

    def _add_attribute_to_history(
        self,
        object_key: ObjectKey,
        attribute_type: AttributeType,
        timestamp: float,
        value_vector_obj: Optional[Union[Vector2D, Vector3D]],
    ):
        """Adds a single attribute value (as a tuple of floats) to the history."""
        if value_vector_obj is None:
            return  # Don't store None values, or decide on a specific handling

        self._ensure_attribute_deque_exists(object_key, attribute_type)
        try:
            self.historical_data[object_key][attribute_type].append((timestamp, _vector_to_floats(value_vector_obj)))
        except TypeError as e:
            logger.error(f"Error converting vector for {object_key}, {attribute_type}: {e}")

    def _process_entity_for_history(self, entity: Any, entity_key: ObjectKey, timestamp: float):
        """Helper to process position and velocity for a given entity."""
        if not entity_key:
            return

        if hasattr(entity, "p"):
            self._add_attribute_to_history(entity_key, AttributeType.POSITION, timestamp, entity.p)
        if hasattr(entity, "v"):
            self._add_attribute_to_history(entity_key, AttributeType.VELOCITY, timestamp, entity.v)
        # If you pre-calculate and store acceleration:
        # if hasattr(entity, "a"):
        #     self._add_attribute_to_history(entity_key, AttributeType.ACCELERATION, timestamp, entity.a)

    def add_game_frame(self, game: GameFrame):
        self.raw_games_history.append(game)
        current_ts = game.ts

        # Process Ball
        if game.ball:
            ball_key = get_structured_object_key(game.ball, TeamType.NEUTRAL)  # Ball is neutral
            if ball_key:
                self._process_entity_for_history(game.ball, ball_key, current_ts)

        # Process Robots (Friendly and Enemy)
        robot_groups = [
            (game.friendly_robots, TeamType.FRIENDLY),
            (game.enemy_robots, TeamType.ENEMY),
        ]
        for robots_dict, team_type in robot_groups:
            for robot_instance in robots_dict.values():
                robot_key = get_structured_object_key(robot_instance, team_type)
                if robot_key:
                    self._process_entity_for_history(robot_instance, robot_key, current_ts)

    def get_historical_attribute_entries(
        self,
        object_key: ObjectKey,
        attribute_type: AttributeType,
        num_points: int,
    ) -> List[Tuple[float, Tuple[float, ...]]]:
        """The last num_points stored `(timestamp, value)` pairs for an object, oldest
        first, values as tuples of floats; empty if there are none. What
        `get_historical_attribute_series` stacks into arrays, for a caller that
        reads a handful of values and would only pay for building the arrays."""
        if num_points <= 0:
            return []
        object_attributes = self.historical_data.get(object_key)
        if not object_attributes:
            return []
        history_deque = object_attributes.get(attribute_type)
        if not history_deque:  # Handles both key not found or empty deque
            return []
        n = len(history_deque)
        return list(islice(history_deque, max(0, n - num_points), n))

    def get_historical_attribute_series(
        self,
        object_key: ObjectKey,
        attribute_type: AttributeType,
        num_points: int,
    ) -> Tuple[np.ndarray, np.ndarray]:  # Returns (timestamps_np, values_np)
        """Retrieves the last num_points of (timestamp, attribute_value_np) for a given object.

        Returns data as NumPy arrays (timestamps, values), oldest to newest. Returns empty NumPy arrays if no data is
        available.
        """
        entries = self.get_historical_attribute_entries(object_key, attribute_type, num_points)
        if not entries:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

        timestamps_list: List[float] = [ts for ts, _ in entries]
        vector_values_list: List[Tuple[float, ...]] = [vec for _, vec in entries]

        timestamps_np = np.array(timestamps_list, dtype=np.float64)
        # vector_values_list holds equal-length tuples of floats, so this is a 2D float64 array.
        values_np = np.array(vector_values_list)

        return timestamps_np, values_np

    def n_steps_ago(self, n: int) -> GameFrame:
        if not (0 < n <= len(self.raw_games_history)):
            raise IndexError(
                f"Cannot get game {n} steps ago. History size: {len(self.raw_games_history)}, requested: {n}"
            )
        return self.raw_games_history[-n]
