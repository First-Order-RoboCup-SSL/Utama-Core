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


_BALL_KEY = ObjectKey(TeamType.NEUTRAL, ObjectType.BALL, 0)


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

    def add_game_frame(self, game: GameFrame):
        # Runs for every object of both teams' frames every tick: one key lookup per object.
        self.raw_games_history.append(game)
        current_ts = game.ts

        if game.ball:
            self._record(_BALL_KEY, game.ball, current_ts)

        for robots_dict, team_type in ((game.friendly_robots, TeamType.FRIENDLY), (game.enemy_robots, TeamType.ENEMY)):
            for robot in robots_dict.values():
                if isinstance(robot, Robot) and isinstance(robot.id, int):
                    self._record(ObjectKey(team_type, ObjectType.ROBOT, robot.id), robot, current_ts)
                else:
                    get_structured_object_key(robot, team_type)  # logs the warning; nothing is stored

    def _record(self, key: ObjectKey, entity: Union[Robot, Ball], timestamp: float) -> None:
        """Stores `entity`'s position and velocity, creating its deques on first use."""
        attributes = self.historical_data.get(key)
        for attribute_type, vector in ((AttributeType.POSITION, entity.p), (AttributeType.VELOCITY, entity.v)):
            if vector is None:
                continue
            if attributes is None:
                attributes = self.historical_data[key] = {}
            history = attributes.get(attribute_type)
            if history is None:
                history = attributes[attribute_type] = deque(maxlen=self.max_history)
            try:
                history.append((timestamp, _vector_to_floats(vector)))
            except TypeError as e:
                logger.error(f"Error converting vector for {key}, {attribute_type}: {e}")

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
