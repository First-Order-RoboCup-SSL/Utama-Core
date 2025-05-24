from collections import deque
from entities.game.game import Game, Robot
from entities.game.ball import Ball 
from vector import VectorObject2D, VectorObject3D
from enum import Enum, auto
from typing import Tuple, Any, Union, Optional, Dict, List 
import numpy as np
from itertools import islice
import logging

logger = logging.getLogger(__name__)

# --- Enums (keep as is) ---
class AttributeType(Enum):
    POSITION = auto()
    VELOCITY = auto()
    # ACCELERATION = auto() # If you decide to store pre-calculated acceleration

class TeamType(Enum):
    FRIENDLY = auto()
    ENEMY = auto()
    NEUTRAL = auto()

class ObjectClass(Enum):
    ROBOT = auto()
    BALL = auto()

ObjectKey = Tuple[TeamType, ObjectClass, int]


def get_structured_object_key(obj: Any, team: TeamType) -> Optional[ObjectKey]:
    if isinstance(obj, Robot) and hasattr(obj, 'id') and isinstance(obj.id, int):
        return (team, ObjectClass.ROBOT, obj.id)
    elif isinstance(obj, Ball): 
        return (TeamType.NEUTRAL, ObjectClass.BALL, 0)
    logger.warning(f"Could not determine ObjectKey for object of type {type(obj)} with team {team}")
    return None


# Helper to convert VectorObject to NumPy array
def _vector_to_numpy(vector: Union[VectorObject2D, VectorObject3D]) -> np.ndarray:
    if isinstance(vector, VectorObject2D):
        return np.array([vector.x, vector.y], dtype=np.float64)
    elif isinstance(vector, VectorObject3D):
        return np.array([vector.x, vector.y, vector.z], dtype=np.float64)
    raise TypeError(f"Unsupported vector type for NumPy conversion: {type(vector)}")


class PastGame:
    def __init__(self, max_history: int):
        self.max_history = max_history
        self.raw_games_history: deque[Game] = deque(maxlen=max_history)

        # Generic historical data storage:
        # ObjectKey -> AttributeType -> deque[(timestamp: float, value: np.ndarray)]
        self.historical_data: Dict[
            ObjectKey, Dict[AttributeType, deque[Tuple[float, np.ndarray]]]
        ] = {}

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
        value_vector_obj: Optional[Union[VectorObject2D, VectorObject3D]],
    ):
        """Adds a single attribute value (after converting to NumPy) to the history."""
        if value_vector_obj is None:
            return # Don't store None values, or decide on a specific handling

        self._ensure_attribute_deque_exists(object_key, attribute_type)
        try:
            value_np = _vector_to_numpy(value_vector_obj)
            self.historical_data[object_key][attribute_type].append((timestamp, value_np))
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


    def add_game(self, game: Game):
        self.raw_games_history.append(game)
        current_ts = game.ts

        # Process Ball
        if game.ball:
            ball_key = get_structured_object_key(game.ball, TeamType.NEUTRAL) # Ball is neutral
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

    def get_historical_attribute_series(
        self,
        object_key: ObjectKey,
        attribute_type: AttributeType,
        num_points: int,
    ) -> Tuple[np.ndarray, np.ndarray]: # Returns (timestamps_np, values_np)
        """
        Retrieves the last num_points of (timestamp, attribute_value_np) for a given object.
        Returns data as NumPy arrays (timestamps, values), oldest to newest.
        Returns empty NumPy arrays if no data is available.
        """
        if num_points <= 0:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

        object_attributes = self.historical_data.get(object_key)
        if not object_attributes:
            # logger.debug(f"No historical data for object_key {object_key}")
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

        history_deque = object_attributes.get(attribute_type)
        if not history_deque: # Handles both key not found or empty deque
            # logger.debug(f"No historical {attribute_type.name} for {object_key}")
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

        # Efficiently get the last num_points using islice
        start_index = max(0, len(history_deque) - num_points)
        relevant_data_iter = islice(history_deque, start_index, len(history_deque))

        timestamps_list: List[float] = []
        vector_values_list: List[np.ndarray] = []
        
        for ts, vec_np in relevant_data_iter:
            timestamps_list.append(ts)
            vector_values_list.append(vec_np)

        if not timestamps_list:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

        timestamps_np = np.array(timestamps_list, dtype=np.float64)
        # vector_values_list contains a list of small np.ndarrays.
        # np.array() will create a 2D array if all elements of the list are 1D arrays of the same size.
        values_np = np.array(vector_values_list)
        
        return timestamps_np, values_np

    def n_steps_ago(self, n: int) -> Game:
        if not (0 < n <= len(self.raw_games_history)):
            raise IndexError(
                f"Cannot get game {n} steps ago. History size: {len(self.raw_games_history)}, requested: {n}"
            )
        return self.raw_games_history[-n]