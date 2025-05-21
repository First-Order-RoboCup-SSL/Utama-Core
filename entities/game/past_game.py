from collections import deque
from entities.game.game import Game # Assuming Game, Robot, Ball types are defined
from vector import VectorObject2D, VectorObject3D # Or your vector types
from enum import Enum, auto
from typing import Tuple, Any, Union, Optional # For type hinting ObjectKey

# --- Enums for more robust identification ---
class AttributeType(Enum):
    POSITION = auto()
    VELOCITY = auto()
    # Add ACCELERATION here if you plan to store it directly in PastGame

class TeamType(Enum):
    FRIENDLY = auto()
    ENEMY = auto()
    NEUTRAL = auto() # For objects like the ball that don't belong to a team

class ObjectClass(Enum):
    ROBOT = auto()
    BALL = auto()

# Define a type alias for our structured object key
# (TeamType, ObjectClass, instance_id: int)
# For instance_id: robots will use their `id`, ball might use 0 or a specific constant.
ObjectKey = Tuple[TeamType, ObjectClass, int]


# --- Modified function to get structured object keys ---
def get_structured_object_key(obj: Any, team: TeamType) -> Optional[ObjectKey]:
    """
    Generates a structured key for game objects.
    `obj` is the game entity (e.g., a robot instance, ball instance).
    `team` is the TeamType enum member.
    """
    if hasattr(obj, 'id') and isinstance(obj.id, int): # For robots
        return (team, ObjectClass.ROBOT, obj.id)
    # Check if it's the ball (assuming ball object doesn't have 'id' like robots)
    # This check might need to be more specific based on your actual ball object type
    elif not hasattr(obj, 'id'): # A simple heuristic for the ball
        # Assuming only one ball, or use a specific ID if your ball object has one
        # Using 0 as a placeholder instance_id for the ball.
        return (TeamType.NEUTRAL, ObjectClass.BALL, 0)
    return None


class PastGame:
    def __init__(self, max_history: int):
        self.max_history = max_history
        self.raw_games_history = deque(maxlen=max_history)

        # Dictionaries now use ObjectKey as their key type
        self.historical_positions: dict[ObjectKey, deque[tuple[float, Union[VectorObject2D, VectorObject3D]]]] = {}
        self.historical_velocities: dict[ObjectKey, deque[tuple[float, Union[VectorObject2D, VectorObject3D]]]] = {}
        # self.historical_accelerations: dict[ObjectKey, deque[...]] = {} # If you add this

    def _ensure_object_history_exists(self, object_key: ObjectKey):
        """Ensures deques exist for the given object_key in all relevant history dicts."""
        if object_key not in self.historical_positions:
            self.historical_positions[object_key] = deque(maxlen=self.max_history)
        if object_key not in self.historical_velocities:
            self.historical_velocities[object_key] = deque(maxlen=self.max_history)
        # if object_key not in self.historical_accelerations:
        #     self.historical_accelerations[object_key] = deque(maxlen=self.max_history)

    def add_game(self, game: Game):
        self.raw_games_history.append(game)
        current_ts = game.ts

        # Extract and store data for the ball
        if game.ball:
            ball_key = get_structured_object_key(game.ball, TeamType.NEUTRAL) # Ball is neutral
            if ball_key:
                self._ensure_object_history_exists(ball_key)
                if hasattr(game.ball, 'p') and game.ball.p is not None:
                    self.historical_positions[ball_key].append((current_ts, game.ball.p))
                if hasattr(game.ball, 'v') and game.ball.v is not None:
                    self.historical_velocities[ball_key].append((current_ts, game.ball.v))

        # Extract and store data for friendly robots
        for robot in game.friendly_robots.values():
            robot_key = get_structured_object_key(robot, TeamType.FRIENDLY)
            if robot_key:
                self._ensure_object_history_exists(robot_key)
                if hasattr(robot, 'p') and robot.p is not None:
                    self.historical_positions[robot_key].append((current_ts, robot.p))
                if hasattr(robot, 'v') and robot.v is not None:
                    self.historical_velocities[robot_key].append((current_ts, robot.v))

        # Extract and store data for enemy robots
        for robot in game.enemy_robots.values():
            robot_key = get_structured_object_key(robot, TeamType.ENEMY)
            if robot_key:
                self._ensure_object_history_exists(robot_key)
                if hasattr(robot, 'p') and robot.p is not None:
                    self.historical_positions[robot_key].append((current_ts, robot.p))
                if hasattr(robot, 'v') and robot.v is not None:
                    self.historical_velocities[robot_key].append((current_ts, robot.v))

    def get_historical_attribute_series(
        self,
        object_key: ObjectKey,        # Now takes the structured ObjectKey
        attribute_key: AttributeType, # Now takes the AttributeType Enum
        num_points: int
    ) -> list[tuple[float, VectorObject2D | VectorObject3D]]:
        """
        Retrieves the last num_points of (timestamp, attribute_value) for a given object.
        Returns data from oldest to newest if available.
        """
        history_store_for_object = None

        if attribute_key == AttributeType.POSITION:
            history_store_for_object = self.historical_positions.get(object_key)
        elif attribute_key == AttributeType.VELOCITY:
            history_store_for_object = self.historical_velocities.get(object_key)
        # elif attribute_key == AttributeType.ACCELERATION:
        #     history_store_for_object = self.historical_accelerations.get(object_key)
        else:
            # This case should ideally not be reached if type hints are respected,
            # but good for robustness.
            raise ValueError(f"Unknown attribute_key: {attribute_key}")

        if not history_store_for_object: # Handles both key not found or empty deque
            # logger.warning(f"No historical {attribute_key.name} for {object_key}")
            return []

        # Deque stores [oldest, ..., newest]. We want the last num_points.
        start_index = max(0, len(history_store_for_object) - num_points)
        return list(history_store_for_object)[start_index:]


    def n_steps_ago(self, n) -> Game:
        if n <= 0 or n > len(self.raw_games_history):
            raise IndexError(f"Cannot get game {n} steps ago. History size: {len(self.raw_games_history)}")
        return self.raw_games_history[-n]