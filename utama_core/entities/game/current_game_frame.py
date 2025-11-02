from dataclasses import dataclass, field
from typing import Optional

from utama_core.entities.data.object import ObjectKey, ObjectType, TeamType
from utama_core.entities.game.game_frame import GameFrame
from utama_core.entities.game.proximity_lookup import ProximityLookup


@dataclass(frozen=True)
class CurrentGameFrame(GameFrame):
    robot_with_ball: Optional[ObjectKey] = field(init=False)
    proximity_lookup: ProximityLookup = field(init=False)

    def __init__(self, game: GameFrame):
        object.__setattr__(self, "ts", game.ts)
        object.__setattr__(self, "my_team_is_yellow", game.my_team_is_yellow)
        object.__setattr__(self, "my_team_is_right", game.my_team_is_right)
        object.__setattr__(self, "friendly_robots", game.friendly_robots)
        object.__setattr__(self, "enemy_robots", game.enemy_robots)
        object.__setattr__(self, "ball", game.ball)
        object.__setattr__(self, "robot_with_ball", self._set_robot_with_ball(game))
        object.__setattr__(self, "proximity_lookup", self._init_proximity_lookup(game))

    def _set_robot_with_ball(self, game: GameFrame) -> ObjectKey:
        """Initialize the robot_with_ball attribute based on the current state of the game.

        If no robot has the ball, return None.
        """
        # TODO: This assumes that only one robot can have the ball at a time.
        for robot in game.friendly_robots.values():
            if robot.has_ball:
                return ObjectKey(TeamType.FRIENDLY, ObjectType.ROBOT, robot.id)
        for robot in game.enemy_robots.values():
            if robot.has_ball:
                return ObjectKey(TeamType.ENEMY, ObjectType.ROBOT, robot.id)
        return None

    def _init_proximity_lookup(self, game: GameFrame) -> ProximityLookup:
        """Initialize the proximity map for the game.

        The proximity map contains the positions of all robots and the ball. It is used to lookup distances between
        objects in the game.
        """
        return ProximityLookup(
            friendly_robots=game.friendly_robots,
            enemy_robots=game.enemy_robots,
            ball=game.ball,
        )
