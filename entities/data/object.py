from typing import NamedTuple
from enum import Enum, auto


class TeamType(Enum):
    FRIENDLY = auto()
    ENEMY = auto()
    NEUTRAL = auto()


class ObjectType(Enum):
    ROBOT = auto()
    BALL = auto()


class ObjectKey(NamedTuple):
    team_type: TeamType
    object_type: ObjectType
    id: int
