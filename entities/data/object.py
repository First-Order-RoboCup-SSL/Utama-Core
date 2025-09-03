from enum import Enum, auto
from typing import NamedTuple


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
