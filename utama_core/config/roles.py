from enum import Enum, auto


class Role(Enum):
    GOALKEEPER = auto()
    DEFENDER = auto()
    STRIKER = auto()
    MIDFIELDER = auto()
    UNASSIGNED = auto()
