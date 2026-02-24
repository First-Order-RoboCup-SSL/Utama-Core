from enum import Enum, auto


class Mode(Enum):
    """
    Environment modes for the robot soccer system.
    """

    RSIM = "rsim"
    GRSIM = "grsim"
    REAL = "real"
    VMAS = "vmas"


mode_str_to_enum = {m.value: m for m in Mode}


class Role(Enum):
    GOALKEEPER = auto()
    DEFENDER = auto()
    STRIKER = auto()
    MIDFIELDER = auto()
    UNASSIGNED = auto()


class Tactic(Enum):
    ATTACKING = auto()
    DEFENDING = auto()
