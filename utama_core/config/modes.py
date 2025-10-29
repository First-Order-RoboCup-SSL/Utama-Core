from enum import Enum


class Mode(Enum):
    """
    Environment modes for the robot soccer system.
    """

    RSIM = "rsim"
    GRSIM = "grsim"
    REAL = "real"


mode_str_to_enum = {m.value: m for m in Mode}
