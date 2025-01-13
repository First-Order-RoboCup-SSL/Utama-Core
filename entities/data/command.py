from collections import namedtuple
from typing import NamedTuple

# velocities: meters per second
# angular_vel: radians per second
RobotCommand = namedtuple(
    "RobotCommand",
    [
        "local_forward_vel",
        "local_left_vel",
        "angular_vel",
        "kick",
        "chip",
        "dribble",
    ],
)

class RobotInfo(NamedTuple):
    has_ball: bool
# TODO: add kicker charged to robot info

