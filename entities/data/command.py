from collections import namedtuple

# velocities: meters per second
# angular_vel: radians per second
RobotCommand = namedtuple(
    "RobotCommand",
    [
        "local_forward_vel",
        "local_left_vel",
        "angular_vel",
        "kick_spd",
        "kick_angle",
        "dribbler_spd",
    ],
)
RobotInfo = namedtuple(
    "RobotInfo",
    ["has_ball"],
)
