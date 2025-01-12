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
    [
        "has_ball",
        "stop_current_command",
        # Flag to control the action of the robot. Defaults to False. If true, the robot will stop.
        # TODO initialize it first (but I don't know where to do it)
    ],
)
