from collections import namedtuple

RSoccerSimCommand = namedtuple(
    "RSoccerSimCommand",
    ["glb_x_vel", "glb_y_vel", "angular_vel", "kick_vel", "dribbler_vel"],
)
