from collections import namedtuple


RefereeData = namedtuple(
    "RefereeData",
    ["t", "referee_command", "stage", "blue_team_info", "yellow_team_info"],
)
