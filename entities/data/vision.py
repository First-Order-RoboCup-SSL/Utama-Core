from collections import namedtuple

# position data: meters
# orientation: radians
BallData = namedtuple("BallData", ["x", "y", "z"])
RobotData = namedtuple("RobotData", ["x", "y", "orientation"])
FrameData = namedtuple("FrameData", ["ts", "yellow_robots", "blue_robots", "ball"])
TeamRobotCoords = namedtuple("TeamRobotCoords", ["team_color", "robots"])
