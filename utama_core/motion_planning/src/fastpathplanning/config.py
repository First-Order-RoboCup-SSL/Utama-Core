from utama_core.config.physical_constants import ROBOT_RADIUS


class fastpathplanningconfig:
    ROBOT_DIAMETER = 2 * ROBOT_RADIUS

    # how fat is the danger zone around obstacles, in multiples of robot diameter
    CLEARANCE_MULTIPLIER = 1.5
    OBSTACLE_CLEARANCE = ROBOT_DIAMETER * CLEARANCE_MULTIPLIER

    # How far outside the danger zone shold the waypoint be
    SUBGOAL_MULTIPLIER = 1.2
    SUBGOAL_DISTANCE = OBSTACLE_CLEARANCE * SUBGOAL_MULTIPLIER

    LOOK_AHEAD_RANGE = 3
    MAXRECURSION_LENGTH = 3
    PROJECTEDFRAMES = 20
    PROJECTION_DISTANCE = 1
