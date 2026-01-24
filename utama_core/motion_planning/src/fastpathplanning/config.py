import logging
from math import dist, pi
from typing import Generator, List, Optional, Tuple, Union

import numpy as np  # type: ignore

from utama_core.config.physical_constants import ROBOT_RADIUS


class fastpathplanningconfig:
    ROBOT_DIAMETER = 2 * ROBOT_RADIUS
    OBSTACLE_CLEARANCE = ROBOT_DIAMETER
    SUBGOAL_DISTANCE = ROBOT_DIAMETER * 3
    LOOK_AHEAD_RANGE = 3
    MAXRECURSION_LENGTH = 3
    PROJECTEDFRAMES = 30
