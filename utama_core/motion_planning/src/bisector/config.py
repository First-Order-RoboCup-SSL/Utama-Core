import logging
from math import dist, pi
from typing import Generator, List, Optional, Tuple, Union

import numpy as np  # type: ignore

from utama_core.config.physical_constants import ROBOT_RADIUS
from utama_core.config.robot_params.rsim import MAX_VEL as RSIM_MAX_VEL


class bisectorplannerconfig:
    ROBOT_DIAMETER = 2 * ROBOT_RADIUS
    CLOSE_LIMIT = 0.5
    SAMPLE_SIZE = 0.1
    MAX_VEL = RSIM_MAX_VEL
