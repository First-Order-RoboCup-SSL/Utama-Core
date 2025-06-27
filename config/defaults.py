import numpy as np
from strategy.common.roles import Role

# Starting positions for right team
RIGHT_START_ONE = [
    (4.2000, 0.0, np.pi),
    (3.4000, -0.2000, np.pi),
    (3.4000, 0.2000, np.pi),
    (0.7000, 0.0, np.pi),
    (0.7000, 2.2500, np.pi),
    (0.7000, -2.2500, np.pi),
    (2.0000, 0.7500, np.pi),
    (2.0000, -0.7500, np.pi),
    (2.0000, 1.5000, np.pi),
    (2.0000, -1.5000, np.pi),
    (2.0000, 2.2500, np.pi),
]

# Starting positions for left team
LEFT_START_ONE = [
    (-4.2000, 0.0, 0),
    (-3.4000, 0.2000, 0),
    (-3.4000, -0.2000, 0),
    (-0.7000, 0.0, 0),
    (-0.7000, -2.2500, 0),
    (-0.7000, 2.2500, 0),
    (-2.0000, -0.7500, 0),
    (-2.0000, 0.7500, 0),
    (-2.0000, -1.5000, 0),
    (-2.0000, 1.5000, 0),
    (-2.0000, -2.2500, 0),
]

ATK_ROLE_MAP_ONE = {
    0: Role.GOALKEEPER,
    1: Role.DEFENDER,
    2: Role.DEFENDER,
    3: Role.MIDFIELDER,
    4: Role.STRIKER,
    5: Role.STRIKER,
}

DEF_ROLE_MAP_ONE = {
    0: Role.GOALKEEPER,
    1: Role.DEFENDER,
    2: Role.DEFENDER,
    3: Role.DEFENDER,
    4: Role.DEFENDER,
    5: Role.STRIKER,
}
