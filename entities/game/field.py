import numpy as np


class Field:
    def __init__(self):
        self.HALF_LENGTH = 5.0  # x value
        self.HALF_WIDTH = 3.0  # y value
        self.HALF_GOAL_WIDTH = 1.0
        self.YELLOW_GOAL_POS = np.array(
            [self.HALF_LENGTH, self.HALF_WIDTH],
            [self.HALF_LENGTH, -self.HALF_WIDTH],
            dtype=np.float32,
        )
        self.BLUE_GOAL_POS = np.array(
            [-self.HALF_LENGTH, self.HALF_WIDTH],
            [-self.HALF_LENGTH, -self.HALF_WIDTH],
            dtype=np.float32,
        )
