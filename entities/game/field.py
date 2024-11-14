import numpy as np


class Field:
    def __init__(self):
        self.HALF_LENGTH = 5.0  # x value
        self.HALF_WIDTH = 3.0  # y value
        self.HALF_GOAL_WIDTH = 1.0
        self.YELLOW_GOAL_POS = np.array(
            (self.HALF_LENGTH, self.HALF_WIDTH),
            (self.HALF_LENGTH, -self.HALF_WIDTH),
            dtype=np.float32,
        )
        self.BLUE_GOAL_POS = np.array(
            (-self.HALF_LENGTH, self.HALF_WIDTH),
            (-self.HALF_LENGTH, -self.HALF_WIDTH),
            dtype=np.float32,
        )

    @property
    def half_length(self) -> float:
        return self.HALF_LENGTH

    @property
    def half_width(self) -> float:
        return self.HALF_WIDTH

    @property
    def half_goal_width(self) -> float:
        return self.HALF_GOAL_WIDTH

    @property
    def yellow_goal_pos(self) -> np.array:
        return self.YELLOW_GOAL_POS

    @property
    def blue_goal_pos(self) -> np.array:
        return self.BLUE_GOAL_POS
