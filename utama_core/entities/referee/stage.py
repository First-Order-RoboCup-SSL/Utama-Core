from enum import Enum


class Stage(Enum):
    """Enum representing a game stage."""

    NORMAL_FIRST_HALF_PRE = 0
    NORMAL_FIRST_HALF = 1
    NORMAL_HALF_TIME = 2
    NORMAL_SECOND_HALF_PRE = 3
    NORMAL_SECOND_HALF = 4
    EXTRA_TIME_BREAK = 5
    EXTRA_FIRST_HALF_PRE = 6
    EXTRA_FIRST_HALF = 7
    EXTRA_HALF_TIME = 8
    EXTRA_SECOND_HALF_PRE = 9
    EXTRA_SECOND_HALF = 10
    PENALTY_SHOOTOUT_BREAK = 11
    PENALTY_SHOOTOUT = 12
    POST_GAME = 13

    @staticmethod
    def from_id(stage_id: int):
        try:
            return Stage(stage_id)
        except ValueError:
            raise ValueError(f"Invalid stage ID: {stage_id}")

    @property
    def stage_id(self):
        return self.value
