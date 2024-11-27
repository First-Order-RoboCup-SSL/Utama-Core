from typing import List
from entities.game.field import Field
from entities.data.vision import FrameData, RobotData, BallData


class Game:
    """
    Class containing states of the entire game and field information.
    """

    def __init__(self):
        self._field = Field()
        self._records = []
        self._yellow_score = 0
        self._blue_score = 0

    def add_new_state(self, frame_data: FrameData) -> None:
        if isinstance(frame_data, FrameData):
            self._records.append(frame_data)
        else:
            raise ValueError("Invalid frame data.")

    def get_robots_pos(self, is_yellow: bool) -> List[RobotData]:
        if not self._records:
            return None
        record = self._records[-1]
        return record.yellow_robots if is_yellow else record.blue_robots

    def get_ball_pos(self) -> BallData:
        if not self._records:
            return None
        return self._records[-1].ball

    def get_latest_frame(self) -> FrameData:
        if not self._records:
            return None
        return self._records[-1]

    @property
    def field(self) -> Field:
        return self._field

    @property
    def current_state(self) -> FrameData:
        return self._records[-1] if self._records else None

    @property
    def records(self) -> list[FrameData]:
        if not self._records:
            return None
        return self._records

    @property
    def yellow_score(self) -> int:
        return self._yellow_score

    @property
    def blue_score(self) -> int:
        return self._blue_score
