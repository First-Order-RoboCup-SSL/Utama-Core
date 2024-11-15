from entities.game.field import Field
from entities.data.vision import FrameData


class Game:
    """
    Class containing states of the entire game and field information.
    """

    def __init__(self):
        self._field = Field()
        self._records = []
        self._yellow_score = 0
        self._blue_score = 0

    @property
    def field(self) -> Field:
        return self._field

    @property
    def current_state(self) -> FrameData:
        return self._records[-1] if self._records else None

    @property
    def records(self) -> list[FrameData]:
        return self._records

    @property
    def yellow_score(self) -> int:
        return self._yellow_score

    @property
    def blue_score(self) -> int:
        return self._blue_score

    def add_new_state(self, frame_data: FrameData) -> None:
        if isinstance(frame_data, FrameData):
            self._records.append(frame_data)
        else:
            raise ValueError("Invalid frame data.")
