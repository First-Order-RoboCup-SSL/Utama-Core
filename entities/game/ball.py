from typing import Optional, Tuple

from entities.data.vision import BallData


class Ball:
    def __init__(self, ball_data: Optional[BallData] = None):
        self._ball_data = ball_data

    def __bool__(self):
        return self._ball_data is not None
    
    def __repr__(self):
        return f"Ball(x={self.x}, y={self.y}, z={self.z})"
    
    @property
    def coords(self) -> Tuple[float, float]:
        if self._ball_data is not None:
            return (self._ball_data.x, self._ball_data.y)
    
    @property
    def x(self) -> float:
        if self._ball_data is None:
            return None
        return self._ball_data.x

    @property
    def y(self) -> float:
        if self._ball_data is None:
            return None
        return self._ball_data.y

    @property
    def z(self) -> float:
        if self._ball_data is None:
            return None
        return self._ball_data.z

    def _update_ball_data(self, value):
        self._ball_data = value
