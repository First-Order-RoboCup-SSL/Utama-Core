from typing import Tuple, Optional

from entities.data.vision import BallData


class Ball:
    def __init__(self, ball_data: Optional[BallData] = None):
        self._ball_data: BallData = ball_data

    # note that ball position is stored in 3D coord: (x, y, z)
    @property
    def ball_data(self) -> BallData:
        return self._ball_data

    @ball_data.setter
    def ball_data(self, value: BallData):
        self._ball_data = value

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


if __name__ == "__main__":
    ball = Ball()
    print(ball.x)
    print(ball.y)
    print(ball.z)
    ball.ball_data = BallData(1, 2, 3, 1)
    print(ball.x)
    print(ball.y)
    print(ball.z)
