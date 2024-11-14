### NOT CURRENTLT USED ###

from typing import Tuple


class Ball:
    def __init__(self, pos: tuple = (0, 0, 0)):
        self._pos = pos

    # note that ball position is stored in 3D coord: (x, y, z)
    @property
    def pos(self) -> Tuple[float, float, float]:
        return self._pos

    @property
    def x(self) -> float:
        return self._pos[0]

    @property
    def y(self) -> float:
        return self._pos[1]

    @property
    def z(self) -> float:
        return self._pos[2]
