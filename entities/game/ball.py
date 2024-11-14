from typing import Tuple


class Ball:
    def __init__(self, pos: tuple = (0, 0)):
        self._pos = pos

    # note that ball position is stored in 3D coord: (x, y, z)
    @property
    def pos(self) -> Tuple[float, float, float]:
        return self._pos
