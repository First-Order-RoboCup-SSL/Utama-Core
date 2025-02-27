from typing import Tuple

from dataclasses import dataclass


@dataclass(frozen=True)
class Ball:
    x: float
    y: float
    z: float

    @property
    def coords(self) -> Tuple[float, float]:
        return self.x, self.y
