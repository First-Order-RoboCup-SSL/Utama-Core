from entities.data.vector import Vector3D
from typing import Tuple
from dataclasses import dataclass


@dataclass(frozen=True)
class Ball:
    p: Vector3D
    v: Vector3D
    a: Vector3D
