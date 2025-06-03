import vector
from typing import Tuple
from dataclasses import dataclass


@dataclass(frozen=True)
class Ball:
    p: vector.VectorObject3D
    v: vector.VectorObject3D
    a: vector.VectorObject3D
