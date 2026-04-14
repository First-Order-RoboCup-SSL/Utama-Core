from dataclasses import dataclass

from utama_core.entities.data.vector import Vector3D


@dataclass(frozen=True)
class Ball:
    p: Vector3D
    v: Vector3D
    a: Vector3D
