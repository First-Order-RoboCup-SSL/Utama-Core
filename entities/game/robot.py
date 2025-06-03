from __future__ import annotations
from dataclasses import dataclass
import logging
import vector

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Robot:
    id: int
    is_friendly: bool
    has_ball: bool  # Friendly and enemy now have this, friendly is from IR sensor, enemy from position
    p: vector.VectorObject2D
    v: vector.VectorObject2D
    a: vector.VectorObject2D
    orientation: float
