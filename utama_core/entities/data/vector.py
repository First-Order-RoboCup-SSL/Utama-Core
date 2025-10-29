import math
from abc import ABC, abstractmethod
from typing import Type, TypeVar

import numpy as np

T = TypeVar("T", bound="VectorBase")


class VectorBase(ABC):
    __slots__ = ("x", "y")

    @abstractmethod
    def mag(self) -> float:
        """
        Calculate the magnitude of the vector.
        """
        ...

    def norm(self: T) -> T:
        """
        Return a normalized copy of the vector. Returns zero vector if magnitude is too small.
        """
        ...

    def dot(self, other: T) -> float:
        """2D: Dot product with another vector."""
        return self.x * other.x + self.y * other.y

    def angle_between(self, other: T) -> float:
        """2D: Angle between self and other vector using only x and y."""
        dx1, dy1 = self.x, self.y
        dx2, dy2 = other.x, other.y

        dot = dx1 * dx2 + dy1 * dy2
        mag1 = math.hypot(dx1, dy1)
        mag2 = math.hypot(dx2, dy2)

        if mag1 == 0 or mag2 == 0:
            return 0.0

        cos_theta = max(-1.0, min(1.0, dot / (mag1 * mag2)))
        return math.acos(cos_theta)

    def angle_to(self, other: T) -> float:
        """
        2D: Calculate the angle from this vector to another vector in radians.
        """
        return math.atan2(other.y - self.y, other.x - self.x)

    def distance_to(self, other: T) -> float:
        """
        2D: Calculate the distance to another vector.
        """
        return math.hypot(other.y - self.y, other.x - self.x)

    def to_array(self) -> np.ndarray:
        return np.array(list(self))

    def __abs__(self) -> float:
        return self.mag()


class Vector2D(VectorBase):
    __slots__ = ()

    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)

    def __iter__(self):
        yield self.x
        yield self.y

    def __getitem__(self, index: int) -> float:
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        raise IndexError("Vector2D index out of range")

    def __add__(self, other: "Vector2D") -> "Vector2D":
        return Vector2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vector2D") -> "Vector2D":
        return Vector2D(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> "Vector2D":
        return Vector2D(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar: float) -> "Vector2D":
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> "Vector2D":
        return Vector2D(self.x / scalar, self.y / scalar)

    def __neg__(self) -> "Vector2D":
        return Vector2D(-self.x, -self.y)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vector2D):
            return NotImplemented
        return math.isclose(self.x, other.x) and math.isclose(self.y, other.y)

    def __array__(self, dtype=None, copy=True):
        return np.array([self.x, self.y], dtype=dtype, copy=copy)

    def mag(self) -> float:
        return math.hypot(self.x, self.y)

    def norm(self) -> "Vector2D":
        """Return a normalized copy of the vector. Returns zero vector if magnitude is too small."""
        magnitude = self.mag()
        if magnitude < 1e-8:
            return Vector2D(0.0, 0.0)
        return Vector2D(self.x / magnitude, self.y / magnitude)

    def __repr__(self):
        return f"Vector2D(x={self.x}, y={self.y})"


class Vector3D(VectorBase):
    __slots__ = ("z",)

    def __init__(self, *coords):
        self.x = float(coords[0])
        self.y = float(coords[1])
        self.z = float(coords[2])

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z

    def __getitem__(self, index: int) -> float:
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        elif index == 2:
            return self.z
        raise IndexError("Vector3D index out of range")

    def __add__(self, other: "Vector3D") -> "Vector3D":
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vector3D") -> "Vector3D":
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> "Vector3D":
        return Vector3D(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: float) -> "Vector3D":
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> "Vector3D":
        return Vector3D(self.x / scalar, self.y / scalar, self.z / scalar)

    def __neg__(self) -> "Vector3D":
        return Vector3D(-self.x, -self.y, -self.z)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vector3D):
            return NotImplemented
        return math.isclose(self.x, other.x) and math.isclose(self.y, other.y) and math.isclose(self.z, other.z)

    def __array__(self, dtype=None, copy=True):
        return np.array([self.x, self.y, self.z], dtype=dtype, copy=copy)

    def mag(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def norm(self) -> "Vector3D":
        """Return a normalized copy of the vector. Returns zero vector if magnitude is too small."""
        magnitude = self.mag()
        if magnitude < 1e-8:
            return Vector3D(0.0, 0.0, 0.0)
        return Vector3D(self.x / magnitude, self.y / magnitude, self.z / magnitude)

    def to_2d(self) -> Vector2D:
        return Vector2D(self.x, self.y)

    def __repr__(self):
        return f"Vector3D(x={self.x}, y={self.y}, z={self.z})"


if __name__ == "__main__":
    import numpy as np

    v2d = Vector2D(3, 4)
    v2d2 = Vector2D(1, 2)
    print(np.dot(v2d, v2d2))  # Should print 11.0
    print(v2d.dot(v2d2))  # Should print 11.0

    print(v2d.angle_to(v2d2))  # Should print -2.356194490192345
    print(np.arctan2(v2d2.y - v2d.y, v2d2.x - v2d.x))  # Should print -2.356194490192345

    d = v2d2 - v2d
    print(np.linalg.norm(d))  # Should print 2.8284271247461903
    print(v2d.distance_to(v2d2))  # Should print 2.8284271247461903
