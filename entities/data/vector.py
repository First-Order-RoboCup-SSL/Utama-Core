from typing import TypeVar, Type
import numpy as np
from abc import ABC, abstractmethod

T = TypeVar("T", bound="VectorBase")


class VectorBase(ABC):
    def __init__(self, *coords):
        self._arr = np.array(coords)

    @property
    def x(self):
        return self._arr[0]

    @property
    def y(self):
        return self._arr[1]

    def mag(self) -> float:
        return np.linalg.norm(self._arr)

    def angle_between(self, other: T) -> float:
        """
        2D: Calculate the angle between this vector and another vector in radians.
        """
        if self._arr.shape != other._arr.shape:
            raise ValueError(
                "Cannot calculate angle between vectors of different dimensions"
            )
        dot_product = np.dot(self._arr, other._arr)
        mag_self = self.mag()
        mag_other = other.mag()
        norm_prod = mag_self * mag_other
        if norm_prod == 0:
            return 0.0
        cos_theta = dot_product / norm_prod
        return np.arccos(np.clip(cos_theta, -1.0, 1.0))

    def angle_to(self, other: T) -> float:
        """
        2D: Calculate the angle from this vector to another vector in radians.
        """
        return np.arctan2(other.y - self.y, other.x - self.x)

    def distance_to(self, other: T) -> float:
        """
        2D: Calculate the distance to another vector.
        """
        return np.hypot(other.y - self.y, other.x - self.x)

    def to_array(self):
        return self._arr

    def __sub__(self: T, other: T) -> T:
        return self.__class__.from_array(self._arr - other._arr)  # type: ignore

    def to_array(self) -> np.ndarray:
        return self._arr

    @classmethod
    @abstractmethod
    def from_array(cls: Type[T], arr: np.ndarray) -> T: ...


class Vector2D(VectorBase):
    def __init__(self, x, y):
        super().__init__(x, y)

    @property
    def x(self):
        return super().x

    @property
    def y(self):
        return super().y

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "Vector2D":
        return cls(arr[0], arr[1])

    def __repr__(self):
        return f"Vector2D(x={self.x}, y={self.y})"


class Vector3D(VectorBase):
    def __init__(self, x, y, z):
        super().__init__(x, y, z)

    @property
    def x(self):
        return super().x

    @property
    def y(self):
        return super().y

    @property
    def z(self):
        return self._arr[2]

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "Vector3D":
        return cls(arr[0], arr[1], arr[2])

    def to_2d(self) -> Vector2D:
        return Vector2D(self.x, self.y)

    def __repr__(self):
        return f"Vector3D(x={self.x}, y={self.y}, z={self.z})"
