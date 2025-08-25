from abc import ABC, abstractmethod
from typing import Type, TypeVar

import numpy as np

T = TypeVar("T", bound="VectorBase")


class VectorBase(ABC):
    def __init__(self, *coords):
        # allow for VectorBase(1, 2) or VectorBase((1, 2)) or VectorBase([1, 2])
        if len(coords) == 1 and isinstance(coords[0], (tuple, list, np.ndarray)):
            self._arr = np.array(coords[0])
        else:
            self._arr = np.array(coords)

    @property
    def x(self):
        return self._arr[0]

    @property
    def y(self):
        return self._arr[1]

    def mag(self) -> float:
        return np.linalg.norm(self._arr)

    def norm(self: T) -> T:
        """
        Normalize the vector to unit length.
        Returns a zero vector if the magnitude is too small.
        """
        magnitude = self.mag()
        if magnitude < 1e-8:
            return self.__class__.from_array(np.zeros_like(self._arr))
        return self.__class__.from_array(self._arr / magnitude)

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
        if isinstance(other, Vector2D):
            return np.hypot(other.y - self.y, other.x - self.x)
        else:
            return np.hypot(other[1] - self.y, other[0] - self.x)

    def to_array(self) -> np.ndarray:
        return self._arr

    def __add__(self: T, other: T) -> T:
        return self.__class__.from_array(self._arr + other._arr)

    def __sub__(self: T, other: T) -> T:
        return self.__class__.from_array(self._arr - other._arr)

    def __mul__(self: T, scalar: float) -> T:
        return self.__class__.from_array(self._arr * scalar)

    def __rmul__(self: T, scalar: float) -> T:
        return self.__mul__(scalar)

    def __truediv__(self: T, scalar: float) -> T:
        return self.__class__.from_array(self._arr / scalar)

    def __neg__(self: T) -> T:
        return self.__class__.from_array(-self._arr)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VectorBase):
            return NotImplemented
        return np.allclose(self._arr, other._arr)

    def __abs__(self) -> float:
        return self.mag()

    def __getitem__(self, idx: int) -> float:
        return self._arr[idx]

    def __iter__(self):
        return iter(self._arr)

    def __array__(self, dtype=None, copy=True):
        return np.array(self._arr, dtype=dtype, copy=copy)

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


if __name__ == "__main__":
    import numpy as np

    v2d = Vector2D(3, 4)
    v2d2 = Vector2D(1, 2)
    print(np.dot(v2d, v2d2))  # Should print 5.0

    print(v2d.angle_to(v2d2))  # Should print 0.5880026035475675
    print(np.arctan2(v2d2.y - v2d.y, v2d2.x - v2d.x))  # Should print 0.5880026035475675

    d = v2d2 - v2d
    print(np.linalg.norm(d))  # Should print 2.23606797749979
    print(v2d.distance_to(v2d2))  # Should print 2.23606797749979
