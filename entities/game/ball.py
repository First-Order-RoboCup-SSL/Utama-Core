from typing import Optional, Tuple

from entities.data.vision import BallData


class Ball:
    __game_update_token = object()
    
    def __init__(self, ball_data: Optional[BallData] = None):
        self._ball_data = ball_data

    def __bool__(self):
        return self._ball_data is not None
    
    def __repr__(self):
        return f"Ball(x={self.x}, y={self.y}, z={self.z})"
    
    # note that ball position is stored in 3D coord: (x, y, z)
    @property
    def ball_data(self) -> BallData:
        return self._ball_data

    @ball_data.setter
    def ball_data(self, value):
        """
        Private setter for ball_data.
        Expects a tuple: (BallData, token).
        """
        if not isinstance(value, tuple) or len(value) != 2:
            raise PermissionError("Direct assignment to ball_data is not allowed. Use the proper update mechanism.")
        data, token = value
        if token is not Ball.__game_update_token:
            raise PermissionError("Only Game is allowed to update ball data.")
        # Bypass the property to set the raw value
        self._ball_data = data

    @property
    def coords(self) -> Tuple[float, float]:
        if self._ball_data is not None:
            return self._ball_data[:2]
    
    @property
    def x(self) -> float:
        if self._ball_data is None:
            return None
        return self._ball_data.x

    @property
    def y(self) -> float:
        if self._ball_data is None:
            return None
        return self._ball_data.y

    @property
    def z(self) -> float:
        if self._ball_data is None:
            return None
        return self._ball_data.z
    
    @classmethod
    def _get_game_update_token(cls):
        """
        Returns the token needed to update ball data.
        By convention, only the Game class should use this.
        """
        return cls.__game_update_token
