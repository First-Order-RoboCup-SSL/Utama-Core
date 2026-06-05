from typing import Optional

from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game

_MIN_BALL_SPEED_X = 0.1  # m/s — below this, treat ball as stationary (filters real-hardware noise)


def predict_ball_pos_at_x(game: Game, x: float) -> Optional[Vector2D]:
    """Predict where the ball will cross a given x-coordinate.

    Returns None if the ball is not moving toward ``x`` (stationary along x,
    or moving away).  Callers should fall back to their own positioning logic
    (e.g. goalkeep uses shadow-based ``stop_y``).
    """
    vel = game.ball.v.to_2d()

    if abs(vel.x) < _MIN_BALL_SPEED_X:
        return None  # Ball not moving fast enough to predict reliably

    pos = game.ball.p.to_2d()
    t = (x - pos.x) / vel.x

    if t < 0:
        return None  # Ball moving away from target x

    y = pos.y + vel.y * t
    return Vector2D(x, y)
