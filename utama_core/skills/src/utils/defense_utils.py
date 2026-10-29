from typing import NamedTuple

import numpy as np

from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game

CURVE_MARGIN = 0.1
CURVE_SHAPE_R = 2.1


class DefenseGeometry(NamedTuple):
    goal_x: float
    goal_half_width: float
    defense_front_x: float
    defense_depth: float
    defense_half_width: float


def _defense_geometry(game: Game) -> DefenseGeometry:
    """Derive all field-shaped defense values from ``game.field``."""
    goal_line = game.field.my_goal_line
    goal_x = float(goal_line[0][0])
    goal_half_width = abs(float(goal_line[0][1]))

    defense_area = game.field.my_defense_area
    defense_front_x = float(defense_area[1][0])
    defense_depth = abs(goal_x - defense_front_x)
    defense_half_width = abs(float(defense_area[0][1]))

    return DefenseGeometry(
        goal_x=goal_x,
        goal_half_width=goal_half_width,
        defense_front_x=defense_front_x,
        defense_depth=defense_depth,
        defense_half_width=defense_half_width,
    )


def calculate_defense_area(game: Game, t: float) -> Vector2D:
    """Defenders' path around the goal in the form of a rounded rectangle.

    around the penalty box. Ensures defenders don't go inside the box and stay
    right on the edge. This edge of the penalty box is theirs - attackers are
    not allowed this close to the box.

    x = a * ((1 - r) * |cos(t)| * cos(t) + r * cos(t))
    y = a * ((1 - r) * |sin(t)| * sin(t) + r * sin(t))

    https://www.desmos.com/calculator/nmaf7rpmnw
    """
    MIN_T = np.pi / 2
    MAX_T = 3 * np.pi / 2

    t = max(MIN_T, min(t, MAX_T))

    cos_t = np.cos(t)
    sin_t = np.sin(t)

    geo = _defense_geometry(game)
    front_extent = geo.defense_depth + CURVE_MARGIN
    vertical_extent = geo.defense_half_width + CURVE_MARGIN
    r = CURVE_SHAPE_R
    rp = Vector2D(
        front_extent * ((1 - r) * (abs(cos_t) * cos_t) + r * cos_t),
        vertical_extent * ((1 - r) * (abs(sin_t) * sin_t) + r * sin_t),
    )

    return make_relative_to_goal_centre(geo.goal_x, rp)


def make_relative_to_goal_centre(goal_centre_x: float, p: Vector2D) -> Vector2D:
    if goal_centre_x < 0:  # Left goal
        return Vector2D(goal_centre_x - p.x, p.y)
    else:
        return Vector2D(goal_centre_x + p.x, p.y)


def clamp_to_goal_width(y: float, goal_half_width: float) -> float:
    return max(min(y, goal_half_width), -goal_half_width)


# Repeated I will remove
# ---------------------------------------------------------------------------
# Shared shadow-geometry helpers (used by both goalkeep and defend_parameter)
# ---------------------------------------------------------------------------


def clamp_y(y: float, limit: float) -> float:
    """Clamp *y* to ``[-limit, limit]``."""
    return max(-limit, min(limit, y))


def intersection_with_x_line(a, b, target_x: float, y_limit: float):
    """Where line *a* -> *b* crosses ``x = target_x``, y clamped to +/-*y_limit*.

    Returns ``(target_x, clamped_y)``.  If the line is vertical (dx ~ 0),
    returns the y of *b* clamped.  If the crossing is behind *a* (t < 0),
    returns the y of *a* clamped.
    """
    xa, ya = a
    xb, yb = b
    dx = xb - xa

    if abs(dx) < 1e-12:
        return (target_x, clamp_y(yb, y_limit))

    t = (target_x - xa) / dx
    if t < 0:
        return (target_x, clamp_y(ya, y_limit))

    y_intersect = ya + t * (yb - ya)
    return (target_x, clamp_y(y_intersect, y_limit))


def single_defender_stop_y(
    ball_pos,
    defender_pos,
    keeper_x: float,
    post_limit: float,
    edge_offset: float,
) -> float:
    """Keeper y-position given a single defender's shadow on the keeper line.

    Projects both edges of the defender to find the shadow interval, then
    positions the keeper in the centre of the largest uncovered gap.
    """
    _, yy_top = intersection_with_x_line(
        (ball_pos.x, ball_pos.y),
        (defender_pos.x, defender_pos.y + edge_offset),
        keeper_x,
        post_limit,
    )
    _, yy_bottom = intersection_with_x_line(
        (ball_pos.x, ball_pos.y),
        (defender_pos.x, defender_pos.y - edge_offset),
        keeper_x,
        post_limit,
    )

    top_gap_size = max(0, post_limit - yy_top)
    bottom_gap_size = max(0, yy_bottom - (-post_limit))

    if top_gap_size > bottom_gap_size:
        return (yy_top + post_limit) / 2

    return (yy_bottom - post_limit) / 2
