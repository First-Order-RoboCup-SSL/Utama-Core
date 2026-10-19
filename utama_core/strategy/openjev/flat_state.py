"""Flat OpenJev: the 9 cells, each robot's action menu, and the state text the model reads.

Flat means no tactics in between: at each decision the model picks one action per outfield robot,
one robot at a time, and `flat.py` carries it out with existing skills. Questions in one decision are
chained: each robot's question lists what the robots asked before it chose, so the robots' actions
are decided jointly (p(a1..a5) = p(a1) p(a2|a1) ...), not independently.

Everything is egocentric (thirds and strips from our attacking point of view, names from
`docs/pitch_zones.md`) and given in words, not coordinates: OpenJev is weak at raw numbers. The
geometry it can't see in words (is a pass lane open, is a shot open) is computed here and stated as
a fact about each option; what to do about it is left to the model.
"""

from __future__ import annotations

import math
from typing import Optional

from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game
from utama_core.shared.pass_and_score_geometry import (
    enemy_goal_line,
    enemy_positions,
    find_best_shot,
    segment_blocked,
)

THIRDS = ("defensive", "middle", "attacking")
STRIPS = ("left", "centre", "right")
CELLS: tuple[str, ...] = tuple(f"{t}_{s}" for t in THIRDS for s in STRIPS)

FREE_RADIUS_M = 0.7
_NARROW_SHOT_M = 0.25  # open gap along their goal line below which a shot is "narrow"


def attack_dir(game: Game) -> float:
    return -1.0 if game.my_team_is_right else 1.0


def cell_coords(game: Game, p: Vector2D) -> tuple[int, int]:
    """(third, strip) indices of world point `p`, from our point of view."""
    a = attack_dir(game)
    half = game.field.half_length
    progress = p.x * a + half  # metres from our own goal line towards theirs
    third = min(2, max(0, int(progress / (2.0 * half / 3.0))))
    lateral = p.y * a  # facing their goal, left is +lateral
    edge = game.field.half_width / 3.0
    strip = 0 if lateral > edge else 2 if lateral < -edge else 1
    return third, strip


def cell_of(game: Game, p: Vector2D) -> str:
    third, strip = cell_coords(game, p)
    return CELLS[third * 3 + strip]


def cell_bounds(game: Game, cell: str) -> tuple[float, float, float, float]:
    """World (x_lo, x_hi, y_lo, y_hi) of `cell`."""
    k = CELLS.index(cell)
    third, strip = divmod(k, 3)
    a = attack_dir(game)
    half, width = game.field.half_length, game.field.half_width
    edge = width / 3.0
    p_lo, p_hi = third * 2.0 * half / 3.0, (third + 1) * 2.0 * half / 3.0
    l_lo, l_hi = [(edge, width), (-edge, edge), (-width, -edge)][strip]
    xs = sorted(((p_lo - half) * a, (p_hi - half) * a))
    ys = sorted((l_lo * a, l_hi * a))
    return xs[0], xs[1], ys[0], ys[1]


def adjacent_cells(cell: str) -> list[str]:
    """`cell` and its up-to-8 neighbours, in `CELLS` order."""
    third, strip = divmod(CELLS.index(cell), 3)
    return [c for i, c in enumerate(CELLS) if abs(i // 3 - third) <= 1 and abs(i % 3 - strip) <= 1]


def dist_word(d: Optional[float]) -> str:
    if d is None or not math.isfinite(d):
        return "far"
    return (
        "touching" if d < 0.3 else "very close" if d < 0.7 else "close" if d < 1.5 else "medium" if d < 3.0 else "far"
    )


def nearest_enemy_dist(game: Game, p: Vector2D) -> float:
    return min((r.p.distance_to(p) for r in game.enemy_robots.values()), default=math.inf)


def shot_word(game: Game, p: Vector2D) -> str:
    """'open, medium range' / 'narrow, close' / 'blocked, long range' for a shot from `p`."""
    goal_x, y1, y2 = enemy_goal_line(game)
    best_y, gap = find_best_shot(p, list(game.enemy_robots.values()), goal_x, y1, y2)
    width = (gap[1] - gap[0]) if (best_y is not None and gap is not None) else 0.0
    quality = "blocked" if best_y is None else "narrow" if width < _NARROW_SHOT_M else "open"
    d = p.distance_to(Vector2D(goal_x, (y1 + y2) / 2.0))
    rng = "close" if d < 2.0 else "medium range" if d < 4.0 else "long range"
    return f"{quality}, {rng}"


def lane_open(game: Game, start: Vector2D, end: Vector2D) -> bool:
    return not segment_blocked(start, end, enemy_positions(game))
