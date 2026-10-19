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

from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game

THIRDS = ("defensive", "middle", "attacking")
STRIPS = ("left", "centre", "right")
CELLS: tuple[str, ...] = tuple(f"{t}_{s}" for t in THIRDS for s in STRIPS)


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
