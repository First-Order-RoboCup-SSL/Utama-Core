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

import json
import math
from typing import Any, Optional

from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game
from utama_core.shared.pass_and_score_geometry import (
    enemy_goal_line,
    enemy_positions,
    find_best_shot,
    has_ball,
    segment_blocked,
)

THIRDS = ("defensive", "middle", "attacking")
STRIPS = ("left", "centre", "right")
CELLS: tuple[str, ...] = tuple(f"{t}_{s}" for t in THIRDS for s in STRIPS)

ACTION_GLOSSARY: dict[str, str] = {
    "shoot": "(ball carrier) shoot at the widest open part of their goal; if no shot is open yet, edge sideways "
    "with the ball until one opens",
    "pass:<id>": "(ball carrier) pass to that teammate; the teammate moves to receive it",
    "dribble:<cell>": "(ball carrier) carry the ball towards that cell (at most 1 m per touch)",
    "clear": "(ball carrier) kick the ball long and wide, upfield towards the nearer touch line, away from our goal",
    "get_ball": "go for the ball: collect a loose ball or challenge the opponent who has it",
    "move:<cell>": "run to open space in that cell (the best spot there for receiving a pass)",
    "mark:<id>": "stand between that opponent and our goal, so it can't receive or run at goal freely",
    "block_shot": "stand on the line between the ball and our goal, just outside our defense area",
    "hold": "stay where you are, facing the ball",
}

STATIC_STATE: dict[str, Any] = {
    "role": "You are the coach of our team in a 6v6 RoboCup small-size-league robot soccer match, and you "
    "control each of our 5 outfield robots directly. At each moment you choose one action for one robot at a "
    "time; the question lists what you already chose for the other robots at this moment, so make the robots "
    "work together as a team. Our goalkeeper (robot 0) plays by itself. Referee restarts (kick-offs, free "
    "kicks, penalties) are taken automatically: you only decide during open play.",
    "pitch": "Everything is from our point of view: we attack towards their goal. Thirds: defensive (nearest "
    "our goal), middle, attacking (nearest their goal). Strips: left, centre, right as we face their goal. "
    "A cell is a third and a strip, e.g. attacking_left.",
    "rules": "Only our goalkeeper may enter our own defense area, and none of our robots may enter theirs. A "
    "robot may carry the ball at most 1 m before it must pass, shoot or clear. A foul gives them a free kick. "
    "Aimless kick: a ball we kick from our own half that crosses their goal line without touching anyone gives "
    "them a free kick from where we kicked it.",
    "actions": ACTION_GLOSSARY,
    "coaching": [
        "Shots win matches here: the ball carrier shoots whenever its shot is open or narrow, from any range. "
        "It passes only when its shot is blocked, to a free teammate with an open lane, preferring one whose own "
        "shot is open; otherwise it dribbles into space, or clears when pressed in our own half.",
        "When they have the ball, two robots go for it (get_ball: the nearer blocks its shot, the other tackles), "
        "two block_shot, and one moves into an attacking cell, ready for the counter-attack.",
        "When we have the ball, two robots move into open attacking cells on different strips and two stay back "
        "on block_shot.",
        "A loose ball near our goal is always someone's to get.",
    ],
    "reading_the_state": "Distances are in words: touching (<0.3 m), very close (<0.7 m), close (<1.5 m), "
    "medium (<3 m), far. 'free' means no opponent within 0.7 m. Shots: open, narrow or blocked, and close "
    "(<2 m), medium (<4 m) or long range. `recent` lists the latest events, newest first.",
}

STATIC_TEXT = json.dumps({"static": STATIC_STATE}, ensure_ascii=False, separators=(",", ":")) + "\n"

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


def carrier(game: Game, robot_ids) -> Optional[int]:
    """Our robot with the ball on (or visibly at) its dribbler, nearest the ball if several."""
    holders = [rid for rid in robot_ids if rid in game.friendly_robots and has_ball(game, rid, visual=True)]
    if not holders:
        return None
    ball = game.ball.p.to_2d()
    return min(holders, key=lambda rid: game.friendly_robots[rid].p.distance_to(ball))


def enemy_keeper_id(game: Game) -> Optional[int]:
    ref = game.referee
    if ref is None:
        return None
    return (ref.blue_team if game.my_team_is_yellow else ref.yellow_team).goalkeeper
