"""Flat OpenJev engine: the model controls every outfield robot directly, no tactics in between.

One `FlatJevTactic` owns all 5 outfield robots (the goalkeeper stays on its pinned slot, referee
restarts on `RefereeOverride`). At each decision it asks OpenJev one question per robot, chained in
the same moment: the ball carrier first, then the others nearest the ball first, each question
listing what was already chosen (see `flat_state.py`). A robot keeps its action until the next
decision, which comes at once when the situation changes (our carrier, possession, the ball's cell,
or a pass, shot or clearance ending), and at least every `1 / decision_hz` s of sim time (1 Hz:
the scripted rule plays as well at 1 Hz with these triggers as at 5 Hz).

Passes, shots and clearances are locked once chosen: the robots involved aren't asked again until
the kick happens or the action fails or times out, so a new decision can't abandon a half-aimed
pass.

`policy="scripted"` replaces the model with `scripted_choice`, a split_shape-like rule over the
same menu: it is the fallback when the server errors, and a model-free control that tells the
action layer's strength apart from the model's choices.

The sim is stepped synchronously, so inference costs wall time, never game time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game
from utama_core.shared.pass_and_score_geometry import (
    clamp_outside_enemy_defense_area,
    clamp_outside_own_defense_area,
    clamp_to_field,
    enemy_goal_line,
    enemy_positions,
    find_best_shot,
    segment_blocked,
)
from utama_core.strategy.openjev import flat_state as fs

LOCKED_ACTIONS = ("pass:", "receive", "shoot", "clear")


@dataclass
class Plan:
    action: str
    since: float
    target: Optional[Vector2D] = None  # move/dribble/hold: fixed when chosen, so the planner keeps one goal
    partner: Optional[int] = None  # pass: the receiver; receive: the passer
    kicked_at: Optional[float] = None  # receive: when the pass was kicked
    lost_since: Optional[float] = None


@dataclass
class FlatMem:
    plans: dict[int, Plan] = field(default_factory=dict)
    carry_origins: dict[int, Optional[Vector2D]] = field(default_factory=dict)
    shot_y: dict[int, Optional[float]] = field(default_factory=dict)
    last_decision_t: Optional[float] = None
    replan: bool = True
    last_carrier: Optional[int] = None
    last_key: Optional[tuple] = None  # (possession, ball cell) at the last tick


def is_locked(plan: Optional[Plan]) -> bool:
    return plan is not None and plan.action.startswith(LOCKED_ACTIONS)


def open_point(game: Game, cell: str, robot_id: int, taken: list[Vector2D]) -> Vector2D:
    """The best spot in `cell` to be available for a pass: an open lane from the ball, room from the
    nearest opponent, a shot from there in the attacking third, away from teammates' spots."""
    x_lo, x_hi, y_lo, y_hi = fs.cell_bounds(game, cell)
    inset = 0.35
    ball = game.ball.p.to_2d()
    enemies = enemy_positions(game)
    me = game.friendly_robots[robot_id].p
    attacking = cell.startswith("attacking")
    goal_x, y1, y2 = enemy_goal_line(game)
    best, best_score = None, -math.inf
    for i in range(3):
        for j in range(3):
            c = Vector2D(
                x_lo + inset + (x_hi - x_lo - 2 * inset) * i / 2.0,
                y_lo + inset + (y_hi - y_lo - 2 * inset) * j / 2.0,
            )
            c = clamp_to_field(clamp_outside_enemy_defense_area(game, clamp_outside_own_defense_area(game, c)), game)
            score = 1.5 * (not segment_blocked(ball, c, enemies)) + min(fs.nearest_enemy_dist(game, c), 1.5)
            if attacking:
                best_y, gap = find_best_shot(c, list(game.enemy_robots.values()), goal_x, y1, y2)
                score += 2.0 * ((gap[1] - gap[0]) if best_y is not None and gap is not None else 0.0)
            score -= 0.15 * c.distance_to(me)
            if any(c.distance_to(o) < 0.8 for o in taken):
                score -= 10.0
            if score > best_score:
                best, best_score = c, score
    return best
