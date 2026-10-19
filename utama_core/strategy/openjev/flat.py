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

from dataclasses import dataclass, field
from typing import Optional

from utama_core.entities.data.vector import Vector2D

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
