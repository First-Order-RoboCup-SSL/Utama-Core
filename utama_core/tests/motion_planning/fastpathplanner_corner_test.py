"""FastPathPlanner around the enemy defense-area corner, with blockers.

Restart stalls keep coming from one geometry: a robot in front of the enemy box
face heading for a ball just past the box corner (a free-kick approach point),
with other robots parked near that corner. Rather than one hand-built case per
stall, this drives the real `_path_to` over seeded random scenarios of that shape
with a simple kinematic robot (moves at most 3 m/s toward the returned carrot each
60 Hz tick) and checks:

- safety, every scenario: the carrot is always finite and the robot never enters
  the real enemy defense area;
- arrival: the robot gets within 0.1 m of the target within 10 s. Scenarios that
  currently fail are pinned as strict xfails. They are FPP's known local minimum
  (roadmap item 5): the detour side flips every tick between two positions when
  the gap between a blocker and the inflated box is narrower than the clearance,
  because the side memory is keyed on the obstacle hit point, which changes
  between the two. A planner fix turns them into XPASS, which fails the run so
  this list gets updated.

Every blocker keeps the route around the outside open, so all targets are reachable.
"""

from __future__ import annotations

import functools
import math
import random

import numpy as np
import pytest

from utama_core.motion_planning.src.fastpathplanning.planner import FastPathPlanner
from utama_core.tests.fixtures.game_builder import build_game, make_ball, make_robot

_STEP_M = 3.0 / 60
_TICKS = 600
_ARRIVED_M = 0.1
# Real enemy defense area with my_team_is_right=False (enemy goal at +x).
_BOX = (3.5, 4.5, -1.0, 1.0)
_SEEDS = range(40)
_KNOWN_STUCK = {0, 19, 26, 30}


def _in_box(x: float, y: float) -> bool:
    return _BOX[0] < x < _BOX[1] and _BOX[2] < y < _BOX[3]


def _scenario(seed: int):
    rng = random.Random(seed)
    side = rng.choice([-1, 1])
    ball = (rng.uniform(3.6, 4.4), side * rng.uniform(1.3, 1.9))
    target = (ball[0] + 0.1, ball[1] + side * 0.03)
    start = (rng.uniform(2.6, 3.2), side * rng.uniform(0.0, 1.2))
    blockers: list[tuple[float, float]] = []
    for _ in range(rng.randint(1, 3)):
        for _attempt in range(50):
            b = (3.25 + rng.uniform(-0.35, 0.35), side * (1.25 + rng.uniform(-0.35, 0.35)))
            if (
                math.dist(b, target) > 0.35
                and math.dist(b, start) > 0.15
                and not _in_box(*b)
                and all(math.dist(b, o) > 0.2 for o in blockers)
            ):
                blockers.append(b)
                break
    return ball, target, start, blockers


@functools.lru_cache(maxsize=None)
def _drive(seed: int) -> dict:
    ball, target, start, blockers = _scenario(seed)
    planner = FastPathPlanner(env=None)
    pos = np.array(start, dtype=float)
    goal = np.array(target)
    result = {"finite": True, "entered_box": False, "arrived": False}
    for tick in range(_TICKS):
        friendly = {1: make_robot(1, is_friendly=True, x=pos[0], y=pos[1])}
        enemy = {i: make_robot(i, is_friendly=False, x=x, y=y) for i, (x, y) in enumerate(blockers)}
        game = build_game(friendly, enemy, make_ball(*ball), my_team_is_right=False, ts=tick / 60)
        carrot = np.asarray(planner._path_to(game, 1, target, game.field.field_bounds), dtype=float)
        if not np.all(np.isfinite(carrot)):
            result["finite"] = False
            return result
        step = carrot - pos
        length = float(np.linalg.norm(step))
        if length > 1e-9:
            pos = pos + step / length * min(length, _STEP_M)
        if _in_box(*pos):
            result["entered_box"] = True
            return result
        if np.linalg.norm(pos - goal) < _ARRIVED_M:
            result["arrived"] = True
            return result
    return result


@pytest.mark.parametrize("seed", _SEEDS)
def test_corner_route_is_safe(seed):
    result = _drive(seed)
    assert result["finite"], "planner returned a NaN/inf carrot"
    assert not result["entered_box"], "robot entered the enemy defense area"


@pytest.mark.parametrize(
    "seed",
    [
        (
            pytest.param(s, marks=pytest.mark.xfail(strict=True, reason="FPP detour-side flip (roadmap item 5)"))
            if s in _KNOWN_STUCK
            else s
        )
        for s in _SEEDS
    ],
)
def test_corner_route_arrives(seed):
    assert _drive(seed)["arrived"]
