"""`examples/ball_trials.py`, the ball part of `pixi run calibrate`: each carry is the distance it
says from standing to standing, runs towards the middle of the field, and the ladder stops a
direction at the first speed that loses the ball."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import FieldBounds

EXAMPLES = Path(__file__).resolve().parents[3] / "examples"
sys.path.insert(0, str(EXAMPLES))
try:
    import ball_trials as ex
finally:
    sys.path.remove(str(EXAMPLES))

BOUNDS = FieldBounds(top_left=(-1.5, 1.125), bottom_right=(1.5, -1.125))


@pytest.mark.parametrize("speed", [0.2, 1.0, 4.0])
def test_a_carry_covers_its_distance_and_ends_standing(speed):
    total = ex.move_seconds(0.5, speed, 4.0)
    dt = 1e-4
    covered = sum(ex.speed_at(i * dt, 0.5, speed, 4.0) * dt for i in range(int(total / dt) + 1))

    assert covered == pytest.approx(0.5, abs=1e-3)
    assert ex.speed_at(0.0, 0.5, speed, 4.0) == 0.0
    assert ex.speed_at(total, 0.5, speed, 4.0) == pytest.approx(0.0, abs=1e-9)
    assert max(ex.speed_at(i * dt, 0.5, speed, 4.0) for i in range(int(total / dt))) <= speed


@pytest.mark.parametrize("direction", ex.DIRECTIONS[:4], ids=lambda d: d.name)
def test_a_carry_runs_towards_the_middle(direction):
    ball = Vector2D(1.0, -0.6)
    heading = ex.carry_heading(ball, BOUNDS, direction)
    travel = heading + direction.angle

    assert math.cos(travel - math.atan2(0.6, -1.0)) == pytest.approx(1.0)


def test_kicks_go_along_the_field_towards_the_further_end():
    assert ex.kick_heading(Vector2D(-0.4, 0.3), BOUNDS) == 0.0
    assert ex.kick_heading(Vector2D(0.4, 0.3), BOUNDS) == math.pi


@pytest.mark.parametrize("x, reach", [(0.0, True), (1.5 - ex.EDGE_M - 0.01, True), (1.5 - ex.EDGE_M + 0.01, False)])
def test_a_ball_near_the_edge_is_asked_for_back(x, reach):
    from types import SimpleNamespace

    game = SimpleNamespace(ball=SimpleNamespace(p=Vector2D(x, 0.0)))
    assert ex.in_reach(game, BOUNDS) is reach


def test_the_ladder_stops_a_direction_at_its_first_loss_and_reports_the_speed_below():
    slow = ex.Direction("forward", 0.0, 0, (0.2, 0.3, 0.5))
    turn = ex.Direction("turn anticlockwise", 0.0, 1, (1.0, 2.0))
    ladder = ex.Ladder((slow, turn), repeats=2)

    for held in (True, True, True, False):  # 0.2 twice, then 0.3 loses on its second trial
        ladder.record(held, 0.0)
    assert ladder.current()[:2] == (turn, 1.0)  # 0.5 is never tried
    for held in (True, True, True, True):
        ladder.record(held, 0.0)

    assert ladder.done
    assert ladder.summary() == {"forward": 0.2, "turn anticlockwise": 2.0}
    assert [r["speed"] for r in ladder.results if r["direction"] == "forward"] == [0.2, 0.2, 0.3, 0.3]


def test_a_direction_losing_at_its_slowest_speed_carries_nothing():
    ladder = ex.Ladder((ex.Direction("left", math.pi / 2, 0, (0.2, 0.3)),), repeats=1)
    ladder.record(False, 0.0)
    assert ladder.done and ladder.summary() == {"left": None}
