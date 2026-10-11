"""Geometry and dispatch of split_wall's opponent direct-free-kick override.

Pins the wall's exact distance and spacing (on both field sides), the marker and
leftover positions, and that only the OPPONENT's direct free kick gets the wall: our
own kick must run the built-in DirectFreeOursStep unchanged.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from utama_core.config.field_params import STANDARD_FIELD_DIMS
from utama_core.engine.referee_override import RefereeOverride
from utama_core.entities.data.vector import Vector2D, Vector3D
from utama_core.entities.game import Ball, Field, Game, GameFrame, GameHistory, Robot
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.shared.pass_and_score_geometry import _DEFENSE_AREA_CLAMP_MARGIN
from utama_core.strategy import split_wall
from utama_core.strategy.split_wall import (
    WALL_DISTANCE_M,
    WALL_SPACING_M,
    leftover_points,
    marker_point,
    wall_points,
)


def _robot(rid, x, y, is_friendly, orientation=0.0):
    return Robot(
        id=rid,
        is_friendly=is_friendly,
        has_ball=False,
        p=Vector2D(x, y),
        v=Vector2D(0, 0),
        a=Vector2D(0, 0),
        orientation=orientation,
    )


def _game(ball_xy, friendly, enemy, my_team_is_right=True, my_team_is_yellow=True):
    zv = Vector3D(0, 0, 0)
    frame = GameFrame(
        ts=0.0,
        my_team_is_yellow=my_team_is_yellow,
        my_team_is_right=my_team_is_right,
        friendly_robots={rid: _robot(rid, x, y, True) for rid, (x, y) in friendly.items()},
        enemy_robots={rid: _robot(rid, x, y, False) for rid, (x, y) in enemy.items()},
        ball=Ball(p=Vector3D(ball_xy[0], ball_xy[1], 0), v=zv, a=zv),
    )
    return Game(
        past=GameHistory(10),
        current=frame,
        field=Field(
            my_team_is_right=my_team_is_right,
            field_dims=STANDARD_FIELD_DIMS,
            field_bounds=STANDARD_FIELD_DIMS.full_field_bounds,
        ),
    )


def _along(p: Vector2D, start: Vector2D, end: Vector2D) -> float:
    """Signed distance of p along the start->end direction."""
    length = math.hypot(end.x - start.x, end.y - start.y)
    return ((p.x - start.x) * (end.x - start.x) + (p.y - start.y) * (end.y - start.y)) / length


def _across(p: Vector2D, start: Vector2D, end: Vector2D) -> float:
    """Signed distance of p across the start->end direction."""
    length = math.hypot(end.x - start.x, end.y - start.y)
    return ((p.x - start.x) * (end.y - start.y) - (p.y - start.y) * (end.x - start.x)) / length


# ---- wall geometry -------------------------------------------------------


def test_wall_sits_on_the_ball_to_goal_line_at_the_keep_out_distance_on_our_right_side():
    ball, goal = Vector2D(0.0, 0.0), Vector2D(4.5, 0.0)
    left, right = wall_points(ball, goal, 2)
    # Both on the x = 0.9 line, 0.2 m apart, symmetric about the ball->goal line.
    assert WALL_DISTANCE_M == pytest.approx(0.9)
    assert left.x == pytest.approx(0.9) and right.x == pytest.approx(0.9)
    assert left.y == pytest.approx(-0.1) and right.y == pytest.approx(0.1)
    assert right.y - left.y == pytest.approx(WALL_SPACING_M)


def test_wall_mirrors_on_the_left_side_where_our_goal_is_at_negative_x():
    ball, goal = Vector2D(0.0, 0.0), Vector2D(-4.5, 0.0)
    pts = sorted(wall_points(ball, goal, 2), key=lambda p: p.y)
    assert [p.x for p in pts] == pytest.approx([-0.9, -0.9])
    assert [p.y for p in pts] == pytest.approx([-0.1, 0.1])


def test_wall_distance_is_measured_along_an_off_axis_ball_to_goal_line():
    ball, goal = Vector2D(1.0, 1.0), Vector2D(4.5, 0.0)
    pts = wall_points(ball, goal, 2)
    assert [_along(p, ball, goal) for p in pts] == pytest.approx([WALL_DISTANCE_M] * 2)
    assert sorted(_across(p, ball, goal) for p in pts) == pytest.approx([-WALL_SPACING_M / 2, WALL_SPACING_M / 2])


def test_wall_clears_the_0_5_m_foul_line_with_margin():
    # The referee's keep-out foul is 0.5 m; the wall must sit beyond it.
    ball, goal = Vector2D(0.0, 0.0), Vector2D(4.5, 0.0)
    for p in wall_points(ball, goal, 2):
        assert p.distance_to(ball) > 0.5 + 0.3


# ---- marking and leftover geometry ---------------------------------------


def test_marker_stands_half_a_metre_goal_side_of_its_man_on_the_man_to_goal_line():
    man, goal = Vector2D(2.0, 1.0), Vector2D(4.5, 0.0)
    mark = marker_point(man, goal)
    assert mark.distance_to(man) == pytest.approx(0.5)
    assert _across(mark, man, goal) == pytest.approx(0.0, abs=1e-9)
    assert mark.distance_to(goal) < man.distance_to(goal)


def test_leftovers_sit_on_the_defense_area_front_edge_where_the_ball_goal_line_crosses_it():
    ball, goal = Vector2D(0.0, 0.0), Vector2D(4.5, 0.0)
    (single,) = leftover_points(ball, goal, 3.5, 1)
    assert single.x == pytest.approx(3.5)
    assert single.y == pytest.approx(0.0)
    two = leftover_points(ball, goal, 3.5, 2)
    assert two[1].y - two[0].y == pytest.approx(split_wall.LEFTOVER_SPACING_M)
    assert all(p.x == pytest.approx(3.5) for p in two)


# ---- full target assignment ----------------------------------------------


def test_opponent_kick_assigns_wall_markers_and_leftover_on_our_right_side():
    # Kicker: enemy 0 is nearest the ball. Enemy 1 is nearest our goal, enemy 2 next.
    game = _game(
        (0.0, 0.0),
        friendly={1: (0.5, 0.1), 2: (0.5, -0.2), 3: (-2.0, 1.0), 4: (-2.5, -1.0), 5: (-1.0, 0.0)},
        enemy={0: (0.1, 0.0), 1: (3.0, 1.0), 2: (1.0, -2.0)},
    )
    targets = split_wall.opponent_free_kick_targets(game, (1, 2, 3, 4, 5))

    assert set(targets) == {1, 2, 3, 4, 5}
    # Wall: robot 2 (lower side) takes y=-0.1, robot 1 takes y=+0.1.
    assert targets[2][0].x == pytest.approx(0.9) and targets[2][0].y == pytest.approx(-0.1)
    assert targets[1][0].x == pytest.approx(0.9) and targets[1][0].y == pytest.approx(0.1)
    # Markers: robot 5 marks enemy 1, robot 4 marks enemy 2.
    assert targets[5][0].distance_to(game.enemy_robots[1].p) < 1.0
    assert targets[4][0].distance_to(game.enemy_robots[2].p) < 1.0
    # Leftover robot 3 goes to the front edge, pushed out of the box by the clamp margin.
    front_x = float(game.field.my_defense_area[1][0])
    assert targets[3][0].x == pytest.approx(front_x - _DEFENSE_AREA_CLAMP_MARGIN)
    assert targets[3][0].y == pytest.approx(0.0)


def test_opponent_kick_mirrors_on_the_left_side():
    game = _game(
        (0.0, 0.0),
        friendly={1: (-0.5, 0.1), 2: (-0.5, -0.2), 3: (2.0, 1.0)},
        enemy={0: (-0.1, 0.0)},
        my_team_is_right=False,
    )
    targets = split_wall.opponent_free_kick_targets(game, (1, 2, 3))
    wall = sorted((targets[1][0], targets[2][0]), key=lambda p: p.y)
    assert wall[0].x == pytest.approx(-0.9) and wall[0].y == pytest.approx(-0.1)
    assert wall[1].x == pytest.approx(-0.9) and wall[1].y == pytest.approx(0.1)


# ---- dispatch: opponent's kick gets the wall, ours does not ---------------


class _FakeMotion:
    def calculate(self, **kwargs):
        return Vector2D(0.0, 0.0), 0.0


def _override_for(command_is_yellow, outfield_ids=(1, 2, 3, 4, 5)):
    return split_wall._restart_override(command_is_yellow, outfield_ids, split_wall.DirectFreeOursStep())


def test_our_own_direct_free_kick_runs_the_built_in_step_unchanged(monkeypatch):
    # We are yellow and the referee gives us DIRECT_FREE_YELLOW: not the wall.
    calls = []
    monkeypatch.setattr(split_wall, "opponent_free_kick_targets", lambda g, ids: calls.append(1) or {})
    game = _game(
        (0.0, 0.0),
        friendly={1: (0.5, 0.1), 2: (-2.0, 1.0), 3: (-2.5, -1.0)},
        enemy={0: (3.0, 0.0)},
        my_team_is_yellow=True,
    )
    ours = _override_for(True)(game, _FakeMotion())
    built_in = RefereeOverride().tick(game, _FakeMotion(), RefereeCommand.DIRECT_FREE_YELLOW)

    assert calls == []
    assert ours == built_in


def test_opponent_direct_free_kick_gets_the_wall(monkeypatch):
    calls = []
    monkeypatch.setattr(split_wall, "opponent_free_kick_targets", lambda g, ids: calls.append(1) or {})
    # We are blue, so the opponent's (yellow) kick is DIRECT_FREE_YELLOW and must be walled.
    game = _game(
        (0.0, 0.0),
        friendly={1: (0.5, 0.1), 2: (-2.0, 1.0)},
        enemy={0: (0.1, 0.0)},
        my_team_is_yellow=False,
    )
    _override_for(True)(game, _FakeMotion())
    assert calls == [1]


def test_blue_key_is_the_opponents_kick_when_we_are_yellow(monkeypatch):
    calls = []
    monkeypatch.setattr(split_wall, "opponent_free_kick_targets", lambda g, ids: calls.append(1) or {})
    game = _game(
        (0.0, 0.0),
        friendly={1: (0.5, 0.1), 2: (-2.0, 1.0)},
        enemy={0: (0.1, 0.0)},
        my_team_is_yellow=True,
    )
    # We are yellow: the BLUE key is the opponent's kick, so it must be walled.
    _override_for(False)(game, _FakeMotion())
    assert calls == [1]
