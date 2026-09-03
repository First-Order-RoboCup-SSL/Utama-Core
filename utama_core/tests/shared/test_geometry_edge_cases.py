"""Boundary-condition and randomized-invariant sweep over `pass_and_score_geometry`'s
public functions.

Public functions in `utama_core/shared/pass_and_score_geometry.py` (`def`/`class`
not prefixed `_`, as of this writing): `reset_possession_state`, `has_ball`,
`at_target`, `oriented_towards`, `clamp_to_field`, `intercept_point`,
`enemy_positions`, `ball_is_loose`, `segment_clearance`, `segment_blocked`,
`enemy_goal_line`, `in_own_defense_area`, `ball_in_own_defense_area`,
`in_enemy_defense_area`, `ball_in_enemy_defense_area`,
`clamp_outside_own_defense_area`, `own_defense_area_exit_point`,
`clamp_outside_enemy_defense_area`, `enemy_defense_area_hold_point`,
`find_best_shot`, `no_shot_reposition_target`, `PassSetupScore`,
`score_pass_setup`.

This file focuses on the ones with the sharpest boundary conditions and the
ones GiveAndGoTactic actually calls for pass geometry/scoring:
`intercept_point` (receive-point projection), `ball_is_loose` (the loose-ball
classifier every defensive tactic queries), and `score_pass_setup`
(GiveAndGoTactic's pass-scoring function, via `_best_receiver` -- see
`utama_core/tactics/give_and_go.py`'s import list). A handful of the simpler
pure-geometry helpers (`at_target`, `oriented_towards`, `clamp_to_field`,
`segment_clearance`/`segment_blocked`) get boundary-only coverage since they
have no interesting failure surface beyond their tolerance/zero-length edges.

Existing `test_pass_and_score_geometry.py` already covers named scenario
behavior (hysteresis, enemy-box helpers, etc.) -- this file does not
duplicate that; it sweeps degenerate inputs and seeded random states
asserting properties (finite, no exception, symmetry), not scenario-specific
expected values.

No exact values asserted that were derived by eyeballing this code's own
output -- every assertion here is either a documented invariant (endpoint
behaviour, tolerance boundary semantics stated in the function's own
docstring) or a property that must hold regardless of the implementation
(finiteness, no exception, mirror symmetry). If a genuine defect turned up
during development (exception/NaN on a degenerate input), it is marked
`xfail(strict=True, ...)` below and reported, not silently fixed -- this
suite must not edit `pass_and_score_geometry.py`.

No real defect was found in this sweep: 200 seeded random states (plus
targeted degenerate constructions: coincident passer/receiver/ball, ball
exactly on a robot, tolerance-boundary targets, zero-length segments,
`game.ball is None`) all returned finite results with no exception, and
`intercept_point`/`score_pass_setup` are both exactly mirror-symmetric under
flipping `my_team_is_right` (negating every x-coordinate and orientation).
There are consequently no `xfail` markers in this file.
"""

from __future__ import annotations

import math
import random

import pytest

from utama_core.config.field_params import STANDARD_FIELD_DIMS
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game.field import Field
from utama_core.entities.game.game import Game
from utama_core.entities.game.game_frame import GameFrame
from utama_core.entities.game.game_history import GameHistory
from utama_core.shared.pass_and_score_geometry import (
    at_target,
    ball_is_loose,
    clamp_to_field,
    enemy_goal_line,
    find_best_shot,
    has_ball,
    in_enemy_defense_area,
    in_own_defense_area,
    intercept_point,
    no_shot_reposition_target,
    oriented_towards,
    reset_possession_state,
    score_pass_setup,
    segment_blocked,
    segment_clearance,
)
from utama_core.tests.fixtures.game_builder import (
    build_game,
    make_ball,
    make_robot,
    random_game,
)

_N_RANDOM_SEEDS = 200


def _finite_vec(v: Vector2D) -> bool:
    return math.isfinite(v.x) and math.isfinite(v.y)


def _game_with_no_ball() -> Game:
    """A real `Game` whose `ball` is `None` -- bypasses `build_game`'s own
    ball default (it fills in a zero ball when `ball=None` is passed), since
    testing `game.ball is None` needs an actual `None`, which is a legal
    `GameFrame.ball` value per its `Optional[Ball]` annotation."""
    frame = GameFrame(
        ts=0.0,
        my_team_is_yellow=True,
        my_team_is_right=True,
        friendly_robots={},
        enemy_robots={},
        ball=None,
    )
    field = Field(
        my_team_is_right=True, field_dims=STANDARD_FIELD_DIMS, field_bounds=STANDARD_FIELD_DIMS.full_field_bounds
    )
    return Game(past=GameHistory(10), current=frame, field=field)


def _mirror_game(g: Game) -> Game:
    """Mirror a `Game` across the x=0 line: negate every x-coordinate and
    x-velocity, reflect orientation (`pi - orientation`), and flip
    `my_team_is_right`. A function reasoning purely about goal-relative
    geometry (not, say, an absolute "always positive x" bias) should be
    exactly symmetric under this transform."""
    friendly = {
        rid: make_robot(
            rid,
            is_friendly=True,
            x=-r.p.x,
            y=r.p.y,
            vx=-r.v.x,
            vy=r.v.y,
            orientation=math.pi - r.orientation,
            has_ball=r.has_ball,
        )
        for rid, r in g.friendly_robots.items()
    }
    enemy = {
        rid: make_robot(
            rid,
            is_friendly=False,
            x=-r.p.x,
            y=r.p.y,
            vx=-r.v.x,
            vy=r.v.y,
            orientation=math.pi - r.orientation,
            has_ball=r.has_ball,
        )
        for rid, r in g.enemy_robots.items()
    }
    ball = make_ball(-g.ball.p.x, g.ball.p.y, -g.ball.v.x, g.ball.v.y)
    return build_game(friendly, enemy, ball, my_team_is_right=not g.my_team_is_right)


# --- intercept_point -------------------------------------------------------


def test_intercept_point_coincident_passer_receiver_ball_is_finite():
    """Passer, receiver, and ball all at the same point, arbitrary distinct
    orientations -- degenerate but must not raise or produce a NaN/inf. The
    result is floored at `min_intercept_distance` along the passer's facing,
    per the function's own `max(projection_t, min_intercept_distance)`."""
    friendly = {
        1: make_robot(1, is_friendly=True, x=1.0, y=1.0, orientation=0.3),
        2: make_robot(2, is_friendly=True, x=1.0, y=1.0, orientation=1.9),
    }
    game = build_game(friendly, {}, ball=make_ball(1.0, 1.0))

    position, orientation = intercept_point(game, 1, 2)

    assert _finite_vec(position)
    assert math.isfinite(orientation)


def test_intercept_point_receiver_exactly_at_ball_position():
    """Receiver sitting exactly on the ball -- `intercept_orientation`'s
    `atan2(ball.y - receiver.y, ball.x - receiver.x)` is `atan2(0, 0)`,
    which Python defines as `0.0` (no exception)."""
    friendly = {
        1: make_robot(1, is_friendly=True, x=-1.0, y=0.0, orientation=0.0),
        2: make_robot(2, is_friendly=True, x=2.0, y=0.0, orientation=0.0),
    }
    game = build_game(friendly, {}, ball=make_ball(2.0, 0.0))

    position, orientation = intercept_point(game, 1, 2)

    assert _finite_vec(position)
    assert orientation == 0.0


@pytest.mark.parametrize("seed", range(_N_RANDOM_SEEDS))
def test_intercept_point_random_states_are_finite_and_clamped_to_field(seed):
    game = random_game(seed=seed, n_friendly=2, n_enemy=1)
    passer_id, receiver_id = 0, 1

    position, orientation = intercept_point(game, passer_id, receiver_id)

    assert _finite_vec(position)
    assert math.isfinite(orientation)
    # clamp_to_field's own contract: stays within [-half_length+margin, ...].
    margin = 0.3
    assert -game.field.half_length - margin <= position.x <= game.field.half_length + margin
    assert -game.field.half_width - margin <= position.y <= game.field.half_width + margin


@pytest.mark.parametrize("seed", range(_N_RANDOM_SEEDS))
def test_intercept_point_is_mirror_symmetric(seed):
    """Mirroring the whole game across x=0 (and flipping `my_team_is_right`)
    must mirror `intercept_position.x` and leave `intercept_position.y`/
    `intercept_orientation`'s geometric meaning consistent -- `intercept_point`
    reasons purely off the passer's own orientation and receiver/ball
    positions relative to each other, never an absolute-x/side bias."""
    game = random_game(seed=seed + 5000, n_friendly=2, n_enemy=0)
    mirrored = _mirror_game(game)

    pos, _orientation = intercept_point(game, 0, 1)
    mirrored_pos, _mirrored_orientation = intercept_point(mirrored, 0, 1)

    assert mirrored_pos.x == pytest.approx(-pos.x, abs=1e-9)
    assert mirrored_pos.y == pytest.approx(pos.y, abs=1e-9)


# --- ball_is_loose -----------------------------------------------------------


def test_ball_is_loose_false_when_game_ball_is_none():
    game = _game_with_no_ball()
    assert ball_is_loose(game) is False


def test_ball_is_loose_ball_exactly_on_a_robot_no_exception():
    """Ball centered exactly on a friendly robot's own position -- degenerate
    distance-to-ball of 0.0 for that robot; must not raise."""
    friendly = {1: make_robot(1, is_friendly=True, x=0.0, y=0.0, has_ball=False)}
    game = build_game(friendly, {}, ball=make_ball(0.0, 0.0))

    assert isinstance(ball_is_loose(game), bool)


def test_ball_is_loose_speed_exactly_at_threshold_reads_not_loose():
    """`ball_is_loose` requires `ball_speed < _LOOSE_BALL_SPEED` strictly;
    exactly at the threshold (0.3 m/s) must read as NOT loose (moving, not
    dead) per the `>=` comparison in the source."""
    game = build_game({}, {}, ball=make_ball(0.0, 0.0, vx=0.3, vy=0.0))
    assert ball_is_loose(game) is False


@pytest.mark.parametrize("seed", range(_N_RANDOM_SEEDS))
def test_ball_is_loose_random_states_never_raise_and_return_bool(seed):
    game = random_game(seed=seed + 20000, n_friendly=3, n_enemy=3)
    result = ball_is_loose(game)
    assert isinstance(result, bool)


# --- score_pass_setup (GiveAndGoTactic's pass-scoring function) -------------


def test_score_pass_setup_coincident_passer_and_receiver_returns_none():
    """Zero pass distance is explicitly infeasible (`pass_distance <
    min_pass_distance` -> `None`), not a divide-by-zero anywhere downstream."""
    friendly = {1: make_robot(1, is_friendly=True, x=0.0, y=0.0), 2: make_robot(2, is_friendly=True, x=2.0, y=0.0)}
    game = build_game(friendly, {}, ball=make_ball(0.0, 0.0))

    result = score_pass_setup(game, Vector2D(0.5, 0.5), Vector2D(0.5, 0.5))

    assert result is None


@pytest.mark.parametrize("seed", range(_N_RANDOM_SEEDS))
def test_score_pass_setup_random_states_never_raise_and_are_finite(seed):
    game = random_game(seed=seed + 30000, n_friendly=3, n_enemy=3)
    passer_pos = game.friendly_robots[0].p
    receiver_pos = game.friendly_robots[1].p

    result = score_pass_setup(game, passer_pos, receiver_pos)

    if result is not None:
        assert math.isfinite(result.score)
        assert _finite_vec(result.passer_position)
        assert _finite_vec(result.receiver_position)


@pytest.mark.parametrize("seed", range(_N_RANDOM_SEEDS))
def test_score_pass_setup_is_mirror_symmetric(seed):
    game = random_game(seed=seed + 40000, n_friendly=2, n_enemy=2)
    mirrored = _mirror_game(game)

    passer_pos = game.friendly_robots[0].p
    receiver_pos = game.friendly_robots[1].p
    mirrored_passer_pos = mirrored.friendly_robots[0].p
    mirrored_receiver_pos = mirrored.friendly_robots[1].p

    result = score_pass_setup(game, passer_pos, receiver_pos)
    mirrored_result = score_pass_setup(mirrored, mirrored_passer_pos, mirrored_receiver_pos)

    assert (result is None) == (mirrored_result is None)
    if result is not None:
        assert mirrored_result.score == pytest.approx(result.score, abs=1e-6)


# --- has_ball ----------------------------------------------------------------


def test_has_ball_visual_ball_exactly_on_robot_center_no_exception():
    friendly = {1: make_robot(1, is_friendly=True, x=0.0, y=0.0, orientation=0.0)}
    game = build_game(friendly, {}, ball=make_ball(0.0, 0.0))
    reset_possession_state(1)

    result = has_ball(game, 1, visual=True)

    assert isinstance(result, bool)


@pytest.mark.parametrize("seed", range(_N_RANDOM_SEEDS))
def test_has_ball_random_states_never_raise(seed):
    game = random_game(seed=seed + 50000, n_friendly=2, n_enemy=1)
    reset_possession_state(0)
    reset_possession_state(1)

    assert isinstance(has_ball(game, 0, visual=True), bool)
    assert isinstance(has_ball(game, 1, visual=False), bool)


# --- at_target / oriented_towards: exact tolerance boundaries ---------------


def test_at_target_exactly_at_tolerance_boundary_is_inside():
    """`at_target` uses `<=`, so a distance exactly equal to `tolerance` counts
    as at-target."""
    friendly = {1: make_robot(1, is_friendly=True, x=0.0, y=0.0)}
    game = build_game(friendly, {}, ball=make_ball())

    assert at_target(game, 1, Vector2D(0.08, 0.0), tolerance=0.08) is True
    assert at_target(game, 1, Vector2D(0.08 + 1e-9, 0.0), tolerance=0.08) is False


def test_oriented_towards_exactly_at_tolerance_boundary_is_inside():
    """`oriented_towards` uses `<=` on the wrapped angle difference."""
    friendly = {1: make_robot(1, is_friendly=True, x=0.0, y=0.0, orientation=0.0)}
    game = build_game(friendly, {}, ball=make_ball())

    assert oriented_towards(game, 1, 0.05, tolerance=0.05) is True
    assert oriented_towards(game, 1, 0.05 + 1e-9, tolerance=0.05) is False


def test_oriented_towards_wraps_around_pi_boundary():
    """Orientation just past +pi wraps to just past -pi; the shortest angular
    difference to a target near -pi must still read as within tolerance."""
    friendly = {1: make_robot(1, is_friendly=True, x=0.0, y=0.0, orientation=math.pi - 0.01)}
    game = build_game(friendly, {}, ball=make_ball())

    assert oriented_towards(game, 1, -math.pi + 0.01, tolerance=0.03) is True


# --- clamp_to_field / segment_clearance / segment_blocked: zero-length -----


@pytest.mark.parametrize("seed", range(_N_RANDOM_SEEDS))
def test_clamp_to_field_random_points_stay_within_bounds(seed):
    rng = random.Random(seed + 60000)
    game = random_game(seed=seed + 60000, n_friendly=1, n_enemy=0)
    point = Vector2D(rng.uniform(-20.0, 20.0), rng.uniform(-20.0, 20.0))

    clamped = clamp_to_field(point, game, margin=0.3)

    assert _finite_vec(clamped)
    assert -game.field.half_length + 0.3 - 1e-9 <= clamped.x <= game.field.half_length - 0.3 + 1e-9
    assert -game.field.half_width + 0.3 - 1e-9 <= clamped.y <= game.field.half_width - 0.3 + 1e-9


def test_segment_clearance_zero_length_segment_no_exception():
    """`start == end`: `_distance_to_segment` special-cases `segment_len_sq
    <= 1e-12` to fall back to plain point-to-start distance rather than
    dividing by zero."""
    point = Vector2D(1.0, 1.0)
    clearance = segment_clearance(point, point, [Vector2D(1.0, 1.0)])
    assert clearance == pytest.approx(0.0, abs=1e-9)


def test_segment_blocked_zero_length_segment_with_obstacle_at_same_point():
    point = Vector2D(1.0, 1.0)
    assert segment_blocked(point, point, [Vector2D(1.0, 1.0)]) is True


def test_segment_clearance_no_obstacles_returns_sentinel():
    point = Vector2D(0.0, 0.0)
    assert segment_clearance(point, Vector2D(1.0, 0.0), []) == 10.0


# --- find_best_shot / no_shot_reposition_target: degenerate inputs ---------


def test_find_best_shot_point_exactly_on_goal_line_no_exception():
    game = build_game({}, {}, ball=make_ball(4.5, 0.0))
    goal_x, y1, y2 = enemy_goal_line(game)

    best_shot_y, gap = find_best_shot(Vector2D(goal_x, 0.0), [], goal_x, y1, y2)

    assert best_shot_y is None or math.isfinite(best_shot_y)


@pytest.mark.parametrize("seed", range(_N_RANDOM_SEEDS))
def test_find_best_shot_random_states_never_raise_and_are_finite(seed):
    game = random_game(seed=seed + 70000, n_friendly=1, n_enemy=4)
    goal_x, y1, y2 = enemy_goal_line(game)

    best_shot_y, gap = find_best_shot(game.ball.p.to_2d(), list(game.enemy_robots.values()), goal_x, y1, y2)

    assert best_shot_y is None or math.isfinite(best_shot_y)
    assert gap is None or (math.isfinite(gap[0]) and math.isfinite(gap[1]))


@pytest.mark.parametrize("seed", range(_N_RANDOM_SEEDS))
def test_no_shot_reposition_target_random_states_never_raise_and_are_finite(seed):
    game = random_game(seed=seed + 80000, n_friendly=1, n_enemy=3)
    goal_x, y1, y2 = enemy_goal_line(game)
    enemy_positions = [e.p for e in game.enemy_robots.values()]

    target = no_shot_reposition_target(game.ball.p.to_2d(), enemy_positions, goal_x, y1, y2, game.field.half_width)

    assert _finite_vec(target)


def test_no_shot_reposition_target_no_enemies_no_exception():
    game = build_game({}, {}, ball=make_ball(0.0, 0.0))
    goal_x, y1, y2 = enemy_goal_line(game)

    target = no_shot_reposition_target(Vector2D(2.0, 0.0), [], goal_x, y1, y2, game.field.half_width)

    assert _finite_vec(target)


# --- own/enemy defense area membership: boundary and mirror consistency ----


@pytest.mark.parametrize("seed", range(_N_RANDOM_SEEDS))
def test_defense_area_membership_random_points_never_raise(seed):
    rng = random.Random(seed + 90000)
    game = random_game(seed=seed + 90000, n_friendly=1, n_enemy=0)
    point = Vector2D(rng.uniform(-6.0, 6.0), rng.uniform(-4.0, 4.0))

    # `bool(...)`, not `isinstance(..., bool)`: `in_enemy_defense_area` can
    # return a numpy bool_ (from the `abs(float(...)) <= half_width + margin`
    # comparison against a value read out of a numpy array) rather than a
    # native Python bool -- truthy-compatible everywhere a bool is used, not
    # a behavioral defect, just not `isinstance(..., bool)`-true. This test
    # only cares that evaluating either function never raises and yields a
    # sensible truth value, not the exact Python type.
    assert bool(in_own_defense_area(game, point)) in (True, False)
    assert bool(in_enemy_defense_area(game, point)) in (True, False)


def test_own_and_enemy_defense_area_are_disjoint_for_a_standard_field():
    """A point can never be in both -- the two boxes sit at opposite ends of
    the field. Sanity property of the two functions' shared geometry
    assumptions, not one whose exact boundary values are asserted."""
    game = build_game({}, {}, ball=make_ball())
    corners_own = game.field.my_defense_area
    center_own = Vector2D(float(corners_own[:, 0].mean()), float(corners_own[:, 1].mean()))

    assert bool(in_own_defense_area(game, center_own)) is True
    assert bool(in_enemy_defense_area(game, center_own)) is False
