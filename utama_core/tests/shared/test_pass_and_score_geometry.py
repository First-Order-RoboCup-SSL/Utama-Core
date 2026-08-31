"""Tests for the enemy-defense-area helpers in `pass_and_score_geometry`.

Mirrors of the existing own-defense-area helpers, added for
`LeadAndSupportTactic`'s enemy-box hold fix — see that tactic's test file
for the end-to-end behavior these enable.
"""

from __future__ import annotations

import math

import pytest

from utama_core.config.field_params import STANDARD_FIELD_DIMS
from utama_core.entities.data.vector import Vector2D, Vector3D
from utama_core.entities.game import Ball, Field, Game, GameFrame, GameHistory, Robot
from utama_core.shared.pass_and_score_geometry import (
    _ACQUIRE_LATERAL_MAX,
    _RELEASE_FORWARD_MAX,
    _RELEASE_LATERAL_MAX,
    ball_in_enemy_defense_area,
    enemy_defense_area_hold_point,
    has_ball,
    reset_possession_state,
)

_FIELD = Field(
    my_team_is_right=True, field_dims=STANDARD_FIELD_DIMS, field_bounds=STANDARD_FIELD_DIMS.full_field_bounds
)
# `enemy_defense_area`'s corners: [goal-line-x, ...], [front-x, ...], ... —
# the "front" edge (facing mid-field) is corner index 1's x, same as
# `in_enemy_defense_area`/`clamp_outside_enemy_defense_area` use internally.
_ENEMY_BOX_FRONT_X = float(_FIELD.enemy_defense_area[1][0])


def _robot(rid: int, x: float, y: float, is_friendly: bool, has_ball: bool = False, orientation: float = 0.0) -> Robot:
    return Robot(
        id=rid,
        is_friendly=is_friendly,
        has_ball=has_ball,
        p=Vector2D(x, y),
        v=Vector2D(0, 0),
        a=Vector2D(0, 0),
        orientation=orientation,
    )


def _game(ball_xy: tuple) -> Game:
    zv = Vector3D(0, 0, 0)
    frame = GameFrame(
        ts=0.0,
        my_team_is_yellow=True,
        my_team_is_right=True,
        friendly_robots={1: _robot(1, 0.0, 0.0, True)},
        enemy_robots={0: _robot(0, ball_xy[0], ball_xy[1], False)},
        ball=Ball(p=Vector3D(ball_xy[0], ball_xy[1], 0), v=zv, a=zv),
    )
    return Game(
        past=GameHistory(10),
        current=frame,
        field=Field(
            my_team_is_right=True, field_dims=STANDARD_FIELD_DIMS, field_bounds=STANDARD_FIELD_DIMS.full_field_bounds
        ),
    )


def test_ball_in_enemy_defense_area_true_when_inside_box():
    game = _game((_ENEMY_BOX_FRONT_X - 0.2, 0.0))
    assert bool(ball_in_enemy_defense_area(game)) is True


def test_ball_in_enemy_defense_area_false_when_outside_box():
    game = _game((0.0, 0.0))  # mid-field
    assert bool(ball_in_enemy_defense_area(game)) is False


def test_ball_in_enemy_defense_area_false_just_outside_front_edge():
    game = _game((_ENEMY_BOX_FRONT_X + 0.05, 0.0))
    assert bool(ball_in_enemy_defense_area(game)) is False


def test_enemy_defense_area_hold_point_stays_outside_box():
    game = _game((_ENEMY_BOX_FRONT_X - 0.2, 0.3))
    hold = enemy_defense_area_hold_point(game, at_y=0.3)
    # my_team_is_right=True -> enemy box is on the left -> outside means x > front edge.
    assert hold.x > _ENEMY_BOX_FRONT_X
    assert hold.y == 0.3


def test_enemy_defense_area_hold_point_clamps_y_inside_box_width():
    game = _game((_ENEMY_BOX_FRONT_X - 0.2, 0.0))
    half_width = STANDARD_FIELD_DIMS.half_defense_area_width
    hold = enemy_defense_area_hold_point(game, at_y=half_width + 5.0)
    assert hold.y < half_width + 5.0


def _visual_game(robot_xy: tuple, robot_orientation: float, ball_xy: tuple, robot_id: int = 1) -> Game:
    """A game with a single friendly robot at `robot_xy`/`robot_orientation`
    and the ball at `ball_xy` — for exercising `has_ball(visual=True)`'s
    dribbler-relative geometry directly, independent of the enemy-box tests
    above (which pin the friendly robot at the origin)."""
    zv = Vector3D(0, 0, 0)
    frame = GameFrame(
        ts=0.0,
        my_team_is_yellow=True,
        my_team_is_right=True,
        friendly_robots={robot_id: _robot(robot_id, robot_xy[0], robot_xy[1], True, orientation=robot_orientation)},
        enemy_robots={},
        ball=Ball(p=Vector3D(ball_xy[0], ball_xy[1], 0), v=zv, a=zv),
    )
    return Game(
        past=GameHistory(10),
        current=frame,
        field=Field(
            my_team_is_right=True, field_dims=STANDARD_FIELD_DIMS, field_bounds=STANDARD_FIELD_DIMS.full_field_bounds
        ),
    )


@pytest.fixture(autouse=True)
def _clear_possession_state():
    """`has_ball(visual=True)`'s commit/release hysteresis (see
    `pass_and_score_geometry._POSSESSION_STATE`'s docstring) is per-robot-id
    module state, so a stale commitment from one test could otherwise leak
    into the next test reusing the same robot id (every test in this file
    uses robot id 1, matching `test_shielding.py`'s equivalent fixture for
    `shielding.py`'s analogous state)."""
    reset_possession_state(1)
    yield
    reset_possession_state(1)


def test_has_ball_visual_true_directly_in_front_within_dribbler_box():
    """Ball sitting squarely in front of the dribbler, facing it head-on —
    the ordinary "just picked it up" case."""
    game = _visual_game((0.0, 0.0), 0.0, (0.10, 0.0))
    assert has_ball(game, 1, visual=True) is True


def test_has_ball_visual_false_ball_directly_behind_robot():
    """A ball chassis-adjacent but *behind* the robot must not read as
    possessed — this was exactly the plain-circle false positive: the old
    check was `distance_to(ball) < capture_distance`, which does not care
    which direction the ball is in."""
    game = _visual_game((0.0, 0.0), 0.0, (-0.10, 0.0))
    assert has_ball(game, 1, visual=True) is False


def test_has_ball_visual_false_ball_beside_robot_lateral_offset():
    """Ball level with the chassis center but well off to the side (a 'ball
    rolled past, robot happened to be standing next to it' case) — inside
    the old 0.15m circle (distance ~0.12m) but well outside the dribbler's
    narrow lateral half-width, so must read False under the new box.

    This is the concrete regression case for the false-positive fix: this
    exact scenario (a ball beside, not in front of, the chassis) was in the
    measured false-positive population from the replay-based Step 1
    measurement (`switch_of_play_vs_default.pkl`/`tiki_taka_vs_counter_press.pkl`),
    and the old plain-circle implementation reads it as True.
    """
    robot_xy = (0.0, 0.0)
    ball_xy = (0.02, 0.12)  # distance ~0.1216m: inside old 0.15m circle
    old_circle_distance = math.hypot(ball_xy[0] - robot_xy[0], ball_xy[1] - robot_xy[1])
    assert old_circle_distance < 0.15, "sanity: must be inside the OLD circular capture_distance"

    game = _visual_game(robot_xy, 0.0, ball_xy)
    assert has_ball(game, 1, visual=True) is False


def test_has_ball_visual_false_ball_in_front_but_facing_away():
    """Ball at the robot's true former-dribbler offset in world space, but
    the robot itself is now facing the opposite way (e.g. it spun around) —
    ball is chassis-adjacent, but nowhere near the dribbler it's now facing
    away from. Another concrete instance of the facing-away false-positive
    class found in the replay measurement."""
    # Ball sits 0.10m in the +x direction from the robot (would be "in
    # front" if the robot faced +x), but the robot faces -x (pi radians).
    game = _visual_game((0.0, 0.0), math.pi, (0.10, 0.0))
    assert has_ball(game, 1, visual=True) is False


def test_has_ball_visual_true_small_lateral_offset_within_dribbler_width():
    """A ball nudged slightly off-center but still within the dribbler's own
    lateral half-width must still read True — the fix narrows the box, it
    must not become so strict that ordinary off-center contact (dribbler
    physically ~8cm wide) reads as a miss."""
    game = _visual_game((0.0, 0.0), 0.0, (0.10, 0.03))
    assert has_ball(game, 1, visual=True) is True


def test_has_ball_visual_respects_capture_distance_parameter():
    """`capture_distance` still narrows/widens the acquire box's forward
    reach, matching its old name/positional meaning for existing callers —
    a ball just past a tightened `capture_distance` reads False."""
    game = _visual_game((0.0, 0.0), 0.0, (0.08, 0.0))
    assert has_ball(game, 1, visual=True, capture_distance=0.05) is False
    reset_possession_state(1)
    assert has_ball(game, 1, visual=True, capture_distance=0.14) is True


def test_has_ball_only_ever_reads_the_friendly_roster():
    """`has_ball` must have no team switch — enemy robots have no real IR
    sensor for tactic code to read, even in sim where rsim's physics engine
    happens to expose ground-truth contact for both teams. A friendly and an
    enemy robot sharing an id, with different `has_ball` values, would make a
    roster mix-up (or a reintroduced team switch defaulting the wrong way)
    read the wrong answer instead of raising."""
    zv = Vector3D(0, 0, 0)
    frame = GameFrame(
        ts=0.0,
        my_team_is_yellow=True,
        my_team_is_right=True,
        friendly_robots={3: _robot(3, 0.0, 0.0, True, has_ball=True)},
        enemy_robots={3: _robot(3, 1.0, 1.0, False, has_ball=False)},
        ball=Ball(p=Vector3D(0.0, 0.0, 0), v=zv, a=zv),
    )
    game = Game(
        past=GameHistory(10),
        current=frame,
        field=Field(
            my_team_is_right=True, field_dims=STANDARD_FIELD_DIMS, field_bounds=STANDARD_FIELD_DIMS.full_field_bounds
        ),
    )
    assert has_ball(game, 3) is True


# ---------------------------------------------------------------------------
# Hysteresis (commit/release) — mirrors test_shielding.py's pattern for
# shielding.py's analogous _COMMITTED_ROBOTS state.
# ---------------------------------------------------------------------------


def test_has_ball_visual_does_not_rechatter_near_acquire_boundary():
    """Regression test for the flicker half of the original complaint: once
    a robot's visual read has gone True, wobbling the ball back out past the
    tighter acquire box (but still within the wider release box) must not
    flip the read back to False — that's what hysteresis is for. Mirrors
    `test_shielding.py`'s `test_shielded_approach_angle_does_not_rechatter_near_commit_boundary`."""
    robot_xy = (0.0, 0.0)

    def _visual_at(forward: float) -> bool:
        game = _visual_game(robot_xy, 0.0, (forward, 0.0))
        return has_ball(game, 1, visual=True)

    # Acquire: ball well within the tight acquire box.
    assert _visual_at(0.10) is True
    # Ball drifts out past the acquire box's forward max, but still within
    # the wider release box -- must NOT drop back to False while committed.
    acquire_and_release_gap = (_RELEASE_FORWARD_MAX - 0.14) / 2
    assert _visual_at(0.14 + acquire_and_release_gap) is True
    # Only once genuinely past the release box does it drop to False.
    assert _visual_at(_RELEASE_FORWARD_MAX + 0.01) is False


def test_has_ball_visual_release_lateral_hysteresis():
    """Same hysteresis, exercised on the lateral axis instead of forward."""
    robot_xy = (0.0, 0.0)

    def _visual_at(lateral: float) -> bool:
        game = _visual_game(robot_xy, 0.0, (0.10, lateral))
        return has_ball(game, 1, visual=True)

    assert _visual_at(0.0) is True
    # Past the tight acquire lateral half-width but within the release one.
    mid = (_ACQUIRE_LATERAL_MAX + _RELEASE_LATERAL_MAX) / 2
    assert _visual_at(mid) is True
    assert _visual_at(_RELEASE_LATERAL_MAX + 0.01) is False


def test_has_ball_visual_acquire_state_is_per_robot():
    """Two robots' hysteresis states must not interfere with each other."""
    zv = Vector3D(0, 0, 0)

    def _game_two(r0_ball_xy, r1_ball_xy_unused=None) -> Game:
        frame = GameFrame(
            ts=0.0,
            my_team_is_yellow=True,
            my_team_is_right=True,
            friendly_robots={
                0: _robot(0, 0.0, 0.0, True, orientation=0.0),
                1: _robot(1, 5.0, 5.0, True, orientation=0.0),
            },
            enemy_robots={},
            ball=Ball(p=Vector3D(r0_ball_xy[0], r0_ball_xy[1], 0), v=zv, a=zv),
        )
        return Game(
            past=GameHistory(10),
            current=frame,
            field=Field(
                my_team_is_right=True,
                field_dims=STANDARD_FIELD_DIMS,
                field_bounds=STANDARD_FIELD_DIMS.full_field_bounds,
            ),
        )

    reset_possession_state(0)
    try:
        # Robot 0 acquires the ball (right next to it).
        game = _game_two((0.10, 0.0))
        assert has_ball(game, 0, visual=True) is True
        # Robot 1 is 5m away from this same ball -- must independently read
        # False, unaffected by robot 0's committed state.
        assert has_ball(game, 1, visual=True) is False
    finally:
        reset_possession_state(0)
