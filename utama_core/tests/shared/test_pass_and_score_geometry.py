"""Tests for the enemy-defense-area helpers in `pass_and_score_geometry`.

Mirrors of the existing own-defense-area helpers, added for
`LeadAndSupportTactic`'s enemy-box hold fix — see that tactic's test file
for the end-to-end behavior these enable.
"""

from __future__ import annotations

import math

import pytest

from utama_core.config.field_params import STANDARD_FIELD_DIMS
from utama_core.config.referee_constants import OPPONENT_DEFENSE_AREA_KEEP_DISTANCE
from utama_core.entities.data.vector import Vector2D, Vector3D
from utama_core.entities.game import Ball, Field, Game, GameFrame, GameHistory, Robot
from utama_core.motion_planning.src.fastpathplanning.config import (
    fastpathplanningconfig,
)
from utama_core.shared.pass_and_score_geometry import (
    _ACQUIRE_LATERAL_MAX,
    _GOAL_POST_SAFETY_MARGIN,
    _NO_SHOT_STRAFE_STEP,
    _RELEASE_FORWARD_MAX,
    _RELEASE_LATERAL_MAX,
    ORIENTATION_TOLERANCE_RAD,
    ball_in_enemy_defense_area,
    ball_is_loose,
    clamp_outside_enemy_defense_area,
    enemy_defense_area_hold_point,
    enemy_goal_line,
    has_ball,
    no_shot_reposition_target,
    reset_possession_state,
    score_pass_setup,
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


def test_enemy_goal_line_insets_both_posts_by_the_safety_margin():
    game = _game((0.0, 0.0))
    goal_x, goal_y1, goal_y2 = enemy_goal_line(game)
    field = STANDARD_FIELD_DIMS
    raw_y1, raw_y2 = -field.half_goal_width, field.half_goal_width
    assert goal_y1 == pytest.approx(raw_y1 + _GOAL_POST_SAFETY_MARGIN)
    assert goal_y2 == pytest.approx(raw_y2 - _GOAL_POST_SAFETY_MARGIN)
    assert abs(goal_x) == pytest.approx(field.full_field_half_length)


def test_a_tolerance_edge_kick_at_the_inset_post_stays_inside_the_true_goal():
    """Regression for the clear_press_plus_vs_high_press tournament replay
    (t=47.8s): a shooter aimed at the raw post edge and kicked while still
    within `ORIENTATION_TOLERANCE_RAD` of that aim (`oriented_towards`
    passed) — but the resulting kick flew past the true post and out of
    bounds instead of through the goal, since the orientation tolerance's
    lateral slop at that shot distance exceeded the margin between the aim
    point and the real post. `enemy_goal_line`'s inset exists so any target
    chosen within its [goal_y1, goal_y2] keeps that worst-case slop inside
    the real posts — this reproduces the exact traced geometry and checks
    the fix holds."""
    field = STANDARD_FIELD_DIMS
    raw_y1, raw_y2 = -field.half_goal_width, field.half_goal_width
    goal_x = field.full_field_half_length
    _, goal_y1, goal_y2 = enemy_goal_line(_game((0.0, 0.0)))

    shooter_pos = Vector2D(3.141, 1.941)  # the traced shooter's position
    aimed_orientation = shooter_pos.angle_to(Vector2D(goal_x, goal_y2))  # aim at the inset (safe) post edge
    # A kick fired at the very edge of the tolerance band around that aim —
    # the worst case `oriented_towards` still lets through.
    kicked_orientation = aimed_orientation + ORIENTATION_TOLERANCE_RAD

    # Where the ball actually crosses the goal line at that kicked angle.
    exit_y = shooter_pos.y + math.tan(kicked_orientation) * (goal_x - shooter_pos.x)
    assert raw_y1 - 1e-6 <= exit_y <= raw_y2 + 1e-6


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


def test_clamp_outside_enemy_defense_area_clears_planners_obstacle_ring():
    """The clamped point must sit past `FastPathPlanner`'s own no-go band
    around the box, not just past the box's raw edge.

    `FastPathPlanner` draws its enemy-defense-area obstacle
    `OPPONENT_DEFENSE_AREA_KEEP_DISTANCE` (0.25m) outside `front_x`, and then
    refuses to route a target's carrot within another `OBSTACLE_CLEARANCE`
    (0.27m at full clearance) of that obstacle line. A clamp that only
    clears `front_x` by a small margin lands inside this combined band: the
    carrot converges on the clearance ring and the actual target -- though
    perfectly legal -- is never reached. Traced live, 2026-09-05
    (tiki_taka_vs_zone_fluid_RK, COMMITTED_FROZEN 10.3s+ during FORCE_START):
    `GiveAndGoTactic`'s in-flight-pass receiver target got clamped to just
    0.02m past the obstacle line and the receiver stalled indefinitely,
    never completing the pass. This starts a raw point deep inside the box
    (mirroring `intercept_point`'s output before clamping) so the assertion
    exercises the real margin, not just an already-clear point passed
    through unchanged.
    """
    game = _game((_ENEMY_BOX_FRONT_X - 0.5, 0.0))
    clamped = clamp_outside_enemy_defense_area(game, Vector2D(_ENEMY_BOX_FRONT_X - 0.5, 0.0))
    obstacle_line_x = _ENEMY_BOX_FRONT_X + OPPONENT_DEFENSE_AREA_KEEP_DISTANCE
    clearance_from_obstacle_line = clamped.x - obstacle_line_x
    assert clearance_from_obstacle_line > fastpathplanningconfig.OBSTACLE_CLEARANCE


# `my_defense_area` for this file's `my_team_is_right=True` fixture: front-x
# 3.5, goal-x 4.5 (box occupies x in [3.5, 4.5]), half-width 1.0 — see
# `test_ball_is_loose_*` below.
_MY_BOX_FRONT_X = 3.5


def _loose_ball_game(ball_xy: tuple, enemy_xy: tuple) -> Game:
    zv = Vector3D(0, 0, 0)
    frame = GameFrame(
        ts=0.0,
        my_team_is_yellow=True,
        my_team_is_right=True,
        friendly_robots={1: _robot(1, 0.0, 0.0, True)},
        enemy_robots={0: _robot(0, enemy_xy[0], enemy_xy[1], False)},
        ball=Ball(p=Vector3D(ball_xy[0], ball_xy[1], 0), v=zv, a=zv),
    )
    return Game(
        past=GameHistory(10),
        current=frame,
        field=Field(
            my_team_is_right=True, field_dims=STANDARD_FIELD_DIMS, field_bounds=STANDARD_FIELD_DIMS.full_field_bounds
        ),
    )


def test_ball_is_loose_true_when_no_enemy_nearby():
    game = _loose_ball_game(ball_xy=(0.0, 0.0), enemy_xy=(10.0, 10.0))
    assert bool(ball_is_loose(game)) is True


def test_ball_is_loose_false_when_enemy_nearby_in_open_play():
    # Ordinary open-field case: an enemy 1m from a dead ball can freely walk
    # up and take it, so it is genuinely contested.
    game = _loose_ball_game(ball_xy=(0.0, 0.0), enemy_xy=(1.0, 0.0))
    assert bool(ball_is_loose(game)) is False


def test_ball_is_loose_true_when_ball_in_own_box_and_enemy_barred_outside_it():
    # Ball dead inside our own defense area; only our keeper may enter it
    # (`DefenseAreaRule`), so an enemy standing just outside the front edge,
    # even well within `_LOOSE_BALL_CONTEST_RANGE`, can never actually reach
    # it. Must read as loose so a defensive tactic sends a retriever
    # (`ShadowAndMarkTactic` holds it at `own_defense_area_exit_point`
    # instead of ignoring the ball forever).
    game = _loose_ball_game(ball_xy=(_MY_BOX_FRONT_X + 0.3, 0.0), enemy_xy=(_MY_BOX_FRONT_X - 0.3, 0.0))
    assert bool(ball_is_loose(game)) is True


def test_ball_is_loose_false_when_ball_in_own_box_and_enemy_also_inside():
    # If the enemy is (illegally, or mid-transition) actually inside the box
    # with the ball, it can genuinely reach it — still contested.
    game = _loose_ball_game(ball_xy=(_MY_BOX_FRONT_X + 0.3, 0.0), enemy_xy=(_MY_BOX_FRONT_X + 0.2, 0.0))
    assert bool(ball_is_loose(game)) is False


def _loose_ball_game_with_friendly_possession(ball_xy: tuple, enemy_xy: tuple) -> Game:
    """Same as `_loose_ball_game`, but the lone friendly robot's `has_ball`
    is True (`Robot` is a frozen dataclass, so this needs its own builder
    rather than mutating `_loose_ball_game`'s result in place)."""
    zv = Vector3D(0, 0, 0)
    frame = GameFrame(
        ts=0.0,
        my_team_is_yellow=True,
        my_team_is_right=True,
        friendly_robots={1: _robot(1, 0.0, 0.0, True, has_ball=True)},
        enemy_robots={0: _robot(0, enemy_xy[0], enemy_xy[1], False)},
        ball=Ball(p=Vector3D(ball_xy[0], ball_xy[1], 0), v=zv, a=zv),
    )
    return Game(
        past=GameHistory(10),
        current=frame,
        field=Field(
            my_team_is_right=True, field_dims=STANDARD_FIELD_DIMS, field_bounds=STANDARD_FIELD_DIMS.full_field_bounds
        ),
    )


def test_ball_is_loose_false_when_a_friendly_robot_already_has_the_ball():
    """Regression for commit a59a8e5 ("Fix same-team ball scrum"): a
    teammate already dribbling the ball is the opposite of loose, regardless
    of ball speed or enemy distance — `ball_speed` alone can't tell
    "abandoned, rolling to a stop" apart from "being carefully carried while
    lining up a pass" (both are slow). Before the fix, `ball_is_loose` never
    looked at `game.friendly_robots[...].has_ball` at all, so with no enemy
    nearby and a near-zero ball speed (the default in `_loose_ball_game`)
    this read True even though a friendly robot already had it — the exact
    same-team-collision mechanism (`ShadowAndMarkTactic`'s retriever driving
    into its own teammate's carrier) the commit fixes.
    """
    game = _loose_ball_game_with_friendly_possession(ball_xy=(0.0, 0.0), enemy_xy=(10.0, 10.0))
    assert bool(ball_is_loose(game)) is False


def test_ball_is_loose_true_when_ball_just_outside_own_box_and_enemy_at_boundary():
    # Found live 2026-09-02 (`tiki_taka_vs_zone_fluid_Rk.pkl`, ticks 540-599):
    # ball dead ~0.03m outside `zone_fluid`'s own defense area, a `tiki_taka`
    # attacker parked ~0.22m away right at the boundary — `ball_is_loose`
    # used to read this as "contested" forever (enemy within
    # `_LOOSE_BALL_CONTEST_RANGE`), so `ShadowAndMarkTactic` never assigned a
    # retriever, and `goalkeep.py`'s `_ball_needs_retrieval` also never
    # fired (`ball_in_own_defense_area` is strictly-inside-only) — the ball
    # died there for the rest of the match. Reproduced at matching scale
    # here: ball just outside the front edge, enemy at the edge itself.
    ball_x = _MY_BOX_FRONT_X - 0.03
    enemy_x = _MY_BOX_FRONT_X - 0.22
    game = _loose_ball_game(ball_xy=(ball_x, 1.0), enemy_xy=(enemy_x, 1.22))
    assert bool(ball_is_loose(game)) is True


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


def test_no_shot_reposition_target_flips_direction_when_clamped_at_the_sideline():
    """Found live (tournament stuck-match investigation, 2026-09-01): a
    carrier already within `margin` of the field's y-boundary has its
    away-from-blocker strafe step clamped right back to (near-)its own
    current position -- the intended `_NO_SHOT_STRAFE_STEP` (0.6m) strafe
    collapses to a few millimetres, recreating the exact "nothing about
    this position ever changes" freeze this function exists to avoid.
    Traced live: a carrier pinned at y=-2.697 (half_width=3.0, margin=0.3,
    clamp at -2.7) computing a step toward -2.7 moved a net 0.3cm and then
    repeated the identical clamped target forever (568 of 600 seconds).
    Regression: when the preferred direction is clamped away, strafe the
    other way instead.
    """
    carrier_pos = Vector2D(0.7576935794358924, -2.6970753163492853)
    # Nearest enemy below the carrier -> away_sign pushes further negative,
    # straight into the clamp -- the exact live scenario.
    enemies = [Vector2D(0.157, -2.694)]

    target = no_shot_reposition_target(
        carrier_pos, enemies, goal_x=4.5, goal_y1=-0.5, goal_y2=0.5, field_half_width=3.0
    )

    assert abs(target.y - carrier_pos.y) == pytest.approx(_NO_SHOT_STRAFE_STEP, abs=1e-6)
    assert target.y > carrier_pos.y  # flipped to the only direction with room


def test_no_shot_reposition_target_takes_the_preferred_direction_when_not_clamped():
    carrier_pos = Vector2D(0.0, 0.0)
    enemies = [Vector2D(0.0, 1.0)]  # enemy above -> step down (away_sign negative)

    target = no_shot_reposition_target(
        carrier_pos, enemies, goal_x=4.5, goal_y1=-0.5, goal_y2=0.5, field_half_width=3.0
    )

    assert target.y == pytest.approx(carrier_pos.y - _NO_SHOT_STRAFE_STEP)


# ---------------------------------------------------------------------------
# score_pass_setup — regression tests for commit dd14f79 ("Fix dead
# pass-scoring path and permanent ball-lock in GiveAndGoTactic").
# ---------------------------------------------------------------------------


def _pass_setup_game(passer_id: int, passer_xy: tuple, receiver_id: int, receiver_xy: tuple) -> Game:
    """`my_team_is_right=True` -> enemy goal is on the left (`goal_x` < 0),
    matching this file's `_FIELD` fixture above. No enemies at all: `_best_
    receiver`'s real caller passes exactly the passer's own `.p` as
    `passer_position` (see `give_and_go.py`), so `game.friendly_robots`
    always contains an entry at distance 0.0 from `passer_position` --
    that's the exact self-distance bug this commit fixes, reproduced here
    directly rather than through the tactic."""
    friendly = {
        passer_id: _robot(passer_id, passer_xy[0], passer_xy[1], True),
        receiver_id: _robot(receiver_id, receiver_xy[0], receiver_xy[1], True),
    }
    zv = Vector3D(0, 0, 0)
    frame = GameFrame(
        ts=0.0,
        my_team_is_yellow=True,
        my_team_is_right=True,
        friendly_robots=friendly,
        enemy_robots={},
        ball=Ball(p=Vector3D(passer_xy[0], passer_xy[1], 0), v=zv, a=zv),
    )
    return Game(
        past=GameHistory(10),
        current=frame,
        field=Field(
            my_team_is_right=True, field_dims=STANDARD_FIELD_DIMS, field_bounds=STANDARD_FIELD_DIMS.full_field_bounds
        ),
    )


def test_score_pass_setup_not_self_rejected_by_passer_and_receiver_own_entries():
    """The core boundary: `passer_position`/`receiver_position` are a LIVE
    robot's own `.p` (exactly what every real caller passes -- see
    `give_and_go.py`'s `_best_receiver`), so `game.friendly_robots` always
    contains an entry at distance exactly 0.0 from at least the passer
    itself. Before the fix, the clearance loop checked every friendly
    robot's position against passer/receiver with no exclusion, so this
    0.0-distance self-entry always tripped `< min_robot_clearance` and
    `score_pass_setup` returned None on literally every real call --
    making the entire highest-scoring-teammate selection dead code. With
    only the passer and receiver themselves on the roster (no third robot in
    the way) and a clear pass distance/lane/shot, this must now return a
    real score, not None.
    """
    game = _pass_setup_game(passer_id=1, passer_xy=(0.0, 0.0), receiver_id=2, receiver_xy=(-1.0, 0.0))
    passer_pos = game.friendly_robots[1].p
    receiver_pos = game.friendly_robots[2].p

    result = score_pass_setup(game, passer_pos, receiver_pos)

    assert result is not None
    assert result.score > 0


def test_score_pass_setup_still_rejects_a_genuine_third_robot_in_the_clearance_zone():
    """The fix narrows the clearance exclusion to robots within
    `min_pass_distance` of the passer/receiver position (i.e. the
    passer/receiver themselves) -- it must not also blind the check to a
    real third teammate close to the passer but outside the exclusion
    radius. With the default parameters `min_robot_clearance` (~0.44m) is
    always smaller than `min_pass_distance` (0.7m), so this scenario can't
    be constructed there (anything close enough to violate clearance is
    also close enough to be excluded as "the passer itself") -- this test
    passes an explicit smaller `min_pass_distance` to decouple the two radii
    and exercise the two checks independently: a third robot at 0.3m from
    the passer must survive the (now-0.2m) self-exclusion radius but still
    trip the (0.44m) clearance check.
    """
    game_frame_positions = {
        1: _robot(1, 0.0, 0.0, True),  # passer
        2: _robot(2, -1.0, 0.0, True),  # receiver
        3: _robot(3, 0.3, 0.0, True),  # third robot, 0.3m from passer
    }
    zv = Vector3D(0, 0, 0)
    frame = GameFrame(
        ts=0.0,
        my_team_is_yellow=True,
        my_team_is_right=True,
        friendly_robots=game_frame_positions,
        enemy_robots={},
        ball=Ball(p=Vector3D(0.0, 0.0, 0), v=zv, a=zv),
    )
    game = Game(
        past=GameHistory(10),
        current=frame,
        field=Field(
            my_team_is_right=True, field_dims=STANDARD_FIELD_DIMS, field_bounds=STANDARD_FIELD_DIMS.full_field_bounds
        ),
    )

    result = score_pass_setup(game, Vector2D(0.0, 0.0), Vector2D(-1.0, 0.0), min_pass_distance=0.2)
    assert result is None


def test_score_pass_setup_prefers_a_longer_goal_advancing_pass_over_a_short_one():
    """The second half of the fix: a `progress` term (net advance toward the
    enemy goal along the attacking axis) must now weigh into the score at
    the same O(1) scale as `shot_gap`/`pass_clearance`, so a longer pass
    that genuinely advances the ball beats a short one that happens to have
    marginally better incidental clearance -- before this term existed, the
    old `-0.03 * pass_distance` penalty was too weak to matter (an 8m pass
    cost only ~0.24) and a short pass could win on clearance alone.

    Enemy goal is at x=-4.5 (`my_team_is_right=True`); a receiver much
    closer to it must score higher than one only marginally closer, all
    else equal (no enemies, so shot_gap/pass_clearance are identical/maxed
    for both).
    """
    game = _pass_setup_game(passer_id=1, passer_xy=(0.0, 0.0), receiver_id=2, receiver_xy=(-1.0, 0.0))
    passer_pos = Vector2D(0.0, 0.0)

    short_pass = score_pass_setup(game, passer_pos, Vector2D(-1.0, 0.0))
    long_pass = score_pass_setup(game, passer_pos, Vector2D(-3.5, 0.0))

    assert short_pass is not None
    assert long_pass is not None
    assert long_pass.score > short_pass.score


def test_carry_origin_starts_at_pickup_and_clears_when_the_ball_is_dropped():
    from types import SimpleNamespace

    from utama_core.entities.data.vector import Vector3D
    from utama_core.shared.pass_and_score_geometry import (
        CARRY_LIMIT_M,
        carry_exhausted,
        carry_origin,
    )

    robot = SimpleNamespace(has_ball=True)
    zero = Vector3D(0.0, 0.0, 0.0)
    game = SimpleNamespace(friendly_robots={1: robot}, ball=SimpleNamespace(p=Vector3D(1.0, 0.0, 0.0), v=zero))

    origin = carry_origin(game, 1, None)
    assert (origin.x, origin.y) == (1.0, 0.0)
    game.ball.p = Vector3D(1.0, CARRY_LIMIT_M - 0.01, 0.0)
    assert carry_origin(game, 1, origin) is origin  # held: origin stays where the dribble began
    assert not carry_exhausted(game, origin)
    game.ball.p = Vector3D(1.0, CARRY_LIMIT_M, 0.0)
    assert carry_exhausted(game, origin)

    robot.has_ball = False
    assert carry_origin(game, 1, origin) is None
    assert not carry_exhausted(game, None)


def test_carry_exhausted_counts_the_distance_needed_to_stop():
    """At 1.6m/s the carrier needs 1.6^2 / (2*2.5) = 0.51m to stop with the ball
    still on the dribbler, so 0.5m carried is already spent; at rest it is not.
    Traced live: DecoyOverload's lure handed over at ~0.6m and ~1.4m/s and still
    fouled at 1.01m while braking and turning to shoot."""
    from types import SimpleNamespace

    from utama_core.entities.data.vector import Vector2D, Vector3D
    from utama_core.shared.pass_and_score_geometry import carry_exhausted

    origin = Vector2D(0.0, 0.0)
    moving = SimpleNamespace(ball=SimpleNamespace(p=Vector3D(0.5, 0.0, 0.0), v=Vector3D(1.6, 0.0, 0.0)))
    at_rest = SimpleNamespace(ball=SimpleNamespace(p=Vector3D(0.5, 0.0, 0.0), v=Vector3D(0.0, 0.0, 0.0)))

    assert carry_exhausted(moving, origin)
    assert not carry_exhausted(at_rest, origin)
