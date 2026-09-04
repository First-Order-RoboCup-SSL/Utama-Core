"""Regression tests for `GiveAndGoTactic`'s first-touch give-up timeouts.

Found live in a tournament replay (`counter_flow_vs_counter_press.pkl`,
2026-09-01): a carrier with no reachable/unblocked teammate held the ball
for 11+ seconds, far past the intended 4s (`_MAX_HOP_TICKS`) abandon budget.
Root cause: the first-touch-of-possession branch in `tick()` unconditionally
reset `mem.hop_ticks = 0` right before incrementing it whenever
`_nearest_safe_receiver` was called, even when it returned `None` — pinning
`hop_ticks` at exactly 1 forever instead of climbing toward `_MAX_HOP_TICKS`.

A second, distinct stall was found live the same day
(`counter_press_vs_counter_flow.pkl`, t=29-41s+): congested play let
`_nearest_safe_receiver` keep finding a *different* fresh candidate every
time the previous one got lane-blocked and abandoned
(`_LANE_BLOCKED_ABANDON_TICKS`, ~0.5s each) — 10 distinct receiver-lock
attempts over 12.5s, no pass ever completed, `hop_count` stuck at 0 the
whole time so `_MAX_HOPS_PER_POSSESSION`'s force_shot never triggered
either. `_MAX_FIRST_TOUCH_TICKS`/`first_touch_stuck` bounds the *cumulative*
time spent in this retry loop, using `mem.ticks_held` (which persists
across individual receiver attempts, unlike `hop_ticks`).
"""

from __future__ import annotations

import math

from utama_core.config.field_params import STANDARD_FIELD_DIMS
from utama_core.engine.context import TickContext
from utama_core.entities.data.vector import Vector2D, Vector3D
from utama_core.entities.game import Field, Game, GameHistory
from utama_core.entities.game.ball import Ball
from utama_core.entities.game.game_frame import GameFrame
from utama_core.entities.game.robot import Robot
from utama_core.motion_planning.src.common.motion_controller import MotionController
from utama_core.tactics.give_and_go import (
    _FIRST_TOUCH_FORCE_SHOT_TICKS,
    _MAX_FIRST_TOUCH_TICKS,
    _MAX_HOP_TICKS,
    GiveAndGoTactic,
    _relocate_target,
)


class _NullMotionController(MotionController):
    """Never actually reached by this test's carrier (no receiver is ever
    locked), but `_relocate_others`' support-repositioning calls into it."""

    def __init__(self):
        super().__init__(mode="rsim")

    def calculate(self, game, robot_id, target_pos, target_oren):
        return Vector2D(0.0, 0.0), 0.0


def _make_game(carrier_pos: Vector2D, teammate_pos: Vector2D, enemy_pos: Vector2D) -> Game:
    friendly = {
        1: Robot(
            id=1, is_friendly=True, has_ball=True, p=carrier_pos, v=Vector2D(0, 0), a=Vector2D(0, 0), orientation=0.0
        ),
        2: Robot(
            id=2, is_friendly=True, has_ball=False, p=teammate_pos, v=Vector2D(0, 0), a=Vector2D(0, 0), orientation=0.0
        ),
    }
    enemy = {
        3: Robot(
            id=3, is_friendly=False, has_ball=False, p=enemy_pos, v=Vector2D(0, 0), a=Vector2D(0, 0), orientation=0.0
        ),
    }
    ball = Ball(Vector3D(carrier_pos.x, carrier_pos.y, 0.0), Vector3D(0.0, 0.0, 0.0), Vector3D(0.0, 0.0, 0.0))
    frame = GameFrame(
        ts=0.0, my_team_is_yellow=True, my_team_is_right=True, friendly_robots=friendly, enemy_robots=enemy, ball=ball
    )
    field = Field(
        my_team_is_right=True,
        field_dims=STANDARD_FIELD_DIMS,
        field_bounds=STANDARD_FIELD_DIMS.full_field_bounds,
    )
    return Game(past=GameHistory(max_history=20), current=frame, field=field)


def test_hop_ticks_climbs_when_no_receiver_is_ever_reachable():
    """With the teammate's only lane blocked by an enemy standing on it,
    `_nearest_safe_receiver` returns None every tick. `hop_ticks` must climb
    toward `_MAX_HOP_TICKS` instead of being reset to 0 every tick it fails.
    """
    carrier_pos = Vector2D(0.0, 0.0)
    teammate_pos = Vector2D(1.0, 0.0)
    enemy_pos = Vector2D(0.5, 0.0)  # sits directly on the carrier->teammate segment

    game = _make_game(carrier_pos, teammate_pos, enemy_pos)
    ctx = TickContext(motion_controller=_NullMotionController(), match_log=None)
    tactic = GiveAndGoTactic()
    mem = tactic.initial_mem()

    for _ in range(10):
        _, mem = tactic.tick(game, ctx, (1, 2), mem)
        assert mem.receiver_id is None  # lane stays blocked the whole test

    # Before the fix, hop_ticks was reset to 0 then incremented to 1 on every
    # tick, so after 10 ticks it would still read 1.
    assert mem.hop_ticks == 10


def test_carrier_eventually_abandons_the_hold_after_max_hop_ticks():
    """Once hop_ticks reaches _MAX_HOP_TICKS, the carrier must fall through to
    the shoot/reposition branch (receiver_id stays None, hop_ticks resets to
    0) rather than holding forever."""
    carrier_pos = Vector2D(0.0, 0.0)
    teammate_pos = Vector2D(1.0, 0.0)
    enemy_pos = Vector2D(0.5, 0.0)

    game = _make_game(carrier_pos, teammate_pos, enemy_pos)
    ctx = TickContext(motion_controller=_NullMotionController(), match_log=None)
    tactic = GiveAndGoTactic()
    mem = tactic.initial_mem()

    for _ in range(_MAX_HOP_TICKS):
        _, mem = tactic.tick(game, ctx, (1, 2), mem)

    assert mem.hop_ticks < _MAX_HOP_TICKS
    assert mem.receiver_id is None


def _make_flicker_game() -> Game:
    """Carrier facing +x, teammate off-axis at (1.5, 2.0), enemy at (1.0, 0.0).

    `_nearest_safe_receiver`'s raw carrier->teammate segment check passes
    (the enemy is nowhere near that line) so the teammate is selected every
    time — but `_pass_exec`'s intercept point follows the carrier's fixed
    orientation (+x, since the fixture never turns it), landing right next
    to the enemy, so the hop's own lane-blocked check trips almost
    immediately and the hop gets abandoned via `_LANE_BLOCKED_ABANDON_TICKS`
    (~0.5s) every single time — the same teammate gets re-picked next tick,
    forever, without ever completing a single pass.
    """
    carrier_pos = Vector2D(0.0, 0.0)
    teammate_pos = Vector2D(1.5, 2.0)
    enemy_pos = Vector2D(1.0, 0.0)
    return _make_game(carrier_pos, teammate_pos, enemy_pos)


def test_ticks_held_keeps_climbing_across_repeated_abandoned_hops():
    """Found live (`counter_press_vs_counter_flow.pkl`, 2026-09-01): 10
    distinct receiver-lock attempts over 12.5s, hop_count stuck at 0 the
    whole time because every hop got lane-blocked-abandoned before
    completing. `ticks_held` (unlike `hop_ticks`, which resets on every
    fresh attempt) must keep climbing across these repeated abandons.
    """
    game = _make_flicker_game()
    ctx = TickContext(motion_controller=_NullMotionController(), match_log=None)
    tactic = GiveAndGoTactic()
    mem = tactic.initial_mem()

    for _ in range(400):
        _, mem = tactic.tick(game, ctx, (1, 2), mem)

    assert mem.hop_count == 0  # no pass ever actually completed
    assert mem.ticks_held == 400  # kept climbing despite many hop_ticks resets


def test_first_touch_stuck_gives_up_after_max_first_touch_ticks():
    """Past `_MAX_FIRST_TOUCH_TICKS`, the carrier must stop re-locking a new
    receiver once any hop already in flight at that point finishes running
    its own course (its own `_LANE_BLOCKED_ABANDON_TICKS` budget), rather
    than retrying forever.
    """
    game = _make_flicker_game()
    ctx = TickContext(motion_controller=_NullMotionController(), match_log=None)
    tactic = GiveAndGoTactic()
    mem = tactic.initial_mem()

    # +40 (not just +5): a hop already in flight when ticks_held crosses
    # _MAX_FIRST_TOUCH_TICKS is allowed to run to its own abandon before
    # `first_touch_stuck` blocks the next selection (see `tick()`'s
    # `mem.receiver_id is not None` in-flight branch).
    for _ in range(_MAX_FIRST_TOUCH_TICKS + 40):
        _, mem = tactic.tick(game, ctx, (1, 2), mem)

    assert mem.receiver_id is None
    assert mem.hop_count == 0


def _make_open_lane_game(carrier_orientation: float) -> Game:
    """No enemies at all (shot lane always open — `find_best_shot` never
    returns None) and a teammate close enough to be a real ally but inside
    `_MIN_SAFE_PASS_DISTANCE`, so `_nearest_safe_receiver` never selects it
    (`dist < _MIN_SAFE_PASS_DISTANCE: continue`) -- `mem.receiver_id` stays
    None forever without needing a blocked lane, isolating the
    `first_touch_stuck`/`force_shot` interaction from any pass mechanics.
    `carrier_orientation` lets a test aim the carrier at the enemy goal
    directly, so once `force_shot` overrides `first_touch_stuck` the tactic
    takes the immediate-`kick()` branch rather than `turn_on_spot`.
    """
    carrier_pos = Vector2D(0.0, 0.0)
    teammate_pos = Vector2D(0.3, 0.3)  # well under _MIN_SAFE_PASS_DISTANCE (0.7m)
    friendly = {
        1: Robot(
            id=1,
            is_friendly=True,
            has_ball=True,
            p=carrier_pos,
            v=Vector2D(0, 0),
            a=Vector2D(0, 0),
            orientation=carrier_orientation,
        ),
        2: Robot(
            id=2, is_friendly=True, has_ball=False, p=teammate_pos, v=Vector2D(0, 0), a=Vector2D(0, 0), orientation=0.0
        ),
    }
    ball = Ball(Vector3D(carrier_pos.x, carrier_pos.y, 0.0), Vector3D(0.0, 0.0, 0.0), Vector3D(0.0, 0.0, 0.0))
    frame = GameFrame(
        ts=0.0, my_team_is_yellow=True, my_team_is_right=True, friendly_robots=friendly, enemy_robots={}, ball=ball
    )
    field = Field(
        my_team_is_right=True,
        field_dims=STANDARD_FIELD_DIMS,
        field_bounds=STANDARD_FIELD_DIMS.full_field_bounds,
    )
    return Game(past=GameHistory(max_history=20), current=frame, field=field)


def test_first_touch_stuck_eventually_force_shoots_instead_of_repositioning_forever():
    """Regression for commit dd14f79's "permanent ball-lock" fix: once
    `mem.ticks_held` crosses `_FIRST_TOUCH_FORCE_SHOT_TICKS`, `force_shot`
    must become True and override `first_touch_stuck` at the shot/reposition
    branch (`if best_shot_y is None or (first_touch_stuck and not
    force_shot):`), so the carrier takes the open shot instead of strafing
    indefinitely. Before the fix, `force_shot` only ever checked
    `mem.hop_count` (which never leaves 0 in this scenario, matching the
    live bug's mechanism), so `first_touch_stuck` alone kept routing to the
    no-open-lane reposition branch forever even with a continuously open
    shot lane -- a carrier held a wide-open lane for 50+ seconds and never
    shot, in the live trace this commit fixes.

    `my_team_is_right=True` -> enemy goal at x=-4.5, y in [-goal_half_width,
    goal_half_width] around 0 -- carrier at the origin facing math.pi (-x,
    straight at the goal) is already oriented toward the shot target, so
    once `force_shot` fires the tactic's `oriented_towards()` check passes
    immediately and it issues `kick()` this same tick, not `turn_on_spot()`.
    """
    game = _make_open_lane_game(carrier_orientation=math.pi)
    ctx = TickContext(motion_controller=_NullMotionController(), match_log=None)
    tactic = GiveAndGoTactic()
    mem = tactic.initial_mem()

    commands = {}
    for _ in range(_FIRST_TOUCH_FORCE_SHOT_TICKS + 5):
        commands, mem = tactic.tick(game, ctx, (1, 2), mem)

    # Sanity: this scenario never completes a hop -- hop_count-gated
    # force_shot never fires, isolating the ticks_held-gated path.
    assert mem.hop_count == 0
    assert mem.ticks_held >= _FIRST_TOUCH_FORCE_SHOT_TICKS
    assert commands[1].kick == 1


# ---------------------------------------------------------------------------
# _relocate_target: support-run direction and depth
# ---------------------------------------------------------------------------
#
# Found live investigating a real "these strategies never shoot" bug
# (high_press/overload_flow/score_aware_zone_flow, 2026-09-05,
# docs/roadmap.md): `_relocate_target` generated candidates at
# `ball_x + dx` with hardcoded positive `dx` -- correct for a team attacking
# +x, but for `my_team_is_right=True` (this codebase's convention: the enemy
# goal then sits at -x, see `Field.enemy_goal_line`) every candidate this
# produced sat *behind* the ball, toward the team's own goal, not ahead of
# it. Separately, `dx` was capped at 1.8m, too shallow to ever reach the
# shot detector's attacking-third gate (the final ~3m of a 9m-long field)
# over repeated hops. Both are fixed together: `attack_sign` is read
# directly off `enemy_goal_line`'s actual x-sign (not off `my_team_is_right`
# by hand), and the candidate `dx` range now reaches deep into the
# attacking third when the field ahead is open.


def _make_relocate_game(my_team_is_right: bool, ball_x: float = 0.0) -> Game:
    friendly = {
        1: Robot(
            id=1,
            is_friendly=True,
            has_ball=True,
            p=Vector2D(ball_x, 0.0),
            v=Vector2D(0, 0),
            a=Vector2D(0, 0),
            orientation=0.0,
        ),
        2: Robot(
            id=2,
            is_friendly=True,
            has_ball=False,
            p=Vector2D(ball_x - 0.5, -1.0),
            v=Vector2D(0, 0),
            a=Vector2D(0, 0),
            orientation=0.0,
        ),
    }
    enemy = {
        3: Robot(
            id=3,
            is_friendly=False,
            has_ball=False,
            p=Vector2D(3.5, 2.5),
            v=Vector2D(0, 0),
            a=Vector2D(0, 0),
            orientation=0.0,
        ),
    }
    ball = Ball(Vector3D(ball_x, 0.0, 0.0), Vector3D(0.0, 0.0, 0.0), Vector3D(0.0, 0.0, 0.0))
    frame = GameFrame(
        ts=0.0,
        my_team_is_yellow=True,
        my_team_is_right=my_team_is_right,
        friendly_robots=friendly,
        enemy_robots=enemy,
        ball=ball,
    )
    field = Field(
        my_team_is_right=my_team_is_right,
        field_dims=STANDARD_FIELD_DIMS,
        field_bounds=STANDARD_FIELD_DIMS.full_field_bounds,
    )
    return Game(past=GameHistory(max_history=20), current=frame, field=field)


def test_relocate_target_advances_toward_enemy_goal_when_attacking_negative_x():
    """my_team_is_right=True -> enemy goal at x=-4.5 (Field.enemy_goal_line).
    The relocation target for the uncommitted teammate must move toward
    -x, not +x -- before the fix, the hardcoded `ball_x + dx` (dx always
    positive) sent every candidate the wrong way for this side."""
    game = _make_relocate_game(my_team_is_right=True, ball_x=0.0)
    target = _relocate_target(game, 2, avoid=[Vector2D(0.0, 0.0)])
    assert target.x < 0.0


def test_relocate_target_advances_toward_enemy_goal_when_attacking_positive_x():
    """Mirror of the above: my_team_is_right=False -> enemy goal at x=+4.5,
    target must move toward +x."""
    game = _make_relocate_game(my_team_is_right=False, ball_x=0.0)
    target = _relocate_target(game, 2, avoid=[Vector2D(0.0, 0.0)])
    assert target.x > 0.0


def test_relocate_target_reaches_the_attacking_third_when_field_is_open():
    """With no congestion forcing a short/retreated candidate, the chosen
    support point should land in the attacking third (the shot detector's
    own threshold: within `_SHOT_ATTACKING_THIRD_M` of the far goal line),
    not merely somewhere closer to it than before. Previously capped at
    ball_x +/- 1.8m, which for a centre-field ball never reaches within
    2.7m of a goal 4.5m out."""
    game = _make_relocate_game(my_team_is_right=True, ball_x=0.0)
    target = _relocate_target(game, 2, avoid=[Vector2D(0.0, 0.0)])
    goal_x = float(game.field.enemy_goal_line[0][0])
    assert abs(target.x - goal_x) <= 1.5
