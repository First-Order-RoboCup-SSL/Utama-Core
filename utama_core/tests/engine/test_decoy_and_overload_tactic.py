"""Unit-level (no rsim) test for `DecoyOverloadTactic`'s "finish"-phase timeout,
plus regression tests for commit a59a8e5 ("Fix same-team ball scrum:
ball_is_loose and DecoyOverloadTactic gaps").

Regression for a real 2026-09-02 full-length-tournament freeze
(`clear_press_plus_vs_shadow_switch_LK.pkl`, t=205.8s): a referee restart can
strand the tactic in `"finish"` with the ball nowhere near either the decoy or
the overloader and no path back to "setup" — `is_committed()` only releases on
`goal_scored`, so with no goal ever scored this held the slot's two robots for
the rest of the match while a different tactic's robot independently
converged on the same displaced ball, and both stalled at
`FastPathPlanner.OBSTACLE_CLEARANCE` apart. Same stalled-phase-with-no-timeout
bug class `pass_and_shoot.py`'s own `_PHASE_TIMEOUT_TICKS` exists to prevent
(see its own `is_committed()`/`phase == "setup"` gating in
`test_pass_and_shoot_tactic.py`).

The timeout check is the first statement in the "finish" branch of `tick()`,
before any `Game`-shaped attribute is touched (`has_ball`, `_decoy_shot_open`,
`_pass_exec` all require a real `Game`). Passing `game=None` here is
deliberate: if the timeout guard didn't short-circuit before those calls, this
test would raise `AttributeError` instead of returning cleanly, proving the
guard actually protects the rest of the branch rather than just existing
alongside it.
"""

from __future__ import annotations

from utama_core.config.field_params import STANDARD_FIELD_DIMS
from utama_core.engine.context import TickContext
from utama_core.entities.data.vector import Vector2D, Vector3D
from utama_core.entities.game import Field, Game, GameHistory
from utama_core.entities.game.ball import Ball
from utama_core.entities.game.game_frame import GameFrame
from utama_core.entities.game.robot import Robot
from utama_core.motion_planning.src.common.motion_controller import MotionController
from utama_core.tactics.decoy_and_overload import (
    _FINISH_TIMEOUT_TICKS,
    _LOOSE_BALL_SPEED,
    _LURE_MAX_TICKS,
    DecoyOverloadMem,
    DecoyOverloadTactic,
    _teammate_already_has_ball,
)


def test_is_committed_true_mid_finish_phase():
    tactic = DecoyOverloadTactic()
    mem = DecoyOverloadMem(decoy_id=1, overloader_id=2, phase="finish", finish_ticks=1)
    assert tactic.is_committed(game=None, mem=mem) is True


def test_finish_phase_times_out_and_releases_the_tactic():
    tactic = DecoyOverloadTactic()
    mem = DecoyOverloadMem(
        decoy_id=1,
        overloader_id=2,
        marker_id=None,
        marker_start_y=0.0,
        phase="finish",
        finish_ticks=_FINISH_TIMEOUT_TICKS,  # already at budget: this tick pushes it over
    )
    ctx = TickContext(motion_controller=None)
    commands, new_mem = tactic.tick(game=None, ctx=ctx, robot_ids=(1, 2), mem=mem)

    assert commands == {}
    assert new_mem.decoy_id is None
    assert tactic.is_committed(game=None, mem=new_mem) is False


def test_finish_phase_does_not_time_out_before_budget():
    """The timeout must not fire on the very first "finish" tick — only once
    `finish_ticks` genuinely exceeds the budget — otherwise every decoy-and-
    overload attempt would abort immediately instead of getting its real
    pass+shoot window. `tick()` itself needs a real `Game` past this point
    (`has_ball`/`_decoy_shot_open`/`_pass_exec`), so this checks the guard
    condition directly rather than calling `tick()` with `game=None` (which
    would legitimately raise once past the guard, on a fresh attempt)."""
    mem = DecoyOverloadMem(decoy_id=1, overloader_id=2, phase="finish", finish_ticks=0)
    mem.finish_ticks += 1
    assert mem.finish_ticks <= _FINISH_TIMEOUT_TICKS


# ---------------------------------------------------------------------------
# Regression tests for commit a59a8e5 ("Fix same-team ball scrum:
# ball_is_loose and DecoyOverloadTactic gaps").
# ---------------------------------------------------------------------------


def _robot(rid: int, x: float, y: float, is_friendly: bool, has_ball: bool = False) -> Robot:
    return Robot(
        id=rid,
        is_friendly=is_friendly,
        has_ball=has_ball,
        p=Vector2D(x, y),
        v=Vector2D(0, 0),
        a=Vector2D(0, 0),
        orientation=0.0,
    )


def _make_scrum_game(decoy_ball: bool, teammate_ball: bool, ball_v: Vector2D = Vector2D(0.0, 0.0)) -> Game:
    """Robot 1 is the tactic's own decoy/robot-of-interest; robot 3 is a
    teammate in a *different* tactic slot (e.g. a `GiveAndGoTactic` carrier)
    that may or may not already have the ball."""
    friendly = {
        1: _robot(1, 0.0, 0.0, True, has_ball=decoy_ball),
        3: _robot(3, 2.0, 0.0, True, has_ball=teammate_ball),
    }
    zv = Vector3D(0, 0, 0)
    ball = Ball(p=Vector3D(2.0, 0.0, 0.0), v=Vector3D(ball_v.x, ball_v.y, 0.0), a=zv)
    frame = GameFrame(
        ts=0.0, my_team_is_yellow=True, my_team_is_right=True, friendly_robots=friendly, enemy_robots={}, ball=ball
    )
    field = Field(
        my_team_is_right=True, field_dims=STANDARD_FIELD_DIMS, field_bounds=STANDARD_FIELD_DIMS.full_field_bounds
    )
    return Game(past=GameHistory(max_history=20), current=frame, field=field)


def test_teammate_already_has_ball_true_when_a_different_friendly_has_it():
    """The core boundary the commit adds: a FRIENDLY robot other than
    `excluding_id` currently holding the ball must read True — before this
    function existed, nothing in `decoy_and_overload.py` ever checked a
    teammate's possession, so a decoy in a different tactic slot would drive
    straight into an already-claimed ball (the live-found scrum)."""
    game = _make_scrum_game(decoy_ball=False, teammate_ball=True)
    assert _teammate_already_has_ball(game, excluding_id=1) is True


def test_teammate_already_has_ball_false_when_excluded_id_itself_has_it():
    """The `excluding_id` robot's own possession must not count as "a
    teammate already has it" — otherwise the decoy could never be judged to
    have fetched the ball itself."""
    game = _make_scrum_game(decoy_ball=True, teammate_ball=False)
    assert _teammate_already_has_ball(game, excluding_id=1) is False


def test_teammate_already_has_ball_true_when_ball_moving_fast_mid_pass():
    """Pins the speed-threshold boundary: even with nobody currently
    possessing the ball (mid-flight between two other teammates), a speed at
    or above `_LOOSE_BALL_SPEED` must still read as "already the team's
    ball, not this tactic's problem" — the second clause the docstring calls
    out as needed because the possession-only check alone let a second scrum
    through exactly here."""
    game = _make_scrum_game(decoy_ball=False, teammate_ball=False, ball_v=Vector2D(_LOOSE_BALL_SPEED, 0.0))
    assert _teammate_already_has_ball(game, excluding_id=1) is True


def test_teammate_already_has_ball_false_when_ball_genuinely_loose():
    """Negative control: nobody has it, and it is at rest (speed 0, well
    under `_LOOSE_BALL_SPEED`) — a genuinely abandoned ball must read False
    so the tactic is still free to go fetch it itself."""
    game = _make_scrum_game(decoy_ball=False, teammate_ball=False, ball_v=Vector2D(0.0, 0.0))
    assert _teammate_already_has_ball(game, excluding_id=1) is False


def test_lure_phase_does_not_advance_to_finish_on_timeout_without_the_ball():
    """Deeper fix in the same commit: the lure-to-finish phase transition
    (`dragged or lure_ticks >= _LURE_MAX_TICKS`) must also require the decoy
    to actually have the ball. Before the fix, a lure that timed out without
    ever fetching the ball (because a teammate elsewhere already had/was
    passing it) still transitioned into "finish", whose shared `_pass_exec`
    call immediately chases the ball itself the instant the "passer" doesn't
    have it -- bypassing the `_teammate_already_has_ball` guards entirely
    and recreating the same scrum one phase later.

    Uses a decoy with no `has_ball` (never collected it) and `lure_ticks`
    already at the timeout budget -- pre-fix this alone flips `mem.phase` to
    "finish"; post-fix it must stay "lure".
    """
    game = _make_scrum_game(decoy_ball=False, teammate_ball=False, ball_v=Vector2D(0.0, 0.0))
    ctx = TickContext(motion_controller=_NullMotionController())
    tactic = DecoyOverloadTactic()
    mem = DecoyOverloadMem(
        decoy_id=1,
        overloader_id=3,
        marker_id=None,
        marker_start_y=0.0,
        phase="lure",
        lure_ticks=_LURE_MAX_TICKS,  # already at/over budget -- would time out this tick
    )

    _commands, new_mem = tactic.tick(game, ctx, (1, 3), mem)

    assert new_mem.phase == "lure"


class _NullMotionController(MotionController):
    """Never actually reached meaningfully (the decoy has no ball, so it's
    driven via `go_to_point`'s support-hold branch or `go_to_ball`, neither
    of which this test inspects) — just needs to return a valid velocity."""

    def __init__(self):
        super().__init__(mode="rsim")

    def calculate(self, game, robot_id, target_pos, target_oren):
        return Vector2D(0.0, 0.0), 0.0
