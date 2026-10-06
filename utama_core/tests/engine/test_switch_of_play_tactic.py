"""Unit-level (no rsim) tests for `SwitchOfPlayTactic`'s settle-debounce logic
and its "relay" source-robot ball-recovery gate.

Covers two fixes, both from the same live-traced match
(`shadow_switch_vs_zone_fluid`, tournament run 2026-09-04):

1. `_debounced_settled`/`SwitchOfPlayMem.settled_ticks` — a robot converged
   in position but still oscillating in speed around
   `_ARRIVAL_SPEED_THRESHOLD` (a real symptom of this codebase's PID
   translation controller having no terminal deadband near a static target —
   see `_SETTLE_DEBOUNCE_TICKS`'s comment in `switch_of_play.py`) must still
   be recognised as settled within a bounded number of ticks, not flap
   forever on a single noisy instantaneous read.
2. `SwitchOfPlayMem.source_had_ball` — the "relay" phase's source robot must
   fall through to `go_to_ball` when it has never actually acquired the ball,
   even while sitting inside `_BALL_RECOVERY_RADIUS` of it, rather than
   treating mere proximity as equivalent to a brief post-acquisition ejection
   and switching to the "hold, face the runner" command forever.

Full end-to-end behaviour (actual robot motion, pass execution) needs
`StrategyRunner` + rsim, out of scope here — see `test_pass_and_shoot_tactic.py`
for the same "test the pure logic, not the simulation" split; test 2's group
below follows `test_decoy_and_overload_tactic.py`'s heavier full-`Game`-fixture
pattern instead, since the bug lives inline in `tick()`, not a standalone
helper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from unittest.mock import patch

import pytest

from utama_core.config.field_params import STANDARD_FIELD_DIMS
from utama_core.engine.context import TickContext
from utama_core.entities.data.vector import Vector2D, Vector3D
from utama_core.entities.game import Field, Game, GameHistory
from utama_core.entities.game.ball import Ball
from utama_core.entities.game.game_frame import GameFrame
from utama_core.entities.game.robot import Robot
from utama_core.motion_planning.src.common.motion_controller import MotionController
from utama_core.shared.pass_and_score_geometry import reset_possession_state
from utama_core.tactics.switch_of_play import (
    _SETTLE_DEBOUNCE_TICKS,
    SwitchOfPlayMem,
    SwitchOfPlayTactic,
    _debounced_settled,
    _settled_at,
)


class _FakeVector2D(Vector2D):
    def to_2d(self) -> Vector2D:
        return Vector2D(self.x, self.y)


@dataclass
class _FakeRobot:
    p: Vector2D
    v: Vector2D = field(default_factory=lambda: Vector2D(0, 0))


class _FakeGame:
    def __init__(self, robot_positions: dict[int, tuple[Vector2D, Vector2D]]):
        self.friendly_robots = {rid: _FakeRobot(p=pos, v=vel) for rid, (pos, vel) in robot_positions.items()}


_TARGET = Vector2D(-2.475, -2.4)
_AT_TARGET = Vector2D(-2.5, -2.4)  # well inside _ARRIVAL_POSITION_TOLERANCE (0.15m)
_SLOW = Vector2D(0.02, 0.0)  # under _ARRIVAL_SPEED_THRESHOLD (0.1 m/s)
_FAST = Vector2D(0.2, 0.0)  # over _ARRIVAL_SPEED_THRESHOLD


def test_settled_at_requires_both_position_and_speed():
    game = _FakeGame({1: (_AT_TARGET, _FAST)})
    assert _settled_at(game, 1, _TARGET) is False

    game = _FakeGame({1: (_AT_TARGET, _SLOW)})
    assert _settled_at(game, 1, _TARGET) is True


def test_debounced_settled_requires_sustained_ticks_not_one_instant():
    """A single in-tolerance, under-threshold tick must not immediately
    report ready — that was never the bug (a single instantaneous read
    already worked when it happened to land True); the bug is a robot that
    keeps *dropping back out* before the pass leg can act on it."""
    mem = SwitchOfPlayMem()
    game = _FakeGame({1: (_AT_TARGET, _SLOW)})

    for _ in range(_SETTLE_DEBOUNCE_TICKS - 1):
        assert _debounced_settled(game, 1, _TARGET, mem) is False
    assert _debounced_settled(game, 1, _TARGET, mem) is True


def test_debounced_settled_survives_oscillating_speed_around_threshold():
    """Live-traced bug: speed flapping above/below _ARRIVAL_SPEED_THRESHOLD
    every tick (0.03-0.22 m/s observed) never let a bare `_settled_at` read
    stay True long enough for `_pass_exec` to start — 51 True flips over
    9.4s, longest run 0.53s. The debounced gate must still reach True within
    a bounded number of ticks once the robot is genuinely oscillating in
    place near the target. Strict tick-by-tick alternation (the worst case)
    must never accumulate enough to settle, since the leaky counter gains 1
    on a hit and loses only 1 on a miss — alternating hit/miss nets to a
    steady 0-1, never climbing toward `_SETTLE_DEBOUNCE_TICKS`."""
    mem = SwitchOfPlayMem()
    game_slow = _FakeGame({1: (_AT_TARGET, _SLOW)})
    game_fast = _FakeGame({1: (_AT_TARGET, _FAST)})

    # Alternating fast/slow never accumulates a long enough streak.
    for _ in range(20):
        result = _debounced_settled(game_slow, 1, _TARGET, mem)
        assert result is False
        assert mem.settled_ticks <= 1
        result = _debounced_settled(game_fast, 1, _TARGET, mem)
        assert result is False
        assert mem.settled_ticks == 0

    # Once the oscillation actually stops (sustained slow ticks), it settles.
    for _ in range(_SETTLE_DEBOUNCE_TICKS - 1):
        assert _debounced_settled(game_slow, 1, _TARGET, mem) is False
    assert _debounced_settled(game_slow, 1, _TARGET, mem) is True


def test_debounced_settled_leaks_by_one_on_a_single_stray_miss():
    """The bug this fix actually targets: a live-traced "relay" window had
    True-run lengths of `[0.067, 0.067, 0.1, 0.067, 0.1, 0.017, ...]`s (4-6
    ticks at 60Hz) — almost every run landing at or just under the 6-tick
    threshold, so a single stray miss near the end of an otherwise-converged
    streak previously wiped `settled_ticks` back to 0 and restarted the
    entire wait. A single miss must now only cost 1 tick of progress, not
    all of it, so a streak of 5 hits + 1 stray miss + 1 more hit settles one
    tick later rather than needing 6 more consecutive hits from scratch."""
    mem = SwitchOfPlayMem()
    game_slow = _FakeGame({1: (_AT_TARGET, _SLOW)})
    game_fast = _FakeGame({1: (_AT_TARGET, _FAST)})

    for _ in range(_SETTLE_DEBOUNCE_TICKS - 1):
        assert _debounced_settled(game_slow, 1, _TARGET, mem) is False
    assert mem.settled_ticks == _SETTLE_DEBOUNCE_TICKS - 1

    # One stray miss costs 1 tick of progress, not the whole streak.
    assert _debounced_settled(game_fast, 1, _TARGET, mem) is False
    assert mem.settled_ticks == _SETTLE_DEBOUNCE_TICKS - 2

    # Resuming settled ticks reaches the threshold shortly after, not from scratch.
    assert _debounced_settled(game_slow, 1, _TARGET, mem) is False
    assert _debounced_settled(game_slow, 1, _TARGET, mem) is True


def test_debounced_settled_resets_on_sustained_position_leaving_tolerance():
    """A genuinely-departed robot (not just one stray miss tick) must still
    reach 0 and require a fresh full debounce window — the leak only
    tolerates brief flicker, not a real reposition."""
    mem = SwitchOfPlayMem()
    far = Vector2D(0.0, 0.0)
    game_at = _FakeGame({1: (_AT_TARGET, _SLOW)})
    game_far = _FakeGame({1: (far, _SLOW)})

    for _ in range(_SETTLE_DEBOUNCE_TICKS - 1):
        _debounced_settled(game_at, 1, _TARGET, mem)
    assert mem.settled_ticks == _SETTLE_DEBOUNCE_TICKS - 1

    for _ in range(_SETTLE_DEBOUNCE_TICKS):
        _debounced_settled(game_far, 1, _TARGET, mem)
    assert mem.settled_ticks == 0


# ---------------------------------------------------------------------------
# "relay" source ball-recovery gate (SwitchOfPlayMem.source_had_ball)
# ---------------------------------------------------------------------------


class _NullMotionController(MotionController):
    """Never actually reached meaningfully -- these tests only inspect which
    of `go_to_ball`/`move` was called for the source robot, not the returned
    velocity itself. Mirrors `test_decoy_and_overload_tactic.py`'s stub."""

    def __init__(self):
        super().__init__(mode="rsim")

    def calculate(self, game, robot_id, target_pos, target_oren):
        return Vector2D(0.0, 0.0), 0.0


@pytest.fixture(autouse=True)
def _reset_possession_hysteresis():
    """`has_ball(..., visual=True)`'s `_POSSESSION_STATE` (module-level,
    keyed by robot_id) carries hysteresis across calls -- reset it around
    every test in this section so one test's outcome can't leak into the
    next via a stale committed-possession state for robot_id 3."""
    reset_possession_state(3)
    yield
    reset_possession_state(3)


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


def _make_relay_game(source_pos: Vector2D, source_has_ball: bool) -> Game:
    """Two-robot-mode "relay": source_id == carrier_id (3), runner_id (1) far
    from its weak-side target so `runner_ready` is False and `tick()` enters
    the source-hold-or-recover branch under test. `source_pos` sits at the
    live-traced bug's ~0.12m distance from the ball by default in the tests
    below (well inside `_BALL_RECOVERY_RADIUS`=0.3m, outside dribbler-contact
    range) — has_ball is the thing under test, not distance.
    """
    ball_pos = Vector2D(0.0, 0.0)
    friendly = {
        3: _robot(3, source_pos.x, source_pos.y, True, has_ball=source_has_ball),
        1: _robot(1, -3.0, -3.0, True, has_ball=False),  # far from any weak-side target: runner never ready
    }
    zv = Vector3D(0, 0, 0)
    ball = Ball(p=Vector3D(ball_pos.x, ball_pos.y, 0.0), v=zv, a=zv)
    frame = GameFrame(
        ts=0.0, my_team_is_yellow=True, my_team_is_right=True, friendly_robots=friendly, enemy_robots={}, ball=ball
    )
    field = Field(
        my_team_is_right=True, field_dims=STANDARD_FIELD_DIMS, field_bounds=STANDARD_FIELD_DIMS.full_field_bounds
    )
    return Game(past=GameHistory(max_history=20), current=frame, field=field)


def _relay_mem(source_had_ball: bool) -> SwitchOfPlayMem:
    return SwitchOfPlayMem(
        phase="relay",
        carrier_id=3,
        pivot_id=3,  # two_robot_mode: pivot_id == carrier_id collapses pivot into runner
        runner_id=1,
        weak_side=1,
        source_had_ball=source_had_ball,
    )


def test_relay_source_that_never_had_ball_falls_through_to_go_to_ball():
    """The live-traced bug: source robot 0.12m from the ball (inside
    `_BALL_RECOVERY_RADIUS`), `has_ball` False, and never held it this
    episode (`source_had_ball=False`). Pre-fix, the bare distance check alone
    sent this into the "hold, face the runner" branch forever, so the robot
    never called `go_to_ball` again and could never actually acquire the
    ball. Post-fix it must call `go_to_ball`."""
    game = _make_relay_game(source_pos=Vector2D(0.12, 0.0), source_has_ball=False)
    ctx = TickContext(motion_controller=_NullMotionController())
    tactic = SwitchOfPlayTactic()
    mem = _relay_mem(source_had_ball=False)

    with (
        patch("utama_core.tactics.switch_of_play.go_to_ball") as mock_go_to_ball,
        patch("utama_core.tactics.switch_of_play.move") as mock_move,
    ):
        mock_go_to_ball.return_value = object()
        mock_move.return_value = object()
        _commands, new_mem = tactic.tick(game, ctx, (3, 1), mem)

    mock_go_to_ball.assert_called_once()
    mock_move.assert_not_called()
    assert new_mem.source_had_ball is False


def test_relay_source_with_live_has_ball_holds_and_faces_runner():
    """Positive control: `has_ball` True right now must always take the hold
    branch (and latch `source_had_ball`), regardless of prior history.

    `has_ball(..., visual=True)` (what every call site in switch_of_play.py
    uses) is a geometric forward/lateral box relative to `robot.orientation`,
    not the raw `Robot.has_ball` IR field -- `_robot()`'s orientation
    defaults to 0.0 (facing +x), so placing the robot at (-0.05, 0.0) against
    a ball at the origin puts the ball 0.05m directly in front of the
    chassis, inside the acquire box (`_ACQUIRE_FORWARD_MAX`=0.14,
    `_ACQUIRE_LATERAL_MAX`=0.05)."""
    game = _make_relay_game(source_pos=Vector2D(-0.05, 0.0), source_has_ball=True)
    ctx = TickContext(motion_controller=_NullMotionController())
    tactic = SwitchOfPlayTactic()
    mem = _relay_mem(source_had_ball=False)

    with (
        patch("utama_core.tactics.switch_of_play.go_to_ball") as mock_go_to_ball,
        patch("utama_core.tactics.switch_of_play.move") as mock_move,
    ):
        mock_go_to_ball.return_value = object()
        mock_move.return_value = object()
        _commands, new_mem = tactic.tick(game, ctx, (3, 1), mem)

    mock_move.assert_called_once()
    mock_go_to_ball.assert_not_called()
    assert new_mem.source_had_ball is True


def test_relay_source_that_previously_held_ball_gets_recovery_grace():
    """The behaviour `_BALL_RECOVERY_RADIUS` is actually meant to preserve:
    a source that genuinely held the ball earlier this "relay" episode
    (`source_had_ball=True`) but has a single `has_ball=False` blip (rsim
    ejection noise) must still take the hold branch while within the
    recovery radius, not immediately fall back to `go_to_ball` and discard
    its aim/hold progress."""
    game = _make_relay_game(source_pos=Vector2D(0.12, 0.0), source_has_ball=False)
    ctx = TickContext(motion_controller=_NullMotionController())
    tactic = SwitchOfPlayTactic()
    mem = _relay_mem(source_had_ball=True)

    with (
        patch("utama_core.tactics.switch_of_play.go_to_ball") as mock_go_to_ball,
        patch("utama_core.tactics.switch_of_play.move") as mock_move,
    ):
        mock_go_to_ball.return_value = object()
        mock_move.return_value = object()
        _commands, new_mem = tactic.tick(game, ctx, (3, 1), mem)

    mock_move.assert_called_once()
    mock_go_to_ball.assert_not_called()
    assert new_mem.source_had_ball is True


def test_relay_source_beyond_recovery_radius_falls_through_even_if_previously_held():
    """The recovery grace is still bounded by distance -- a source that held
    the ball long ago but has since drifted well clear of it (beyond
    `_BALL_RECOVERY_RADIUS`) must not hold-and-face-runner indefinitely."""
    game = _make_relay_game(source_pos=Vector2D(1.0, 0.0), source_has_ball=False)
    ctx = TickContext(motion_controller=_NullMotionController())
    tactic = SwitchOfPlayTactic()
    mem = _relay_mem(source_had_ball=True)

    with (
        patch("utama_core.tactics.switch_of_play.go_to_ball") as mock_go_to_ball,
        patch("utama_core.tactics.switch_of_play.move") as mock_move,
    ):
        mock_go_to_ball.return_value = object()
        mock_move.return_value = object()
        tactic.tick(game, ctx, (3, 1), mem)

    mock_go_to_ball.assert_called_once()
    mock_move.assert_not_called()


class _PerRobotMotionController(MotionController):
    def __init__(self):
        super().__init__(mode="rsim")
        self.targets: dict = {}

    def calculate(self, game, robot_id, target_pos, target_oren):
        self.targets[robot_id] = target_pos
        return Vector2D(0.0, 0.0), 0.0


def test_relay_runner_meets_a_rolling_pass_instead_of_returning_to_its_spot():
    """Once the relay pass is rolling, the runner steps onto the ball's path; that
    unsettles it from its weak-side spot, which used to send it back there while the
    ball rolled past (tournament_20260927_220824: runners drifted from 0.05 to 0.16 m
    off the path)."""
    friendly = {
        3: _robot(3, 0.3, 0.0, True, has_ball=False),
        1: _robot(1, -1.0, 0.07, True, has_ball=False),
    }
    ball = Ball(p=Vector3D(0.0, 0.0, 0.0), v=Vector3D(-3.0, 0.0, 0.0), a=Vector3D(0, 0, 0))
    frame = GameFrame(
        ts=0.0, my_team_is_yellow=True, my_team_is_right=True, friendly_robots=friendly, enemy_robots={}, ball=ball
    )
    field = Field(
        my_team_is_right=True, field_dims=STANDARD_FIELD_DIMS, field_bounds=STANDARD_FIELD_DIMS.full_field_bounds
    )
    game = Game(past=GameHistory(max_history=20), current=frame, field=field)
    ctx = TickContext(motion_controller=_PerRobotMotionController())

    commands, _mem = SwitchOfPlayTactic().tick(game, ctx, (3, 1), _relay_mem(source_had_ball=True))

    target = ctx.motion_controller.targets[1]
    assert target.x == pytest.approx(-1.0)
    assert target.y == pytest.approx(0.0, abs=1e-6)
    assert commands[1].dribble


def test_a_lone_carrier_at_the_ball_plays_it_instead_of_holding_it():
    """Solo allocation (one robot in the slot), the live-traced stall: the robot
    stopped 0.13 m from a stationary ball, facing it, inside the visual has_ball
    box but short of contact, and "held" it where it stood for 10 s until
    no_progress (tournament_20261005_170958, overload_press vs split_shape t=479).
    With an open lane at the enemy goal it must shoot, not stand still."""
    zv = Vector3D(0, 0, 0)
    # my_team_is_right: we attack the goal at -x, so a robot east of the ball faces it and the goal
    carrier = Robot(
        id=3,
        is_friendly=True,
        has_ball=False,
        p=Vector2D(0.13, 0.0),
        v=Vector2D(0, 0),
        a=Vector2D(0, 0),
        orientation=math.pi,
    )
    frame = GameFrame(
        ts=0.0,
        my_team_is_yellow=True,
        my_team_is_right=True,
        friendly_robots={3: carrier},
        enemy_robots={},
        ball=Ball(p=Vector3D(0.0, 0.0, 0.0), v=zv, a=zv),
    )
    field = Field(
        my_team_is_right=True, field_dims=STANDARD_FIELD_DIMS, field_bounds=STANDARD_FIELD_DIMS.full_field_bounds
    )
    game = Game(past=GameHistory(max_history=20), current=frame, field=field)

    commands, mem = SwitchOfPlayTactic().tick(
        game, TickContext(motion_controller=_NullMotionController()), (3,), SwitchOfPlayMem()
    )

    assert commands[3].kick
    assert mem.phase == "assess"  # nothing to commit to with one robot
