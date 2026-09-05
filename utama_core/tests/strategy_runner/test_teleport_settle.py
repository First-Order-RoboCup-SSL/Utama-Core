"""Unit tests for `StrategyRunner._tick_teleport_settle` — the re-pin window
that counters the native rsim engine's post-teleport ball velocity spike.

Traced live in a tournament run
(replays/tournament_20260905_002133/counter_flow_vs_zone_fluid_LK,
2026-09-05): `strategy_runner.py`'s teleport-on-BALL_PLACEMENT/DIRECT_FREE/
STOP path calls `sim_controller.teleport_ball(x, y)` (zero velocity) to
instantly simulate ball placement. The native reset itself correctly zeroes
the ball's ODE body velocity, but the very next physics step after any reset
produces a large spurious velocity (tens of m/s observed) — a
contact-resolution artifact, not something the caller controls. That single
spike sent the ball rolling roughly a metre away from the placement target,
and since `GameStateMachine._ball_placement_done()` gates the
BALL_PLACEMENT_* -> next_command auto-advance purely on ball-to-target
distance, the restart never advanced — RESTART_STALL for the rest of that
match (215s).

Reproducing the exact native-engine trigger deterministically in a fast test
proved unreliable (the spike depends on rsim's contact-resolution state,
which the synthetic scenarios in test_referee_rsim.py did not consistently
hit) — so these tests exercise `_tick_teleport_settle`'s own state-machine
logic directly, injecting a synthetic spike via a fake `rsim_env.frame.ball`,
rather than depending on rsim's native physics to reproduce the spike.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from utama_core.config.enums import Mode
from utama_core.run.strategy_runner import (
    _TELEPORT_SETTLE_MAX_EXTENSIONS,
    _TELEPORT_SETTLE_TICKS,
    StrategyRunner,
)


def _make_bare_runner(ball_v_sequence: list[tuple[float, float]]) -> StrategyRunner:
    """A `StrategyRunner` with only the attributes `_tick_teleport_settle` and
    `_teleport_ball_and_settle` touch — bypasses `__init__` (which wires up
    vision/referee/replay machinery irrelevant to this unit) via `__new__`.

    `ball_v_sequence` is consumed one entry per `_tick_teleport_settle` call
    (via `rsim_env.frame.ball.v_x/v_y`), simulating what the raw sim frame
    would report at that point in the tick — the last entry repeats once
    exhausted, same as a settled ball holding still.
    """
    runner = StrategyRunner.__new__(StrategyRunner)
    runner.mode = Mode.RSIM
    runner.sim_controller = MagicMock()
    runner.rsim_env = SimpleNamespace(frame=SimpleNamespace(ball=SimpleNamespace(v_x=0.0, v_y=0.0)))
    runner._teleport_settle_target = None
    runner._teleport_settle_ticks_left = 0
    runner._teleport_settle_extensions_left = 0

    state = {"i": 0}

    def _advance_ball_v():
        i = min(state["i"], len(ball_v_sequence) - 1)
        vx, vy = ball_v_sequence[i]
        runner.rsim_env.frame.ball.v_x = vx
        runner.rsim_env.frame.ball.v_y = vy
        state["i"] += 1

    runner._advance_ball_v = _advance_ball_v
    return runner


def test_settle_window_re_pins_for_fixed_ticks_when_no_spike():
    """No spike ever observed -> re-pins exactly `_TELEPORT_SETTLE_TICKS`
    times, then stops (matches the well-behaved/no-bug case)."""
    runner = _make_bare_runner([(0.0, 0.0)] * 20)
    runner._teleport_ball_and_settle(1.0, 2.0)
    runner.sim_controller.teleport_ball.reset_mock()  # drop the initial call

    repin_calls = 0
    for _ in range(_TELEPORT_SETTLE_TICKS + 5):
        runner._advance_ball_v()
        before = runner.sim_controller.teleport_ball.call_count
        runner._tick_teleport_settle()
        if runner.sim_controller.teleport_ball.call_count > before:
            repin_calls += 1

    assert repin_calls == _TELEPORT_SETTLE_TICKS
    assert runner._teleport_settle_target is None


def test_settle_window_extends_while_spike_persists():
    """A spike observed right as the fixed window is about to close extends
    the window instead of letting the ball drift uncorrected — this is the
    mechanism that fixes the traced RESTART_STALL (a fixed-length-only window
    let the spike through on whichever tick it happened to land on)."""
    # Spike for the first `_TELEPORT_SETTLE_TICKS` checks (worse than the
    # traced case, which only spiked once), then clears.
    spike_ticks = _TELEPORT_SETTLE_TICKS + 2
    sequence = [(30.0, -5.0)] * spike_ticks + [(0.0, 0.0)] * 10
    runner = _make_bare_runner(sequence)
    runner._teleport_ball_and_settle(1.0, 2.0)
    runner.sim_controller.teleport_ball.reset_mock()

    ticks_run = 0
    while runner._teleport_settle_target is not None and ticks_run < 50:
        runner._advance_ball_v()
        runner._tick_teleport_settle()
        ticks_run += 1

    assert runner._teleport_settle_target is None, "settle window never closed"
    # Re-pinned through every spiking tick, not just the original fixed window.
    assert runner.sim_controller.teleport_ball.call_count > _TELEPORT_SETTLE_TICKS
    # Every re-pin call targeted the original placement target, never drifted.
    for call in runner.sim_controller.teleport_ball.call_args_list:
        assert call.args == (1.0, 2.0)


def test_settle_window_extension_is_bounded():
    """A ball that never stops "spiking" (e.g. genuinely fast in-flight at
    the moment of teleport, not a reset artifact) must not stall the re-pin
    loop forever — extensions are capped."""
    sequence = [(30.0, -5.0)] * 200
    runner = _make_bare_runner(sequence)
    runner._teleport_ball_and_settle(1.0, 2.0)
    runner.sim_controller.teleport_ball.reset_mock()

    ticks_run = 0
    while runner._teleport_settle_target is not None and ticks_run < 200:
        runner._advance_ball_v()
        runner._tick_teleport_settle()
        ticks_run += 1

    assert runner._teleport_settle_target is None, "unbounded spike must still terminate the settle window"
    # Fixed window + every extension, no more.
    assert runner.sim_controller.teleport_ball.call_count <= _TELEPORT_SETTLE_TICKS + _TELEPORT_SETTLE_MAX_EXTENSIONS


def test_tick_teleport_settle_is_noop_without_an_armed_target():
    runner = _make_bare_runner([(30.0, -5.0)] * 5)
    runner._tick_teleport_settle()
    runner.sim_controller.teleport_ball.assert_not_called()
