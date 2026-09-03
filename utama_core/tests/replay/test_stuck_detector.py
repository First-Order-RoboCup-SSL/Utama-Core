"""Tests for `utama_core.replay.stuck_detector` (gap #11 prototype)."""

from __future__ import annotations

import math
import pickle
from pathlib import Path

import pytest

from utama_core.entities.data.vector import Vector2D, Vector3D
from utama_core.entities.game import Ball, GameFrame, Robot
from utama_core.replay.entities import ReplayMetadata
from utama_core.replay.stuck_detector import find_stuck_windows

_TICK_HZ = 60

_REPO_ROOT = Path(__file__).resolve().parents[3]
# Real tournament replays used by a couple of tests below to pin behaviour
# against actual match data (see `docs/STRATEGY_DEVELOPMENT.md`'s
# Observability section) rather than a synthetic fixture alone.
# `replays/` is gitignored (not shipped with the repo), so these are
# skipped rather than failing when the directory isn't present locally.
_RESTART_STALL_REPLAY = _REPO_ROOT / "replays/tournament_20260903_122808/high_line_zone_vs_overload_flow.npz"
_HIGH_TRAVEL_REPLAY = _REPO_ROOT / "replays/tournament_20260903_112025/clear_press_plus_vs_give_and_go_solo.pkl"


def _robot(rid: int, x: float, y: float, *, has_ball: bool = False) -> Robot:
    return Robot(
        id=rid,
        is_friendly=True,
        has_ball=has_ball,
        p=Vector2D(x, y),
        v=Vector2D(0, 0),
        a=Vector2D(0, 0),
        orientation=0.0,
    )


def _write_replay(path, frame_fn, n_ticks: int, dt: float = 1.0 / _TICK_HZ):
    with open(path, "wb") as f:
        pickle.dump(ReplayMetadata(my_team_is_yellow=True, exp_friendly=1, exp_enemy=0), f)
        for i in range(n_ticks):
            pickle.dump(frame_fn(i * dt, i), f)


def test_flags_frozen_ball_with_oscillating_robot(tmp_path):
    """Ball parked; robot 0 oscillates at 1Hz with 0.4m amplitude — a stuck point."""

    def frame(ts: float, i: int) -> GameFrame:
        osc_x = 0.4 * math.sin(2 * math.pi * 1.0 * ts)
        return GameFrame(
            ts=ts,
            my_team_is_yellow=True,
            my_team_is_right=True,
            friendly_robots={0: _robot(0, osc_x, 0.0)},
            enemy_robots={},
            ball=Ball(p=Vector3D(1.0, 1.0, 0.0), v=Vector3D(0, 0, 0), a=Vector3D(0, 0, 0)),
        )

    path = tmp_path / "stuck.pkl"
    _write_replay(path, frame, n_ticks=6 * _TICK_HZ)  # 6s

    windows = find_stuck_windows(path, window_s=3.0, stride_s=1.0, min_duration_s=3.0)

    assert windows, "expected at least one stuck window to be flagged"
    assert all(0 in w.oscillating_robot_ids for w in windows)
    assert all(w.ball_std < 0.05 for w in windows)


def test_does_not_flag_normal_play(tmp_path):
    """Ball moving steadily, robot chasing it — ordinary play, not stuck."""

    def frame(ts: float, i: int) -> GameFrame:
        return GameFrame(
            ts=ts,
            my_team_is_yellow=True,
            my_team_is_right=True,
            friendly_robots={0: _robot(0, ts * 0.5, 0.0)},
            enemy_robots={},
            ball=Ball(p=Vector3D(ts * 0.6, 0.0, 0.0), v=Vector3D(0.6, 0, 0), a=Vector3D(0, 0, 0)),
        )

    path = tmp_path / "normal.pkl"
    _write_replay(path, frame, n_ticks=6 * _TICK_HZ)

    windows = find_stuck_windows(path, window_s=3.0, stride_s=1.0, min_duration_s=3.0)
    assert windows == []


def test_does_not_flag_robot_settling_to_a_stop(tmp_path):
    """Ball frozen (legal stoppage), robot converging to a hold point — not oscillation."""

    def frame(ts: float, i: int) -> GameFrame:
        settled_x = 1.0 * math.exp(-ts)  # decays toward 0, no oscillation
        return GameFrame(
            ts=ts,
            my_team_is_yellow=True,
            my_team_is_right=True,
            friendly_robots={0: _robot(0, settled_x, 0.0)},
            enemy_robots={},
            ball=Ball(p=Vector3D(2.0, 2.0, 0.0), v=Vector3D(0, 0, 0), a=Vector3D(0, 0, 0)),
        )

    path = tmp_path / "settling.pkl"
    _write_replay(path, frame, n_ticks=6 * _TICK_HZ)

    windows = find_stuck_windows(path, window_s=3.0, stride_s=1.0, min_duration_s=3.0)
    assert windows == []


def test_empty_replay_returns_no_windows(tmp_path):
    path = tmp_path / "empty.pkl"
    with open(path, "wb") as f:
        pickle.dump(ReplayMetadata(my_team_is_yellow=True, exp_friendly=1, exp_enemy=0), f)

    assert find_stuck_windows(path) == []


def _write_possessor_replay(path, hold_duration_s: float, dt: float = 1.0 / _TICK_HZ):
    """Robot 0 picks up a motionless ball at t=2s and holds it (itself
    motionless) for `hold_duration_s`, then releases it and the ball rolls
    away. Robot 1 oscillates throughout (same signal
    `test_flags_frozen_ball_with_oscillating_robot` uses) so a long enough
    hold has an oscillation signal to actually flag — mirroring the real
    replay this scenario is modelled on (`clear_danger_vs_clear_press_plus`,
    see `_DEFAULT_POSSESSOR_STILL_MIN_S`'s docstring), where robot 0 sitting
    on the ball coincided with other robots genuinely moving around it, not
    a frozen-everything scene. Used to pin the `restart_stall_s`-scale
    duration floor on `_possessor_is_motionless`'s bypass.
    """
    hold_start = 2.0
    hold_end = hold_start + hold_duration_s
    n_ticks = int((hold_end + 4.0) / dt)

    def frame(ts: float, i: int) -> GameFrame:
        holding = hold_start <= ts < hold_end
        if ts < hold_end:
            bx = 0.0
        else:
            bx = (ts - hold_end) * 0.5  # ball rolls away once released
        osc_x = 0.4 * math.sin(2 * math.pi * 1.0 * ts)
        return GameFrame(
            ts=ts,
            my_team_is_yellow=True,
            my_team_is_right=True,
            friendly_robots={
                0: _robot(0, 0.0, 0.0, has_ball=holding),
                1: _robot(1, osc_x, 2.0),
            },
            enemy_robots={},
            ball=Ball(p=Vector3D(bx, 0.0, 0.0), v=Vector3D(0, 0, 0), a=Vector3D(0, 0, 0)),
        )

    _write_replay(path, frame, n_ticks=n_ticks, dt=dt)


def test_possessor_motionless_bypass_requires_sustained_hold(tmp_path):
    """Regression test for the over-flagging fix: a robot parked motionless
    on the ball for an ordinary few-second ball-control moment (well under
    `_DEFAULT_POSSESSOR_STILL_MIN_S`) must NOT bypass the possession
    exclusion — only a hold sustained anywhere near that floor should. This
    pins the exact boundary the fix introduced: before adding the
    `possessor_still_min_s` duration check, `_possessor_is_motionless` fired
    on any window where a single robot held the ball motionless for
    `_CONTINUOUS_POSSESSION_FRACTION` of that one window regardless of how
    long the hold actually lasted, flagging this 4s hold exactly like a
    genuine multi-second stall (see `_DEFAULT_POSSESSOR_STILL_MIN_S`'s
    docstring for the real-replay case, `clear_danger_vs_clear_press_plus`,
    that caught it)."""
    path = tmp_path / "short_hold.pkl"
    _write_possessor_replay(path, hold_duration_s=4.0)

    windows = find_stuck_windows(path, window_s=3.0, stride_s=1.0, min_duration_s=3.0)
    assert windows == [], f"a short, ordinary ball-control hold must not be flagged as stuck, got {windows}"


def test_possessor_motionless_bypass_fires_on_long_hold(tmp_path):
    """The flip side of the regression test above: a robot parked
    motionless on the ball for well beyond `_DEFAULT_POSSESSOR_STILL_MIN_S`
    (here 30s, twice the 15s floor) is exactly the stall the possession
    exclusion's bypass exists to still catch — this must remain flagged."""
    path = tmp_path / "long_hold.pkl"
    _write_possessor_replay(path, hold_duration_s=30.0)

    windows = find_stuck_windows(path, window_s=3.0, stride_s=1.0, min_duration_s=3.0)
    assert windows, "a robot parked motionless on the ball for 30s should be flagged as stuck"
    assert all(w.kind == "oscillation" for w in windows)


@pytest.mark.skipif(not _RESTART_STALL_REPLAY.exists(), reason="real tournament replay not present locally")
def test_real_replay_restart_stall_window():
    """`high_line_zone_vs_overload_flow` (2026-09-03) stalls in a referee
    restart from ~32s to the end of the match at 65s — the case
    `kind="restart_stall"` was added to catch (see module docstring)."""
    windows = find_stuck_windows(_RESTART_STALL_REPLAY)
    restart_windows = [w for w in windows if w.kind == "restart_stall"]
    assert restart_windows, f"expected a restart_stall window, got {windows}"
    assert any(30.0 <= w.t_start <= 35.0 for w in restart_windows), (
        f"expected a restart_stall window starting between 30s and 35s, got " f"{[w.t_start for w in restart_windows]}"
    )


@pytest.mark.skipif(not _HIGH_TRAVEL_REPLAY.exists(), reason="real tournament replay not present locally")
def test_real_replay_high_ball_travel_no_oscillation_windows():
    """`clear_press_plus_vs_give_and_go_solo` (49.88m of ball travel per its
    `.stats.json`, no `stall_events` recorded by the in-match watchdog) is
    ordinary, non-stuck play — a calibration sweep across every replay
    under `replays/` (2026-09-03, see stuck_detector.py's `_main`/
    `_scan_replays`) used this and similar high-ball-travel matches to
    retune `_possessor_is_motionless`'s bypass after it was over-flagging
    ordinary short ball-control moments on nearly every match; this pins
    that it stays at zero oscillation windows."""
    windows = find_stuck_windows(_HIGH_TRAVEL_REPLAY)
    oscillation_windows = [w for w in windows if w.kind == "oscillation"]
    assert oscillation_windows == [], f"expected no oscillation windows, got {oscillation_windows}"
