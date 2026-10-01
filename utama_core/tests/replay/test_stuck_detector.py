"""Tests for `utama_core.replay.stuck_detector` (gap #11 prototype)."""

from __future__ import annotations

import json
import math
import pickle
from pathlib import Path

import pytest

from utama_core.entities.data.vector import Vector2D, Vector3D
from utama_core.entities.game import Ball, GameFrame, Robot
from utama_core.replay.entities import ReplayMetadata
from utama_core.replay.stuck_detector import find_stuck_windows

_TICK_HZ = 60


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


def _still_scene(ts: float, i: int) -> GameFrame:
    return GameFrame(
        ts=ts,
        my_team_is_yellow=True,
        my_team_is_right=True,
        friendly_robots={0: _robot(0, -1.0, 0.0)},
        enemy_robots={},
        ball=Ball(p=Vector3D(1.0, 1.0, 0.0), v=Vector3D(0, 0, 0), a=Vector3D(0, 0, 0)),
    )


def _write_referee_sidecar(replay_path, events):
    sidecar = replay_path.with_suffix("").with_suffix(".intentions.jsonl")
    sidecar.write_text("".join(json.dumps({"event": "referee", "sim_time": t, "command": c}) + "\n" for t, c in events))


@pytest.mark.parametrize("held_s,flagged", [(14.0, False), (16.0, True)])
def test_a_restart_held_past_restart_stall_s_is_a_restart_stall(tmp_path, held_s, flagged):
    """A kickoff that never advances, with the ball and every robot still, is flagged
    once it has been held for `restart_stall_s` (15 s by default) and not before. The
    oscillation signal can't see it: nothing moves."""
    path = tmp_path / "match.pkl"
    _write_replay(path, _still_scene, n_ticks=int((2.0 + held_s + 1.0) * _TICK_HZ))
    _write_referee_sidecar(
        path, [(0.0, "NORMAL_START"), (2.0, "PREPARE_KICKOFF_YELLOW"), (2.0 + held_s, "NORMAL_START")]
    )

    windows = [w for w in find_stuck_windows(path) if w.kind == "restart_stall"]

    assert bool(windows) is flagged, windows
    if flagged:
        assert windows[0].t_start == pytest.approx(2.0, abs=0.05)
