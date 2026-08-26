"""Tests for `utama_core.replay.stuck_detector` (gap #11 prototype)."""

from __future__ import annotations

import math
import pickle

import pytest

from utama_core.entities.data.vector import Vector2D, Vector3D
from utama_core.entities.game import Ball, GameFrame, Robot
from utama_core.replay.entities import ReplayMetadata
from utama_core.replay.stuck_detector import find_stuck_windows

_TICK_HZ = 60


def _robot(rid: int, x: float, y: float) -> Robot:
    return Robot(
        id=rid, is_friendly=True, has_ball=False, p=Vector2D(x, y), v=Vector2D(0, 0), a=Vector2D(0, 0), orientation=0.0
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
