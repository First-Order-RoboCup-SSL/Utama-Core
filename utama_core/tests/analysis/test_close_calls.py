"""`close_calls` finds where two robots came near, how fast they were closing, and the referee
command in force; one robot read twice is not a close call."""

from __future__ import annotations

import numpy as np
import pytest

from utama_core.analysis.close_calls import CLOSE_M, GHOST_M, close_calls


def _track(t: np.ndarray, x: np.ndarray, y: float = 0.0) -> np.ndarray:
    return np.column_stack([t, np.zeros_like(t), x, np.full_like(t, y), np.zeros_like(t)])


T = np.arange(0.0, 4.0, 1 / 60)


def test_two_robots_meeting_head_on_are_one_close_call_with_their_closing_speed():
    # Each at 0.5 m/s towards the other until 1.0 m apart... then 0.2 m, then they part.
    a = np.where(T < 2.0, -1.0 + 0.5 * T, 0.0 - 0.5 * (T - 2.0)) - 0.1
    b = -a
    rows = close_calls(
        {("yellow", 0): _track(T, a), ("blue", 0): _track(T, b)},
        [(0.0, "HALT"), (1.0, "FORCE_START")],
    )

    assert len(rows) == 1
    r = rows[0]
    assert r["nearest_m"] == pytest.approx(0.2, abs=0.02)
    assert r["approach_mps"] == pytest.approx(1.0, abs=0.05)
    assert r["command"] == "FORCE_START" and r["after_command_s"] == pytest.approx(r["t"] - 1.0)


def test_robots_never_nearer_than_close_m_give_none():
    rows = close_calls(
        {("yellow", 0): _track(T, np.zeros_like(T)), ("blue", 0): _track(T, np.full_like(T, CLOSE_M + 0.01))}
    )
    assert rows == []


def test_one_robot_read_as_two_patterns_is_not_a_close_call():
    rows = close_calls(
        {("yellow", 1): _track(T, np.zeros_like(T)), ("blue", 3): _track(T, np.full_like(T, GHOST_M - 0.01))}
    )
    assert rows == []


def test_a_robot_unseen_is_not_compared_across_the_gap():
    seen = (T < 1.0) | (T > 3.0)
    a = _track(T[seen], np.zeros(seen.sum()))
    b = _track(T, np.where((T > 1.5) & (T < 2.5), 0.15, 1.0))  # comes near only while a is unseen
    assert close_calls({("yellow", 0): a, ("blue", 0): b}) == []
