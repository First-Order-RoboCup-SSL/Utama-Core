import numpy as np

from utama_core.entities.referee.referee_command import RefereeCommand as C
from utama_core.replay.restart_outcomes import analyse_match, episodes, summarise

_BALL = (1.0, 0.0)


def _rows(*spans):
    """(command, n_ticks[, ball]) spans -> per-tick (t, command, ball) rows at 60 Hz."""
    rows, tick = [], 0
    for span in spans:
        cmd, n = span[0], span[1]
        ball = span[2] if len(span) > 2 else _BALL
        for _ in range(n):
            rows.append((tick / 60.0, cmd, ball))
            tick += 1
    return rows


def _outcomes(rows):
    return [(e["kind"], e["reached_normal_start"], e["outcome"]) for e in episodes(rows)]


def test_a_restart_is_taken_once_the_ball_moves_after_normal_start():
    rows = _rows((C.DIRECT_FREE_YELLOW, 10), (C.NORMAL_START, 5), (C.NORMAL_START, 5, (1.05, 0.0)))
    assert _outcomes(rows) == [("DIRECT_FREE", True, "taken")]


def test_the_ball_must_move_the_full_threshold_to_count_as_taken():
    rows = _rows((C.DIRECT_FREE_YELLOW, 10), (C.NORMAL_START, 5, (1.049, 0.0)))
    assert _outcomes(rows) == [("DIRECT_FREE", True, "match_ended")]


def test_a_penalty_stopped_before_normal_start_is_voided():
    # The 2026-09-28 keep-out bug: every PREPARE_PENALTY ended in STOP.
    rows = _rows((C.PREPARE_PENALTY_YELLOW, 60), (C.STOP, 5), (C.FORCE_START, 5))
    assert _outcomes(rows) == [("PREPARE_PENALTY", False, "voided")]


def test_stopped_before_kick_and_timeout_are_told_apart():
    rows = _rows(
        (C.DIRECT_FREE_BLUE, 5),
        (C.NORMAL_START, 5),
        (C.STOP, 5),
        (C.PREPARE_KICKOFF_YELLOW, 5),
        (C.NORMAL_START, 5),
        (C.FORCE_START, 5),
    )
    assert _outcomes(rows) == [
        ("DIRECT_FREE", True, "stopped_before_kick"),
        ("PREPARE_KICKOFF", True, "timeout"),
    ]


def test_summarise_counts_outcomes_per_kind():
    eps = [
        {"kind": "PREPARE_PENALTY", "reached_normal_start": False, "outcome": "voided"},
        {"kind": "PREPARE_PENALTY", "reached_normal_start": True, "outcome": "taken"},
        {"kind": "DIRECT_FREE", "reached_normal_start": True, "outcome": "taken"},
    ]
    s = summarise(eps)
    assert (s["restarts"], s["reached_normal_start"]) == (3, 2)
    assert s["by_kind"]["PREPARE_PENALTY"] == {"n": 2, "reached_normal_start": 1, "outcomes": {"voided": 1, "taken": 1}}


def test_analyse_match_reads_a_columnar_replay(tmp_path):
    n = 30
    commands = np.array([C.PREPARE_KICKOFF_YELLOW.value] * 10 + [C.NORMAL_START.value] * 20, dtype=np.int8)
    ball = np.zeros((n, 3))
    ball[20:, 0] = 0.2
    path = tmp_path / "a_vs_b.npz"
    np.savez(path, ts=100.0 + np.arange(n) / 60.0, has_referee=np.ones(n, bool), referee_command=commands, ball_p=ball)

    (ep,) = analyse_match(path)
    assert (ep["match"], ep["kind"], ep["outcome"]) == ("a_vs_b", "PREPARE_KICKOFF", "taken")
    assert ep["t"] == 0.0
