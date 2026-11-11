"""`tools/screen.py`'s fixed rule and the pure parts behind it; playing a start is not tested here."""

import random

import pytest

from tools import screen
from tools.screen import PairResult, Start


def _pair(change_gd: int, base_gd: int, fired: bool = True) -> PairResult:
    def measures(gd):
        return {"goals_for": max(gd, 0), "goals_against": max(-gd, 0), "upfield_m": 0.0}

    return PairResult(Start("r.npz", 10.0, 1), fired, 1.0 if fired else None, measures(change_gd), measures(base_gd))


def test_identical_trajectories_never_diverge():
    rows = [(0.0, 1.0), (0.5, 1.5)]
    assert screen.diverged_at(rows, list(rows)) is None


def test_divergence_is_the_first_tick_apart_by_more_than_the_tolerance():
    a = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]
    b = [(0.0, 0.0), (1.0, 1.0 + screen.SAME_M / 2), (2.0, 2.0 + 2 * screen.SAME_M)]
    assert screen.diverged_at(a, b) == 2


def test_a_robot_missing_in_one_trajectory_is_a_divergence():
    assert screen.diverged_at([(0.0, 0.0, 1.0)], [(0.0, 0.0)]) == 0
    assert screen.diverged_at([(0.0,), (1.0,)], [(0.0,)]) == 1


def test_fired_share_boundary():
    """At least `MIN_FIRED_SHARE` of the starts must fire: 6 of 60 is enough, 5 is not."""
    assert screen.call(60, [0.0] * 5) == "SKIP: did not fire"
    assert screen.call(60, [0.0] * 6) == "WORTH EVALUATING"
    assert screen.call(0, []) == "SKIP: did not fire"


def test_it_skips_only_when_goals_are_lower_by_more_than_two_standard_errors():
    clearly_worse = [-1.0] * 10 + [0.0] * 2
    assert screen.call(60, clearly_worse) == "SKIP: fewer goals in the starts"
    mean, se = screen.mean_se([-1.0, 1.0] * 5 + [-1.0])  # slightly lower, well within noise
    assert mean < 0 < mean + screen.GOAL_SE * se
    assert screen.call(60, [-1.0, 1.0] * 5 + [-1.0]) == "WORTH EVALUATING"


def test_a_change_with_more_goals_is_never_skipped():
    assert screen.call(60, [1.0] * 30) == "WORTH EVALUATING"


def test_goal_difference_change_is_change_minus_base():
    assert screen.goal_difference_changes([_pair(1, 0), _pair(0, 1), _pair(-1, -1)]) == [1, -1, 0]


def test_mean_se_edge_cases():
    assert screen.mean_se([]) == (0.0, 0.0)
    assert screen.mean_se([2.0]) == (2.0, 0.0)
    assert screen.mean_se([1.0, 3.0]) == pytest.approx((2.0, 1.0))


def test_drawn_times_leave_room_for_a_whole_start():
    times = screen.draw_times(random.Random(0), 100.0, 500)
    assert min(times) >= screen.EDGE_S
    assert max(times) <= 100.0 - screen.EDGE_S - screen.SECONDS


def test_report_counts_fired_starts_and_calls_on_them_only():
    results = [_pair(0, 0, fired=False)] * 50 + [_pair(-1, 0)] * 10
    lines, numbers = screen.report(results, 60)
    assert lines[0] == "fired in 10 of 60 starts"
    assert numbers["fired"] == 10
    assert numbers["call"] == "SKIP: fewer goals in the starts"


def _run(root, name, pairs):
    import json

    d = root / name
    d.mkdir()
    (d / "summary.json").write_text(json.dumps({"results": [{"config_a": a, "config_b": b} for a, b in pairs]}))
    for a, b in pairs:
        (d / f"{a}_vs_{b}.npz").touch()
    return d


def test_starts_come_from_the_latest_run_with_several_replays_of_the_strategy(tmp_path):
    """A one-match `--pair` run is newer but gives too few situations; it is the fallback only."""
    full = _run(tmp_path, "tournament_1", [("s", "x"), ("y", "s"), ("s", "z")])
    _run(tmp_path, "tournament_2", [("x", "s")])
    assert screen.latest_run_with("s", tmp_path) == full
    assert screen.latest_run_with("x", tmp_path).name == "tournament_2"


def test_replays_where_it_was_our_side_come_first(tmp_path):
    d = _run(tmp_path, "tournament_1", [("y", "s"), ("s", "x"), ("a", "b")])
    assert [p.name for p in screen.replays_with(d, "s")] == ["s_vs_x.npz", "y_vs_s.npz"]


def test_command_at_is_the_last_referee_command_at_or_before_t(tmp_path):
    import json

    rows = [
        {"event": "referee", "sim_time": 4.0, "command": "STOP"},
        {"event": "intention", "sim_time": 5.0},
        {"event": "referee", "sim_time": 0.0, "command": "NORMAL_START"},
    ]
    sidecar = tmp_path / "a_vs_b.intentions.jsonl"
    sidecar.write_text("\n".join(json.dumps(r) for r in rows) + "\nnot json\n")
    timeline = screen.referee_timeline(sidecar)
    assert timeline == [(0.0, "NORMAL_START"), (4.0, "STOP")]
    assert screen.command_at(timeline, -1.0) is None
    assert screen.command_at(timeline, 3.99) == "NORMAL_START"
    assert screen.command_at(timeline, 4.0) == "STOP"  # at the change, the new command, as the scenario reads it
