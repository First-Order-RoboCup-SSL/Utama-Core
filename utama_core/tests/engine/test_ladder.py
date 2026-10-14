"""`tools/evaluation/ladder.py`: the stopping rule, the reference pool and the fixtures. Pure:
no match is played."""

from __future__ import annotations

import json

import pytest

from tools.evaluation import ladder

MAX = 2 * len(ladder.SETTINGS)


def _r(a, b, sa, sb):
    return {"config_a": a, "config_b": b, "score_a": sa, "score_b": sb}


# --- stopping rule ---------------------------------------------------------------------------


@pytest.mark.parametrize("wins, losses", [(3, 0), (0, 3), (4, 0)])
def test_no_stop_before_min_matches_even_with_a_lead(wins, losses):
    assert not ladder.decided(wins, losses, ladder.MIN_MATCHES - 1, MAX)


def test_stops_at_min_matches_on_a_lead_of_exactly_lead():
    assert ladder.decided(ladder.LEAD, 0, ladder.MIN_MATCHES, MAX)
    assert ladder.decided(0, ladder.LEAD, ladder.MIN_MATCHES, MAX)
    assert ladder.decided(ladder.LEAD, 0, 5, MAX)  # 3W 2D


def test_does_not_stop_on_a_lead_one_short():
    assert not ladder.decided(ladder.LEAD - 1, 0, ladder.MIN_MATCHES, MAX)
    assert not ladder.decided(3, 1, 4, MAX)  # margin 2 with 4 left


def test_stops_once_the_sign_cannot_change():
    # margin 2 with 1 match left: the most the last match moves it is 1
    assert ladder.decided(4, 2, MAX - 1, MAX)
    # margin 2 with exactly 2 left could still end level: keep going
    assert not ladder.decided(3, 1, MAX - 2, MAX)


def test_level_pairing_plays_every_match():
    for played in range(ladder.MIN_MATCHES, MAX):
        assert not ladder.decided(1, 1, played, MAX)
    assert ladder.decided(1, 1, MAX, MAX)


# --- reference pool --------------------------------------------------------------------------

_RESULTS = [
    _r("a", "b", 2, 0),  # a beats b
    _r("a", "c", 1, 1),
    _r("a", "d", 3, 0),
    _r("b", "c", 0, 1),
    _r("b", "d", 1, 0),
    _r("c", "d", 0, 0),
]


def test_standings_points_and_goal_difference():
    table = ladder.standings(_RESULTS)
    assert table["a"] == {"points": 7, "goal_difference": 5, "matches": 3}
    assert table["c"] == {"points": 5, "goal_difference": 1, "matches": 3}
    assert table["b"] == {"points": 3, "goal_difference": -2, "matches": 3}
    assert table["d"] == {"points": 1, "goal_difference": -4, "matches": 3}


def test_pool_is_the_top_by_points_and_leaves_out_the_candidate():
    assert ladder.reference_pool(_RESULTS, "z", size=2) == ["a", "c"]
    assert ladder.reference_pool(_RESULTS, "a", size=2) == ["c", "b"]


def test_pool_leaves_out_retired_strategies(monkeypatch):
    monkeypatch.setattr(ladder, "RETIRED", {"a"})
    assert ladder.reference_pool(_RESULTS, "z", size=2) == ["c", "b"]


def test_pool_breaks_a_points_tie_on_goal_difference_then_name():
    results = [_r("x", "y", 3, 0), _r("y", "z", 1, 0), _r("z", "x", 1, 0)]  # 3 points each
    # x GD +2, z GD 0, y GD -2
    assert ladder.reference_pool(results, "none", size=3) == ["x", "z", "y"]
    level = [_r("p", "q", 1, 0), _r("q", "p", 1, 0)]  # same points, same GD
    assert ladder.reference_pool(level, "none", size=2) == ["p", "q"]


def _write_run(root, name, config_names, n_results, fuzz_seed=None):
    run = root / name
    run.mkdir()
    summary = {
        "config_names": config_names,
        "fuzz_seed": fuzz_seed,
        "results": [_r("a", "b", 0, 0)] * n_results,
    }
    (run / "summary.json").write_text(json.dumps(summary))
    return run / "summary.json"


def test_latest_round_robin_skips_pair_runs_fuzzed_runs_and_partial_runs(tmp_path):
    full = _write_run(tmp_path, "tournament_20261001_000000", ["a", "b", "c", "d"], 6)
    _write_run(tmp_path, "tournament_20261002_000000", ["a", "b"], 1)  # --pair
    _write_run(tmp_path, "tournament_20261003_000000", ["a", "b", "c", "d"], 6, fuzz_seed=1)
    _write_run(tmp_path, "tournament_20261004_000000", ["a", "b", "c", "d"], 5)  # one short
    assert ladder.latest_round_robin(tmp_path) == full


def test_latest_round_robin_none_when_there_is_none(tmp_path):
    assert ladder.latest_round_robin(tmp_path) is None


# --- fixtures --------------------------------------------------------------------------------


def test_a_mirrored_pair_swaps_the_teams_and_keeps_the_setting():
    assert ladder.fixtures("cand", "opp", (True, False)) == [
        ("cand", "opp", True, False),
        ("opp", "cand", True, False),
    ]


def test_the_round_robin_setting_comes_first_and_has_the_round_robin_file_tag():
    # The first mirrored pair is the round-robin's own match (and its --both-sides twin),
    # so a stored round-robin record is reused and its replay name matches.
    assert ladder.SETTINGS[0] == (True, True)
    assert ladder._suffix(True, True) == ""
    assert {ladder._suffix(*s) for s in ladder.SETTINGS[1:]} == {"_Rk", "_LK", "_Lk"}


def test_summarise_counts_from_the_candidates_side():
    matches = [
        {"result": _r("cand", "opp", 2, 1)},
        {"result": _r("opp", "cand", 1, 1)},
        {"result": _r("opp", "cand", 3, 0)},
    ]
    s = ladder.summarise(matches, "cand")
    assert (s["wins"], s["draws"], s["losses"], s["points"], s["goal_difference"]) == (1, 1, 1, 4, -2)
    assert s["points_per_match"] == pytest.approx(1.33)
