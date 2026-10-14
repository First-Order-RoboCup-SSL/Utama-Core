import importlib.util
from pathlib import Path

import pytest

_TOOL = Path(__file__).resolve().parents[3] / "tools" / "bench_vs_standings.py"
_spec = importlib.util.spec_from_file_location("bench_vs_standings", _TOOL)
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)


def test_spearman_of_the_same_and_the_reversed_order():
    assert tool.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert tool.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)


def test_spearman_gives_tied_values_their_average_rank():
    # ranks of [1, 2, 2, 3] are [1, 2.5, 2.5, 4]
    assert tool._ranks([1, 2, 2, 3]) == [1, 2.5, 2.5, 4]
    assert tool.spearman([1, 2, 2, 3], [1, 2, 3, 4]) == pytest.approx(0.9486833, abs=1e-6)


def test_standings_are_per_match_points_and_goal_difference():
    summary = {"strategies": {"a": {"matches": 4, "wins": 2, "draws": 1, "goals_for": 5, "goals_against": 1}}}
    assert tool.standings(summary) == {"a": {"points": 7 / 4, "goal_diff": 1.0}}


def test_bench_scores_average_each_strategy_s_outcomes_and_skip_errored_runs():
    def play(job):
        strategy, _opponent, scenario = job
        if scenario == "broken":
            return strategy, 0, False
        return strategy, (1 if strategy == "good" else -1), True

    scores = tool.bench_scores(["good", "bad"], ["opp"], ["s1", "s2", "broken"], workers=1, play=play)
    assert scores == {"good": 1.0, "bad": -1.0}
