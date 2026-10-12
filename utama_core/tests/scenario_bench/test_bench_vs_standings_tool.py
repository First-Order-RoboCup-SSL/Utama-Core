import dataclasses
import importlib.util
from pathlib import Path

import pytest

from utama_core.scenario_bench.hand_authored_scenarios import (
    all_hand_authored_scenarios,
)

_TOOL = Path(__file__).resolve().parents[3] / "tools" / "bench_vs_standings.py"
_spec = importlib.util.spec_from_file_location("bench_vs_standings", _TOOL)
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)


def test_spearman_of_the_same_and_the_reversed_order():
    assert tool.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert tool.spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)


def test_spearman_gives_tied_values_their_average_rank():
    assert tool._ranks([5.0, 1.0, 5.0]) == [2.5, 1.0, 2.5]


def test_pearson_sees_the_size_of_differences_not_only_their_order():
    assert tool.pearson([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    assert tool.pearson([1, 2, 3], [1, 2, 30]) < tool.spearman([1, 2, 3], [1, 2, 30]) == pytest.approx(1.0)


def _result(a, b, score_a, score_b):
    return {"config_a": a, "config_b": b, "score_a": score_a, "score_b": score_b}


def test_goal_difference_is_per_opponent_from_either_side_of_the_match():
    summary = {
        "results": [
            _result("x", "o1", 3, 1),  # x played o1 as config_a: +2
            _result("o2", "x", 0, 1),  # and o2 as config_b: +1
            _result("x", "o3", 0, 5),  # an opponent left out
            _result("o1", "o2", 9, 0),  # not x's
        ]
    }
    assert tool.goal_difference(summary, "x", {"o1", "o2"}) == pytest.approx(1.5)


def test_a_candidate_s_effect_leaves_out_starts_played_against_it_or_the_baseline():
    base = list(all_hand_authored_scenarios())[0]

    def start(sid: str, opponent: str):
        scenario = dataclasses.replace(base.scenario, config_b_name=opponent)
        provenance = dataclasses.replace(base.provenance, situation="own kickoff")
        return dataclasses.replace(base, scenario_id=sid, scenario=scenario, provenance=provenance)

    starts = [start("a", "high_press"), start("b", "counter_flow"), start("c", "split_shape")]
    ok = {"outcome": 3, "error": None}
    results = {
        "build_counter_flow_kernel_strategy": {"a": ok, "b": ok, "c": ok},
        "build_split_shape_kernel_strategy": {"a": {"outcome": 0, "error": None}, "b": ok, "c": ok},
    }

    rows = tool.paired_rows(starts, results, "build_counter_flow_kernel_strategy", "build_split_shape_kernel_strategy")

    assert rows == [
        {
            "situation": "own kickoff",
            "candidate_outcomes": [3],
            "baseline_outcomes": [0],
            "candidate_errors": [],
            "baseline_errors": [],
        }
    ]
