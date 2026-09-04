"""Tests for `utama_core.replay.scenario_scorer`.

Runs real headless rsim matches (short horizons) — this module can't be
tested with pure fixtures the way the harvester's transition-detection can,
since scoring means actually ticking a `StrategyRunner` forward. Kept fast
by using short horizons and the cheapest available kernel strategy.
"""

from __future__ import annotations

from utama_core.replay.hand_authored_scenarios import all_hand_authored_scenarios
from utama_core.replay.scenario_scorer import ScenarioOutcome, score_scenario


def _kickoff_scenario():
    return next(bs for bs in all_hand_authored_scenarios() if bs.scenario_id == "kickoff_center_v1")


def test_score_scenario_returns_a_result_with_stats():
    bench_scenario = _kickoff_scenario()
    result = score_scenario(
        bench_scenario,
        candidate_config="build_default_kernel_strategy",
        opponent_config="build_default_kernel_strategy",
        horizon_s=3.0,
        stats_path="/tmp/test_score_scenario_stats.json",
    )

    assert result.error is None
    assert result.scenario_id == "kickoff_center_v1"
    assert isinstance(result.outcome, ScenarioOutcome)
    assert result.ticks_run > 0


def test_score_scenario_reports_error_on_bad_config():
    bench_scenario = _kickoff_scenario()
    result = score_scenario(
        bench_scenario,
        candidate_config="not_a_real_strategy",
        opponent_config="build_default_kernel_strategy",
        horizon_s=3.0,
        stats_path="/tmp/test_score_scenario_bad_config.json",
    )

    assert result.error is not None
    assert result.ticks_run == 0
