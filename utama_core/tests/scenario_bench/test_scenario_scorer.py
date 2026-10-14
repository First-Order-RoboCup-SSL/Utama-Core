"""Tests for `utama_core.scenario_bench.scenario_scorer`.

Runs real headless rsim matches (short horizons) — this module can't be
tested with pure fixtures the way the harvester's transition-detection can,
since scoring means actually ticking a `StrategyRunner` forward. Kept fast
by using short horizons and the cheapest available kernel strategy.
"""

from __future__ import annotations

import dataclasses

from utama_core.engine.match_stats import MatchStats
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.scenario_bench.hand_authored_scenarios import (
    all_hand_authored_scenarios,
)
from utama_core.scenario_bench.scenario_scorer import (
    FLICKER_S,
    SIGNALS,
    ScenarioOutcome,
    _classify_outcome,
    _RealLossWatch,
    score_scenario,
    start_signals,
)


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
    assert set(result.signals) == set(SIGNALS)


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


def _watch(ticks) -> int:
    """ticks: (t, command, friendly turnovers so far, possession side)"""
    watch = _RealLossWatch()
    for t, cmd, turnovers, side in ticks:
        watch.step(t, cmd, turnovers, side)
    return watch.losses


_LIVE_CMD = RefereeCommand.NORMAL_START


def test_a_turnover_won_back_within_the_flicker_window_is_not_a_loss():
    """Raw `MatchStats.turnovers` counts two robots on one ball flipping "nearest"; the
    bench used to score every such flip as TURNOVER. Only an opponent holding the ball
    for more than `FLICKER_S` is a loss."""
    flicker = [
        (0.0, _LIVE_CMD, 0, "friendly"),
        (0.5, _LIVE_CMD, 1, "enemy"),
        (0.5 + FLICKER_S, _LIVE_CMD, 1, "friendly"),
    ]
    assert _watch(flicker) == 0
    kept = [(0.0, _LIVE_CMD, 0, "friendly"), (0.5, _LIVE_CMD, 1, "enemy"), (0.51 + FLICKER_S, _LIVE_CMD, 1, "enemy")]
    assert _watch(kept) == 1


def test_a_restart_to_the_opponent_is_a_loss_only_if_we_had_the_ball():
    blue_free_kick = RefereeCommand.DIRECT_FREE_BLUE
    assert _watch([(0.0, _LIVE_CMD, 0, "friendly"), (0.1, blue_free_kick, 0, "friendly")]) == 1
    assert _watch([(0.0, _LIVE_CMD, 0, "enemy"), (0.1, blue_free_kick, 0, "enemy")]) == 0
    # a turnover while play is stopped is a restart handover, not a loss
    stop = RefereeCommand.STOP
    assert _watch([(0.0, stop, 0, "friendly"), (0.5, stop, 1, "enemy"), (2.0, stop, 1, "enemy")]) == 0


def test_raw_turnovers_without_a_real_loss_do_not_score_turnover():
    before = MatchStats({}, {}, {})
    after = dataclasses.replace(before, turnovers=4, attacking_third_entries=1)
    assert _classify_outcome(before, after, False, False, real_losses=0) == ScenarioOutcome.ENTRY_RETAINED
    assert _classify_outcome(before, after, False, False, real_losses=1) == ScenarioOutcome.TURNOVER


def test_bench_plays_with_the_round_robins_motion_planner(monkeypatch):
    """Banks are harvested from round-robins, which run `match.run_match`'s
    control scheme (fpp). The scorer played every start with trajsample instead, so a
    bench A/B measured play under another planner, and a planner change made in fpp
    left every outcome unchanged."""
    import inspect

    from tools.evaluation import match
    from utama_core.scenario_bench import scenario_scorer

    seen = {}
    monkeypatch.setattr(scenario_scorer, "StrategyRunner", lambda **kwargs: seen.update(kwargs))
    scenario_scorer._build_runner("press_and_pass", "low_block", stats_path="unused")

    assert seen["control_scheme"] == inspect.signature(match.run_match).parameters["control_scheme"].default


def test_start_signals_are_the_candidate_s_side_only():
    shot = {"side": "friendly", "open_goal": 0.5, "shot_after_s": None}
    chances = {
        "shots": [shot, {**shot, "open_goal": 0.25}, {**shot, "side": "enemy", "open_goal": 0.75}],
        "regains": [
            {"side": "friendly", "shot_after_s": 2.0},
            {"side": "friendly", "shot_after_s": None},
            {"side": "enemy", "shot_after_s": 1.0},
        ],
        "danger": {"friendly": {"s": 4.5, "spells": 1}, "enemy": {"s": 9.0, "spells": 2}},
    }

    assert start_signals(chances, real_losses=2, entries=1) == {
        "shots": 2,
        "open_shots": 0.75,
        "regains_to_shot": 1,
        "entries": 1,
        "real_losses": 2,
        "shots_faced": 1,
        "open_shots_faced": 0.75,
        "danger_s": 4.5,
    }
