"""Tests for `utama_core.replay.dynamic_screen`.

Runs real short headless matches (same constraint as `test_scenario_scorer.py`).
Doesn't try to force every verdict category — the champion-vs-itself pairing
this test uses is expected to land DETERMINED or DEAD, which is enough to
confirm the classification pipeline runs end to end and returns a valid
verdict; forcing NOISY/INFORMATIVE deterministically would require a policy
pair known to diverge, which isn't this module's concern to guarantee.
"""

from __future__ import annotations

from utama_core.replay.dynamic_screen import (
    DynamicScreenResult,
    ScreenVerdict,
    screen_scenario,
)
from utama_core.replay.hand_authored_scenarios import all_hand_authored_scenarios


def _kickoff_scenario():
    return next(bs for bs in all_hand_authored_scenarios() if bs.scenario_id == "kickoff_center_v1")


def test_screen_scenario_returns_valid_verdict():
    bench_scenario = _kickoff_scenario()
    result = screen_scenario(
        bench_scenario,
        champion_config="build_default_kernel_strategy",
        pool_configs=("build_default_kernel_strategy",),
        horizon_s=3.0,
    )

    assert isinstance(result, DynamicScreenResult)
    assert result.verdict in ScreenVerdict
    assert len(result.outcomes) == 2  # champion-vs-self + one pool member
    assert result.scenario_id == "kickoff_center_v1"


def test_screen_scenario_champion_vs_self_is_low_variance():
    """Champion vs itself, same scenario, deterministic sim: repeated runs
    should land on the identical outcome both times (rsim is deterministic
    given identical inputs — see tools/metric_correlation.py's note), so the
    stdev across the two champion-vs-self-equivalent runs here is 0."""
    bench_scenario = _kickoff_scenario()
    result = screen_scenario(
        bench_scenario,
        champion_config="build_default_kernel_strategy",
        pool_configs=("build_default_kernel_strategy",),
        horizon_s=3.0,
    )

    assert result.outcome_stdev == 0.0
    assert result.verdict in (ScreenVerdict.DEAD, ScreenVerdict.DETERMINED)
