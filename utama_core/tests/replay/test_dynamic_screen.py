"""Tests for `utama_core.replay.dynamic_screen`.

Runs real short headless matches (same constraint as `test_scenario_scorer.py`).
Doesn't try to force every verdict category; confirms the pipeline runs end to
end and returns a valid verdict.
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


def test_screen_scenario_drops_duplicate_opponents_and_repeats_per_seed():
    """The default pool used to be the champion itself, so the screen played one
    deterministic match twice: spread 0, every live scenario DETERMINED. Duplicates
    are dropped; each remaining opponent is played from `repeats` jittered starts."""
    result = screen_scenario(
        _kickoff_scenario(),
        champion_config="build_default_kernel_strategy",
        pool_configs=("build_default_kernel_strategy",),
        horizon_s=3.0,
        repeats=2,
    )

    assert isinstance(result, DynamicScreenResult)
    assert result.verdict in ScreenVerdict
    assert len(result.outcomes) == 2  # one opponent (the champion), two starts
    assert result.policy_stdev == 0.0  # a single opponent has no policy spread
    assert result.scenario_id == "kickoff_center_v1"
