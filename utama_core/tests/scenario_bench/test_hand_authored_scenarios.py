"""Tests for `utama_core.scenario_bench.hand_authored_scenarios` (the ~20 fixed
bench anchors from roadmap item 14's bank v1 shape).

Every anchor must pass the static screen — these are meant to be the most
trustworthy scenarios in the bank, so a hand-authored scenario failing the
same screen a harvested one would fail is a bug in the authoring, not a
false positive to special-case around.
"""

from __future__ import annotations

from utama_core.config.field_params import STANDARD_FIELD_DIMS
from utama_core.scenario_bench.hand_authored_scenarios import (
    all_hand_authored_scenarios,
)
from utama_core.scenario_bench.start import (
    ScenarioLifecycle,
    ScenarioTrigger,
    start_tags,
    static_screen,
)
from utama_core.tests.replay.test_scenario import (
    _ApplyScenarioTestManager,
    _assert_close_to_scenario,
    _build_runner_and_scenario,
)


def test_all_hand_authored_scenarios_pass_static_screen():
    for bench_scenario in all_hand_authored_scenarios():
        result = static_screen(bench_scenario.scenario)
        assert result.ok, f"{bench_scenario.scenario_id} failed static screen: {result.violations}"


def test_all_hand_authored_scenarios_have_unique_ids():
    ids = [bs.scenario_id for bs in all_hand_authored_scenarios()]
    assert len(ids) == len(set(ids))


def test_all_hand_authored_scenarios_are_tagged_hand_authored():
    for bench_scenario in all_hand_authored_scenarios():
        assert bench_scenario.provenance.trigger is ScenarioTrigger.HAND_AUTHORED
        assert bench_scenario.provenance.anchor_tick is None
        assert bench_scenario.lifecycle is ScenarioLifecycle.CANDIDATE


def test_kickoff_anchor_applies_to_a_live_runner():
    """Proves a hand-authored `BenchScenario` flows through the same
    `apply_scenario` path a harvested one would — the schema isn't just
    self-consistent on paper, it round-trips through a real headless rsim
    runner (same pattern `test_scenario.py` uses for replay-sourced
    scenarios, minus the replay file dependency)."""
    bench_scenario = next(bs for bs in all_hand_authored_scenarios() if bs.scenario_id == "kickoff_center_v1")
    scenario = bench_scenario.to_scenario()
    runner = _build_runner_and_scenario(scenario)
    manager = _ApplyScenarioTestManager(scenario)
    manager._runner = runner  # noqa: SLF001 - test-only wiring, see eval_status

    passed = runner.run_test(test_manager=manager, episode_timeout=10.0, rsim_headless=True)

    assert manager.error is None, f"apply_scenario raised: {manager.error}"
    assert passed, "test episode did not complete"
    assert manager.applied

    _assert_close_to_scenario(manager.post_apply_positions, scenario, tol=0.15)


def test_every_anchor_is_played_from_the_candidate_s_side():
    """The candidate is friendly and defends the right goal (scorer: my_team_is_right=True).
    The anchors were once returned in the frame they are written in, friendly on the left:
    the candidate's keeper started in the goal it attacks, and the free kick "near our box"
    was taken in front of the opponent's."""
    anchors = {bs.scenario_id: bs for bs in all_hand_authored_scenarios()}
    for bs in anchors.values():
        keeper = next(r for r in bs.scenario.friendly_robots if r.id == 0)
        assert keeper.x > STANDARD_FIELD_DIMS.full_field_half_length - 0.5, bs.scenario_id

    assert start_tags(anchors["direct_free_defending_near_box_v1"])["third"] == "defensive"
    assert start_tags(anchors["direct_free_attacking_near_box_v1"])["third"] == "attacking"
    assert start_tags(anchors["open_play_3v2_counter_v1"])["third"] == "attacking"
