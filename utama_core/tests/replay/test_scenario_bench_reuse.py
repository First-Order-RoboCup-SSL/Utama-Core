"""`tools/scenario_bench.py --reuse`: a start whose key is stored is not played again.

Starts and keys are stubbed: `fingerprint.bench_key` is replaced by a key built from the
start and a per-config version number, so "editing" a config is bumping its version.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from utama_core.replay.hand_authored_scenarios import all_hand_authored_scenarios
from utama_core.replay.match_cache import MatchCache

_TOOL = Path(__file__).resolve().parents[3] / "tools" / "scenario_bench.py"
_spec = importlib.util.spec_from_file_location("scenario_bench", _TOOL)
scenario_bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scenario_bench)

_SCENARIOS = list(all_hand_authored_scenarios())[:3]


@pytest.fixture
def rig(tmp_path, monkeypatch):
    rig = type("Rig", (), {})()
    rig.played = []
    rig.version = {"build_cand_kernel_strategy": 0, "build_base_kernel_strategy": 0, "build_opp_kernel_strategy": 0}
    rig.outcome = 1
    rig.cache = MatchCache(tmp_path)

    def fake_start_result(bench_scenario, config, opponent, horizon_s):
        rig.played.append((bench_scenario.scenario_id, config))
        return {"outcome": rig.outcome, "foul": False, "stalled": False, "error": None}

    def fake_key(_graph, start, config, opponent, *, horizon_s):
        return f"{start['scenario_id']}-{config}{rig.version[config]}-{opponent}-{horizon_s}"

    monkeypatch.setattr(scenario_bench, "_start_result", fake_start_result)
    monkeypatch.setattr(scenario_bench, "bench_key", fake_key)
    monkeypatch.setattr(scenario_bench, "CodeGraph", lambda: None)
    monkeypatch.setattr(scenario_bench, "_resolve_config_name", lambda name: f"build_{name}_kernel_strategy")
    monkeypatch.setattr(scenario_bench.match_cache, "MatchCache", lambda: rig.cache)

    def run(spot_check=0.0):
        rig.played.clear()
        reuse = scenario_bench._Reuse(_SCENARIOS, ["cand", "base"], "opp", 20.0, 1, spot_check)
        rows, _ = scenario_bench._score(
            _SCENARIOS, candidate="cand", opponent="opp", horizon_s=20.0, repeats=1, baseline="base", reuse=reuse
        )
        return rows, reuse.finish()

    rig.run = run
    return rig


def test_a_second_run_plays_nothing_and_scores_the_same(rig):
    first, _ = rig.run()
    second, report = rig.run()

    assert rig.played == []
    assert report == {"reused": 6, "played": 0, "spot_checked": 0, "mismatches": 0}
    assert second == first


def test_after_the_candidate_changes_only_the_candidate_plays(rig):
    rig.run()
    rig.version["build_cand_kernel_strategy"] += 1

    _, report = rig.run()

    assert sorted(config for _sid, config in rig.played) == ["cand"] * 3
    assert report["reused"] == 3


def test_a_spot_check_mismatch_evicts_the_records_the_run_used(rig):
    rig.run()
    rig.outcome = -1  # every replay now disagrees with its record

    _, report = rig.run(spot_check=0.2)

    assert report["mismatches"] == 1
    assert list(rig.cache.root.rglob("*.json")) == []
