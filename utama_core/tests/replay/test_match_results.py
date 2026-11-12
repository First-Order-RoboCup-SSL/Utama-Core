"""`replay/match_results.py`: the one store of match results, its fixtures and its records."""

from __future__ import annotations

import json

from utama_core.replay import match_results as mr
from utama_core.replay.match_results import MatchResults

_FROZEN_EVENT = {
    "kind": "COMMITTED_FROZEN",
    "sim_time": 12.5,
    "tick": 1250,
    "referee_command": "NORMAL_START",
    "duration_s": 9.0,
    "tactic_ids": ["t1"],
    "robot_ids": [1, 2],
    "diagnosis": "ball held by enemy 3",
}


class _Graph:
    """Stands in for CodeGraph: the store asks a graph only for the base fingerprint."""

    def __init__(self, base="base"):
        self.base = base

    def base_fingerprint(self, entries):
        return self.base


def _result(a="a", b="b", score_a=1, score_b=0, stats=None):
    return {"config_a": a, "config_b": b, "score_a": score_a, "score_b": score_b, "stats": stats}


def _rec(env="e", a_fp="fa", b_fp="fb", seed=None, a="a", b="b", stats=None, **kw):
    losses = {"match": "a_vs_b", "turnovers": []}
    return mr.record(
        env, a_fp, b_fp, _result(a, b, stats=stats), seed=seed, source="run", restarts=[], losses=losses, **kw
    )


def test_a_stored_record_reads_back_as_it_was_written(tmp_path):
    store = MatchResults(tmp_path)
    rec = _rec(stats={"shots": {"friendly": 3, "enemy": 1}, "stall_events": [], "possession_pct": (0.5, 0.5)})
    store.put(rec)

    assert store.get("e", rec["fixture"]) == rec
    assert rec["shots_a"] == 3 and rec["shots_b"] == 1
    assert rec["losses"] == {"match": "a_vs_b", "turnovers": []} and rec["restarts"] == []


def test_an_unknown_fixture_and_a_corrupt_file_are_misses_not_errors(tmp_path):
    store = MatchResults(tmp_path)
    assert store.get("e", "ab12") is None
    store.path("e", "ab12").parent.mkdir(parents=True)
    store.path("e", "ab12").write_text("{not json")
    assert store.get("e", "ab12") is None


def test_a_fixture_depends_on_environment_players_order_sides_and_seed():
    f = mr.fixture_id
    base = f("e", "1", "2", 1)
    assert f("e", "1", "2", 2) != base  # the noise seed
    assert f("e", "9", "2", 1) != base  # config_a's code
    assert f("e", "2", "1", 1) != base  # who is config_a
    assert f("other", "1", "2", 1) != base  # the environment
    assert f("e", "1", "2", 1, a_is_right=False) != base
    assert f("e", "1", "2", 1, a_kicks_off=False) != base
    assert f("e", "1", "2", 1) == base


def test_the_environment_covers_every_setting_but_the_noise_seed():
    g = _Graph()
    noise = {"seed": 1, "latency_s": [0.05, 0.1]}
    base = mr.environment(g, duration_seconds=1200.0, sim_noise=noise)
    assert mr.environment(g, duration_seconds=1200.0, sim_noise={**noise, "seed": 2}) == base
    assert mr.environment(g, duration_seconds=300.0, sim_noise=noise) != base
    assert mr.environment(g, duration_seconds=1200.0, sim_noise=None) != base
    assert mr.environment(g, duration_seconds=1200.0, sim_noise=noise, control_scheme="dwa") != base
    assert mr.environment(g, duration_seconds=1200.0, sim_noise=noise, fuzz_seed=1) != base
    assert mr.environment(g, duration_seconds=1200.0, sim_noise=noise, fuzz_seed=1, fuzz_interval_s=(1.0, 2.0)) != (
        mr.environment(g, duration_seconds=1200.0, sim_noise=noise, fuzz_seed=1)
    )
    # the fuzz interval only matters when fuzzing is on
    assert mr.environment(g, duration_seconds=1200.0, sim_noise=noise, fuzz_interval_s=(1.0, 2.0)) == base
    # shared code edits move the environment
    assert mr.environment(_Graph("edited"), duration_seconds=1200.0, sim_noise=noise) != base


def test_load_reads_only_the_environment_asked_for(tmp_path):
    store = MatchResults(tmp_path)
    mine = _rec(env="e", seed=1)
    store.put(mine)
    store.put(_rec(env="other", seed=1))

    assert list(store.load("e")) == [mine["fixture"]]
    assert store.load("nowhere") == {}


def test_load_names_a_player_by_its_fingerprint(tmp_path):
    """A frozen file and the same code in the repo are one player, whichever played the match."""
    store = MatchResults(tmp_path)
    store.put(_rec(a="split_shape", a_fp="1", b="other", b_fp="2"))

    loaded = store.load("e", fps={"frozen/split_shape@abc.strategy": "1"})
    (rec,) = loaded.values()
    assert (rec["a"], rec["b"]) == ("frozen/split_shape@abc.strategy", "other")


def test_a_torn_file_is_skipped_and_the_rest_still_load(tmp_path):
    store = MatchResults(tmp_path)
    good = _rec(seed=1)
    store.put(good)
    (tmp_path / "e" / "torn.json").write_text('{"env": "e", "fixt')

    assert list(store.load("e")) == [good["fixture"]]


def test_stall_events_are_stored_compactly_with_their_diagnosis():
    rec = _rec(stats={"shots": {}, "stall_events": [_FROZEN_EVENT]})
    assert rec["stalls"] == [{"kind": "COMMITTED_FROZEN", "sim_time": 12.5, "diagnosis": "ball held by enemy 3"}]


def test_no_stall_events_is_an_empty_list_and_unknown_is_none():
    assert _rec(stats={"stall_events": []})["stalls"] == []
    assert _rec(stats=None)["stalls"] is None  # no stats at all: unknown, not "no stalls"
    assert _rec(stats={"shots": {}})["stalls"] is None  # stats without the stall field: unknown


def test_stall_summary_drops_the_detail_a_record_does_not_need():
    rows = mr.stall_summary([{**_FROZEN_EVENT, "diagnosis": ""}])
    assert rows == [{"kind": "COMMITTED_FROZEN", "sim_time": 12.5}]
    assert json.loads(json.dumps(rows)) == rows


def test_two_processes_storing_the_same_match_use_different_temp_files(tmp_path, monkeypatch):
    # Round-robins on different strategy branches share the store. With one shared temp file,
    # one process could rename away the other's half-written file and fail the other's rename.
    import os
    from pathlib import Path

    temps = []
    real_replace = Path.replace

    def _recording_replace(self, target):
        temps.append(self.name)
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", _recording_replace)
    store = MatchResults(tmp_path)
    rec = _rec(seed=1)
    for pid in (101, 202):
        monkeypatch.setattr(os, "getpid", lambda pid=pid: pid)
        store.put(rec)
    assert len(set(temps)) == 2
    assert store.get("e", rec["fixture"]) == rec
    assert not list(tmp_path.rglob("*.tmp"))
