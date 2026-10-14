from utama_core.replay.match_cache import MatchCache, differences, spot_check_sample

_RECORD = {
    "result": {"config_a": "a", "config_b": "b", "score_a": 1, "score_b": 0, "stats": {"shots": (3, 1)}},
    "restarts": [{"match": "a_vs_b", "kind": "kickoff"}],
    "losses": {"match": "a_vs_b", "turnovers": []},
}


def test_a_stored_record_reads_back_and_matches_the_one_written(tmp_path):
    cache = MatchCache(tmp_path)
    cache.put("ab12", _RECORD)

    assert differences(cache.get("ab12"), _RECORD) == []


def test_an_unknown_or_evicted_key_has_no_record(tmp_path):
    cache = MatchCache(tmp_path)
    assert cache.get("ab12") is None
    cache.put("ab12", _RECORD)
    cache.evict("ab12")
    assert cache.get("ab12") is None


def test_a_corrupt_record_is_a_miss_not_an_error(tmp_path):
    cache = MatchCache(tmp_path)
    cache.path("ab12").parent.mkdir(parents=True)
    cache.path("ab12").write_text("{not json")
    assert cache.get("ab12") is None


def test_differences_names_each_part_that_disagrees():
    fresh = {**_RECORD, "result": {**_RECORD["result"], "score_a": 2}, "losses": {"turnovers": [1]}}
    assert differences(_RECORD, fresh) == ["result", "losses"]


def test_spot_check_sample_takes_the_fraction_and_at_least_one():
    keys = [f"k{i}" for i in range(200)]
    assert len(spot_check_sample(keys, 0.05)) == 10
    assert len(spot_check_sample(keys[:3], 0.05)) == 1
    assert spot_check_sample(keys, 0.0) == set()
    assert spot_check_sample([], 0.05) == set()


def test_spot_check_sample_is_the_same_every_run_and_independent_of_order():
    keys = [f"k{i}" for i in range(50)]
    assert spot_check_sample(keys, 0.1) == spot_check_sample(list(reversed(keys)), 0.1)


def test_two_processes_storing_the_same_match_use_different_temp_files(tmp_path, monkeypatch):
    # Round-robins on different strategy branches share the cache. With one shared temp file,
    # one process could rename away the other's half-written file and fail the other's rename.
    import os
    from pathlib import Path

    temps = []
    real_replace = Path.replace

    def _recording_replace(self, target):
        temps.append(self.name)
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", _recording_replace)
    cache = MatchCache(tmp_path)
    for pid in (101, 202):
        monkeypatch.setattr(os, "getpid", lambda pid=pid: pid)
        cache.put("ab" * 32, {"result": {"score_a": 1}})
    assert len(set(temps)) == 2
    assert cache.get("ab" * 32) == {"result": {"score_a": 1}}
    assert not list(tmp_path.rglob("*.tmp"))
