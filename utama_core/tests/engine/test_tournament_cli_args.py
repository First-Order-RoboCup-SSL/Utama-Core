"""Tests for `round_robin.py`'s hand-rolled
`--fuzz-restarts`/`--fuzz-interval` CLI parsing (see `docs/STRATEGY_DEVELOPMENT.md`).
`round_robin.py` has no `argparse` — flags are parsed by
scanning `sys.argv` directly inside `main()` — so these tests monkeypatch
`sys.argv` and stub out `run_match` (never actually running a match/simulator)
to check the flags are parsed and threaded through correctly.
"""

from __future__ import annotations

import json

import pytest

from tools.evaluation import round_robin as tournament


class _Graph:
    """Stands in for CodeGraph: the store needs only the fingerprints' names."""

    def base_fingerprint(self, entries):
        return "base"

    def strategy_fingerprint(self, name):
        return f"fp-{name}"


@pytest.fixture(autouse=True)
def _no_real_fingerprints(monkeypatch, tmp_path):
    """Every test here runs with a fake CodeGraph and its own replay folder, so the match results
    store is written under tmp_path."""
    monkeypatch.setattr(tournament, "CodeGraph", _Graph)
    monkeypatch.setattr(tournament, "REPLAY_BASE_PATH", tmp_path)


def _stub_result(config_a_name, config_b_name, **_kwargs):
    return tournament.MatchResult(config_a=config_a_name, config_b=config_b_name, score_a=1, score_b=0, stats=None)


def test_fuzz_restarts_and_interval_parsed_and_passed_to_run_match(tmp_path, monkeypatch):
    calls = []

    def _fake_run_match(config_a_name, config_b_name, **kwargs):
        calls.append(
            {
                "config_a": config_a_name,
                "config_b": config_b_name,
                "fuzz_seed": kwargs.get("fuzz_seed"),
                "fuzz_interval_s": kwargs.get("fuzz_interval_s"),
            }
        )
        return _stub_result(config_a_name, config_b_name)

    monkeypatch.setattr(tournament, "run_match", _fake_run_match)
    monkeypatch.setattr(
        "sys.argv",
        [
            "tournament.py",
            "--sequential",
            "--fuzz-restarts",
            "7",
            "--fuzz-interval",
            "5",
            "10",
            "low_block",
            "tiki_taka",
        ],
    )

    tournament.main()

    assert len(calls) == 1
    assert calls[0]["fuzz_seed"] == 7
    assert calls[0]["fuzz_interval_s"] == (5.0, 10.0)

    # summary.json must record the seed/interval for reproducibility.
    run_dirs = list(tmp_path.glob("tournament_*"))
    assert len(run_dirs) == 1
    summary = json.loads((run_dirs[0] / "summary.json").read_text())
    assert summary["fuzz_seed"] == 7
    assert summary["fuzz_interval_s"] == [5.0, 10.0]


def test_fuzz_restarts_off_by_default(tmp_path, monkeypatch):
    calls = []

    def _fake_run_match(config_a_name, config_b_name, **kwargs):
        calls.append({"fuzz_seed": kwargs.get("fuzz_seed"), "fuzz_interval_s": kwargs.get("fuzz_interval_s")})
        return _stub_result(config_a_name, config_b_name)

    monkeypatch.setattr(tournament, "run_match", _fake_run_match)
    monkeypatch.setattr(
        "sys.argv",
        [
            "tournament.py",
            "--sequential",
            "low_block",
            "tiki_taka",
        ],
    )

    tournament.main()

    assert len(calls) == 1
    assert calls[0]["fuzz_seed"] is None
    # Default interval is still threaded through even when fuzzing is off.
    assert calls[0]["fuzz_interval_s"] == (25.0, 45.0)

    run_dirs = list(tmp_path.glob("tournament_*"))
    summary = json.loads((run_dirs[0] / "summary.json").read_text())
    assert summary["fuzz_seed"] is None
    assert summary["fuzz_interval_s"] is None
    # The run's matches go to the match results store under REPLAY_BASE_PATH, so a test run never
    # writes into the real one.
    (path,) = (tmp_path / "match_results").rglob("*.json")
    assert json.loads(path.read_text())["source"] == run_dirs[0].name


def test_pair_plays_exactly_that_one_pairing_in_the_given_order(tmp_path, monkeypatch):
    """`--pair a b` replays one fixture (e.g. to diagnose a stall) with a as config_a,
    not the sorted order a round-robin over those two configs would use."""
    calls = []

    def _fake_run_match(config_a_name, config_b_name, **_kwargs):
        calls.append((config_a_name, config_b_name))
        return _stub_result(config_a_name, config_b_name)

    monkeypatch.setattr(tournament, "run_match", _fake_run_match)
    monkeypatch.setattr("sys.argv", ["tournament.py", "--sequential", "--no-save", "--pair", "tiki_taka", "low_block"])

    tournament.main()

    assert calls == [("tiki_taka", "low_block")]


def test_help_prints_usage_and_plays_nothing(monkeypatch, capsys):
    # `--help` used to be taken for a config name, so it failed instead of listing the flags.
    def _no_match(*_args, **_kwargs):
        raise AssertionError("--help must not play a match")

    monkeypatch.setattr(tournament, "run_match", _no_match)
    for flag in ("--help", "-h"):
        monkeypatch.setattr("sys.argv", ["round_robin.py", flag])
        tournament.main()
        out = capsys.readouterr().out
        assert "--pair A B" in out and "replays/match_results/" in out


def test_no_configs_plays_every_config_but_the_retired_and_a_named_retired_one_still_plays(tmp_path, monkeypatch):
    calls = []

    def _fake_run_match(config_a_name, config_b_name, *_args, **_kwargs):
        calls.append((config_a_name, config_b_name))
        return _stub_result(config_a_name, config_b_name)

    names = ["a", "b", "c"]
    monkeypatch.setattr(tournament, "run_match", _fake_run_match)
    monkeypatch.setattr(tournament, "_CONFIG_NAMES", names)
    monkeypatch.setattr(tournament, "RETIRED", {"c"})

    monkeypatch.setattr("sys.argv", ["round_robin.py", "--sequential", "--no-save"])
    tournament.main()
    assert calls == [("a", "b")]

    calls.clear()
    monkeypatch.setattr("sys.argv", ["round_robin.py", "--sequential", "--no-save", "a", "c"])
    tournament.main()
    assert calls == [("a", "c")]


def test_sim_noise_parsed_and_passed_to_run_match_and_summary(tmp_path, monkeypatch):
    from utama_core.rsoccer_simulator.src.Utils.sim_noise import SimNoise

    calls = []

    def _fake_run_match(config_a_name, config_b_name, **kwargs):
        calls.append(kwargs.get("sim_noise"))
        return _stub_result(config_a_name, config_b_name)

    monkeypatch.setattr(tournament, "run_match", _fake_run_match)
    monkeypatch.setattr(
        "sys.argv",
        [
            "tournament.py",
            "--sequential",
            "--sim-noise",
            '{"seed": 3, "latency_s": [0.05, 0.1], "kick_speed_sd": 0.2}',
            "low_block",
            "tiki_taka",
        ],
    )

    tournament.main()

    assert calls == [SimNoise(seed=3, latency_s=(0.05, 0.1), kick_speed_sd=0.2)]
    (summary_path,) = tmp_path.glob("tournament_*/summary.json")
    assert json.loads(summary_path.read_text())["sim_noise"]["seed"] == 3


def test_match_seconds_is_part_of_the_environment_so_each_length_has_its_own_results(tmp_path, monkeypatch):
    # A shortened run's matches must not be taken for full-length ones: each match length is its own
    # environment in the store, and the length is recorded in the summary.
    monkeypatch.setattr(tournament, "run_match", _stub_result)
    monkeypatch.setattr(
        "sys.argv", ["tournament.py", "--sequential", "--match-seconds", "300", "low_block", "tiki_taka"]
    )
    tournament.main()
    (summary_path,) = tmp_path.glob("tournament_*/summary.json")
    assert json.loads(summary_path.read_text())["max_match_seconds"] == 300.0

    monkeypatch.setattr("sys.argv", ["tournament.py", "--sequential", "low_block", "tiki_taka"])
    tournament.main()

    envs = sorted((tmp_path / "match_results").iterdir())
    assert len(envs) == 2
    assert all(len(list(env.glob("*.json"))) == 1 for env in envs)
