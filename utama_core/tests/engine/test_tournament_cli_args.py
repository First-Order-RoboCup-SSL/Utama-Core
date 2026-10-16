"""Tests for `round_robin.py`'s hand-rolled
`--fuzz-restarts`/`--fuzz-interval` CLI parsing (see `docs/STRATEGY_DEVELOPMENT.md`).
`round_robin.py` has no `argparse` — flags are parsed by
scanning `sys.argv` directly inside `main()` — so these tests monkeypatch
`sys.argv` and stub out `run_match` (never actually running a match/simulator)
to check the flags are parsed and threaded through correctly.
"""

from __future__ import annotations

import json

from tools.evaluation import round_robin as tournament


def _stub_result(config_a_name, config_b_name, **_kwargs):
    return tournament.MatchResult(config_a=config_a_name, config_b=config_b_name, score_a=1, score_b=0, stats=None)


def test_fuzz_restarts_and_interval_parsed_and_passed_to_run_match(tmp_path, monkeypatch):
    calls = []

    def _fake_run_match(
        config_a_name, config_b_name, run_dir=None, control_scheme="fpp", fuzz_seed=None, fuzz_interval_s=(25.0, 45.0)
    ):
        calls.append(
            {
                "config_a": config_a_name,
                "config_b": config_b_name,
                "fuzz_seed": fuzz_seed,
                "fuzz_interval_s": fuzz_interval_s,
            }
        )
        return _stub_result(config_a_name, config_b_name)

    monkeypatch.setattr(tournament, "run_match", _fake_run_match)
    monkeypatch.setattr(tournament, "REPLAY_BASE_PATH", tmp_path)
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
            "build_low_block_kernel_strategy",
            "build_tiki_taka_kernel_strategy",
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

    def _fake_run_match(
        config_a_name, config_b_name, run_dir=None, control_scheme="fpp", fuzz_seed=None, fuzz_interval_s=(25.0, 45.0)
    ):
        calls.append({"fuzz_seed": fuzz_seed, "fuzz_interval_s": fuzz_interval_s})
        return _stub_result(config_a_name, config_b_name)

    monkeypatch.setattr(tournament, "run_match", _fake_run_match)
    monkeypatch.setattr(tournament, "REPLAY_BASE_PATH", tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "tournament.py",
            "--sequential",
            "build_low_block_kernel_strategy",
            "build_tiki_taka_kernel_strategy",
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


def test_pair_plays_exactly_that_one_pairing_in_the_given_order(tmp_path, monkeypatch):
    """`--pair a b` replays one fixture (e.g. to diagnose a stall) with a as config_a,
    not the sorted order a round-robin over those two configs would use."""
    calls = []

    def _fake_run_match(config_a_name, config_b_name, *_args, **_kwargs):
        calls.append((config_a_name, config_b_name))
        return _stub_result(config_a_name, config_b_name)

    monkeypatch.setattr(tournament, "run_match", _fake_run_match)
    monkeypatch.setattr(tournament, "REPLAY_BASE_PATH", tmp_path)
    monkeypatch.setattr("sys.argv", ["tournament.py", "--sequential", "--no-save", "--pair", "tiki_taka", "low_block"])

    tournament.main()

    assert calls == [("build_tiki_taka_kernel_strategy", "build_low_block_kernel_strategy")]


def test_help_prints_usage_and_plays_nothing(monkeypatch, capsys):
    # `--help` used to be taken for a config name, so it failed instead of listing the flags.
    def _no_match(*_args, **_kwargs):
        raise AssertionError("--help must not play a match")

    monkeypatch.setattr(tournament, "run_match", _no_match)
    for flag in ("--help", "-h"):
        monkeypatch.setattr("sys.argv", ["round_robin.py", flag])
        tournament.main()
        out = capsys.readouterr().out
        assert "--pair A B" in out and "--reuse" in out


def test_no_configs_plays_every_config_but_the_retired_and_a_named_retired_one_still_plays(tmp_path, monkeypatch):
    calls = []

    def _fake_run_match(config_a_name, config_b_name, *_args, **_kwargs):
        calls.append((config_a_name, config_b_name))
        return _stub_result(config_a_name, config_b_name)

    names = ["build_a_kernel_strategy", "build_b_kernel_strategy", "build_c_kernel_strategy"]
    monkeypatch.setattr(tournament, "run_match", _fake_run_match)
    monkeypatch.setattr(tournament, "REPLAY_BASE_PATH", tmp_path)
    monkeypatch.setattr(tournament, "_CONFIG_NAMES", names)
    monkeypatch.setattr(tournament, "RETIRED", {"build_c_kernel_strategy"})

    monkeypatch.setattr("sys.argv", ["round_robin.py", "--sequential", "--no-save"])
    tournament.main()
    assert calls == [("build_a_kernel_strategy", "build_b_kernel_strategy")]

    calls.clear()
    monkeypatch.setattr("sys.argv", ["round_robin.py", "--sequential", "--no-save", "a", "c"])
    tournament.main()
    assert calls == [("build_a_kernel_strategy", "build_c_kernel_strategy")]


def test_sim_noise_parsed_and_passed_to_run_match_and_summary(tmp_path, monkeypatch):
    from utama_core.rsoccer_simulator.src.Utils.sim_noise import SimNoise

    calls = []

    def _fake_run_match(config_a_name, config_b_name, **kwargs):
        calls.append(kwargs.get("sim_noise"))
        return _stub_result(config_a_name, config_b_name)

    monkeypatch.setattr(tournament, "run_match", _fake_run_match)
    monkeypatch.setattr(tournament, "REPLAY_BASE_PATH", tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "tournament.py",
            "--sequential",
            "--sim-noise",
            '{"seed": 3, "latency_s": [0.05, 0.1], "kick_speed_sd": 0.2}',
            "build_low_block_kernel_strategy",
            "build_tiki_taka_kernel_strategy",
        ],
    )

    tournament.main()

    assert calls == [SimNoise(seed=3, latency_s=(0.05, 0.1), kick_speed_sd=0.2)]
    (summary_path,) = tmp_path.glob("tournament_*/summary.json")
    assert json.loads(summary_path.read_text())["sim_noise"]["seed"] == 3


def test_match_seconds_keys_the_match_cache_and_is_recorded_in_the_summary(tmp_path, monkeypatch):
    # A shortened run's matches must not be cached under the full-length key, or a later
    # full-length `--reuse` run would serve them as full matches.
    durations = []

    def _fake_match_key(_graph, _a, _b, *, duration_seconds, **_kwargs):
        durations.append(duration_seconds)
        return f"key-{duration_seconds}"

    class _EmptyCache:
        def get(self, _key):
            return None

        def put(self, _key, _record):
            pass

    monkeypatch.setattr(tournament, "run_match", _stub_result)
    monkeypatch.setattr(tournament, "match_key", _fake_match_key)
    monkeypatch.setattr(tournament, "CodeGraph", lambda: None)
    monkeypatch.setattr(tournament.match_cache, "MatchCache", _EmptyCache)
    monkeypatch.setattr(tournament, "REPLAY_BASE_PATH", tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "tournament.py",
            "--sequential",
            "--reuse",
            "--match-seconds",
            "300",
            "build_low_block_kernel_strategy",
            "build_tiki_taka_kernel_strategy",
        ],
    )

    tournament.main()

    assert durations and set(durations) == {300.0}
    (summary_path,) = tmp_path.glob("tournament_*/summary.json")
    assert json.loads(summary_path.read_text())["max_match_seconds"] == 300.0
