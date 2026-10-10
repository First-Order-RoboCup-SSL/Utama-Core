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
