"""Tests for `smoke_tournament.py`'s (formerly `tournament.py`) hand-rolled
`--fuzz-restarts`/`--fuzz-interval` CLI parsing (see `docs/STRATEGY_DEVELOPMENT.md`).
`smoke_tournament.py` has no `--help` and no `argparse` — flags are parsed by
scanning `sys.argv` directly inside `main()` — so these tests monkeypatch
`sys.argv` and stub out `run_match` (never actually running a match/simulator)
to check the flags are parsed and threaded through correctly.
"""

from __future__ import annotations

import json

import smoke_tournament as tournament


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
