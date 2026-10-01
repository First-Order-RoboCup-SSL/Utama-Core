"""`smoke_tournament.py --reuse`: a match whose key is stored is not played again.

Matches, keys and the replay analyses are stubbed: `fingerprint.match_key` is replaced by
a key built from a per-config version number, so "editing" a config is bumping its
version. `tests/replay/test_fingerprint.py` covers the real keys and
`tests/replay/test_match_cache.py` the store.
"""

from __future__ import annotations

import json

import pytest

import smoke_tournament as tournament
from utama_core.replay.match_cache import MatchCache

_CONFIGS = ["tiki_taka", "low_block", "high_press"]


class _Always5(dict):
    def get(self, key, default=None):
        return 5


_EVERY_PAIR_SCORES_5 = _Always5()


@pytest.fixture
def rig(tmp_path, monkeypatch):
    rig = type("Rig", (), {})()
    rig.played = []
    rig.version = {f"build_{c}_kernel_strategy": 0 for c in _CONFIGS}
    rig.score = {}  # pair -> score_a, to make a replay disagree with its stored record
    rig.cache = MatchCache(tmp_path / "cache")
    rig.on_play = None

    def fake_run_match(a, b, run_dir=None, *_args, **_kwargs):
        rig.played.append((a, b))
        (run_dir / f"{tournament._tag((a, b))}.npz").write_bytes(b"")
        if rig.on_play is not None:
            rig.on_play()
        return tournament.MatchResult(config_a=a, config_b=b, score_a=rig.score.get((a, b), 1), score_b=0, stats={})

    def fake_losses(run_dir, workers=8):
        return [
            {"match": p.name[: -len(".npz")], "turnovers": [], "restarts": []} for p in sorted(run_dir.glob("*.npz"))
        ]

    monkeypatch.setattr(tournament, "run_match", fake_run_match)
    monkeypatch.setattr(tournament, "REPLAY_BASE_PATH", tmp_path)
    monkeypatch.setattr(tournament, "CodeGraph", lambda: None)
    monkeypatch.setattr(tournament, "match_key", lambda _g, a, b, **_kw: f"{a}{rig.version[a]}{b}{rig.version[b]}")
    monkeypatch.setattr(tournament.match_cache, "MatchCache", lambda: rig.cache)
    monkeypatch.setattr(tournament.restart_outcomes, "analyse_match", lambda path: [])
    monkeypatch.setattr(tournament.turnover_breakdown, "analyse_run", fake_losses)
    monkeypatch.setattr(tournament.turnover_breakdown, "breakdown", lambda losses: {})
    monkeypatch.setattr(tournament.turnover_breakdown, "report", lambda *a: "")
    monkeypatch.setattr(tournament.turnover_breakdown, "real_loss_kinds", lambda r: {})
    monkeypatch.setattr(tournament, "_print_ball_losses", lambda *a: None)

    def run(*flags):
        rig.played.clear()
        monkeypatch.setattr("sys.argv", ["tournament.py", "--sequential", "--reuse", *flags, *_CONFIGS])
        tournament.main()
        (summary_path,) = sorted(tmp_path.glob("tournament_*/summary.json"))[-1:]
        return json.loads(summary_path.read_text())

    rig.run = run
    return rig


def test_a_second_run_reuses_every_stored_match_and_reports_the_same_results(rig):
    first = rig.run("--spot-check", "0")
    second = rig.run("--spot-check", "0")

    assert rig.played == []
    assert second["reuse"] == {"reused": 3, "spot_checked": 0, "played": 0}
    assert all(r["reused"] for r in second["results"])
    key = lambda r: (r["config_a"], r["config_b"])  # noqa: E731
    strip = lambda rs: sorted(({k: v for k, v in r.items() if k != "reused"} for r in rs), key=key)  # noqa: E731
    assert strip(second["results"]) == strip(first["results"])
    assert second["standings"] == first["standings"]


def test_after_one_config_changes_only_its_matches_play(rig):
    rig.run("--spot-check", "0")
    rig.version["build_low_block_kernel_strategy"] += 1

    rig.run("--spot-check", "0")

    assert sorted(rig.played) == [
        ("build_high_press_kernel_strategy", "build_low_block_kernel_strategy"),
        ("build_low_block_kernel_strategy", "build_tiki_taka_kernel_strategy"),
    ]


def test_a_spot_check_mismatch_evicts_the_run_s_records_and_fails_strict(rig):
    rig.run("--spot-check", "0")
    rig.score = _EVERY_PAIR_SCORES_5  # whichever match is spot-checked now disagrees

    with pytest.raises(SystemExit, match="differ from their stored records"):
        rig.run("--spot-check", "0.34", "--strict")

    assert len(rig.played) == 1  # one of three replayed as the spot-check
    assert list(rig.cache.root.rglob("*.json")) == []  # the mismatch and the two it reused


def test_a_run_whose_code_changes_while_it_plays_stores_nothing(rig):
    def edit():
        rig.version["build_tiki_taka_kernel_strategy"] = 9

    rig.on_play = edit

    rig.run("--spot-check", "0")

    assert list(rig.cache.root.rglob("*.json")) == []


def test_reuse_with_no_save_is_refused(monkeypatch):
    monkeypatch.setattr("sys.argv", ["tournament.py", "--reuse", "--no-save", *_CONFIGS])
    with pytest.raises(SystemExit, match="--reuse"):
        tournament.main()
