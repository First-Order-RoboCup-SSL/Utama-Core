"""Round-robin reuse: a match the match results store holds is taken from it, not played.

Matches, fingerprints and the replay analyses are stubbed: a config's fingerprint is its name
plus a version number, so "editing" a config is bumping its version. `tests/replay/test_fingerprint.py`
covers the real fingerprints and `tests/replay/test_match_results.py` the store.
"""

from __future__ import annotations

import json

import pytest

from tools.evaluation import round_robin as tournament
from utama_core.replay import match_results

_CONFIGS = ["tiki_taka", "low_block", "high_press"]


class _Graph:
    """Stands in for CodeGraph: fingerprints follow the rig's versions."""

    def __init__(self, rig):
        self.rig = rig

    def base_fingerprint(self, entries):
        return "base"

    def strategy_fingerprint(self, name):
        return f"fp-{name}-{self.rig.version[name]}"


@pytest.fixture
def rig(tmp_path, monkeypatch):
    rig = type("Rig", (), {})()
    rig.played = []
    rig.version = {c: 0 for c in _CONFIGS}
    rig.score = {}  # pair -> score_a, to make a replay disagree with its stored record
    rig.on_play = None

    def fake_run_match(a, b, run_dir=None, match_tag_suffix="", **_kwargs):
        rig.played.append((a, b))
        if run_dir is not None:
            (run_dir / f"{a}_vs_{b}{match_tag_suffix}.npz").write_bytes(b"")
        if rig.on_play is not None:
            rig.on_play()
        return tournament.MatchResult(config_a=a, config_b=b, score_a=rig.score.get((a, b), 1), score_b=0, stats={})

    def fake_analyse(npz):
        return [], {"match": npz.name[: -len(".npz")], "turnovers": []}

    monkeypatch.setattr(tournament, "run_match", fake_run_match)
    monkeypatch.setattr(tournament, "REPLAY_BASE_PATH", tmp_path)
    monkeypatch.setattr(tournament, "CodeGraph", lambda: _Graph(rig))
    monkeypatch.setattr(tournament, "analyse_replay", fake_analyse)
    monkeypatch.setattr(tournament, "_print_ball_losses", lambda *a: None)
    monkeypatch.setattr(tournament.turnover_breakdown, "breakdown", lambda losses: {})
    monkeypatch.setattr(tournament.turnover_breakdown, "report", lambda *a: "")
    monkeypatch.setattr(tournament.turnover_breakdown, "real_loss_kinds", lambda r: {})

    def run(*flags):
        rig.played.clear()
        monkeypatch.setattr("sys.argv", ["round_robin.py", "--sequential", *flags, *_CONFIGS])
        tournament.main()
        (summary_path,) = sorted(tmp_path.glob("tournament_*/summary.json"))[-1:]
        return json.loads(summary_path.read_text())

    rig.run = run
    rig.store = match_results.open_store(tmp_path)
    return rig


def test_a_second_run_takes_every_stored_match_and_reports_the_same_results(rig):
    first = rig.run()
    second = rig.run()

    assert rig.played == []
    assert all(r["reused"] for r in second["results"])
    assert not any(r["reused"] for r in first["results"])
    key = lambda r: (r["config_a"], r["config_b"])  # noqa: E731
    strip = lambda rs: sorted(({k: v for k, v in r.items() if k != "reused"} for r in rs), key=key)  # noqa: E731
    assert strip(second["results"]) == strip(first["results"])
    assert second["standings"] == first["standings"]


def test_after_one_config_changes_only_its_matches_play(rig):
    rig.run()
    rig.version["low_block"] += 1

    rig.run()

    assert sorted(rig.played) == [
        ("high_press", "low_block"),
        ("low_block", "tiki_taka"),
    ]


def test_a_stored_match_keeps_the_restarts_and_losses_its_replay_gave(rig, monkeypatch):
    losses = {"match": "x", "turnovers": [{"kind": "pass"}]}
    monkeypatch.setattr(tournament, "analyse_replay", lambda npz: ([], losses))
    rig.run()
    rig.played.clear()

    rig.run()

    assert rig.played == []
    stored = [json.loads(p.read_text()) for p in rig.store.root.rglob("*.json")]
    assert len(stored) == 3 and all(r["losses"] == losses and r["restarts"] == [] for r in stored)


def test_a_match_is_stored_only_if_its_code_is_unchanged_when_it_finishes(rig):
    # Editing tiki_taka during the first match leaves that match's code as it was, so it is stored.
    # The two tiki_taka matches finish under changed code, so neither is stored.
    def edit():
        rig.version["tiki_taka"] = 9

    rig.on_play = edit

    rig.run()

    stored = [json.loads(p.read_text()) for p in rig.store.root.rglob("*.json")]
    assert [(r["a"], r["b"]) for r in stored] == [("high_press", "low_block")]


def test_a_no_save_run_plays_and_stores_nothing(rig, monkeypatch):
    monkeypatch.setattr("sys.argv", ["round_robin.py", "--sequential", "--no-save", *_CONFIGS])
    tournament.main()

    assert len(rig.played) == 3
    assert not list(rig.store.root.rglob("*.json"))


def test_a_run_with_other_match_seconds_plays_every_match_again(rig, monkeypatch):
    rig.run()
    rig.played.clear()

    monkeypatch.setattr("sys.argv", ["round_robin.py", "--sequential", "--match-seconds", "300", *_CONFIGS])
    tournament.main()

    assert len(rig.played) == 3
    assert len(list(rig.store.root.iterdir())) == 2  # one environment per match length
