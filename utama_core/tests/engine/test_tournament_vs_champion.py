"""Tests for `round_robin.py --vs-champion`: the day-to-day health check, one match per active
strategy against the champion (see `docs/STRATEGY_DEVELOPMENT.md`). Pure, no sim: `run_match` is
stubbed and the catalog, retired set and opponents file are faked.
"""

from __future__ import annotations

import json

import pytest

from tools.evaluation import round_robin as tournament
from tools.evaluation.sim_settings import sim_noise_from_spec


class _Graph:
    """Stands in for CodeGraph: the config names here are not strategies."""

    def base_fingerprint(self, entries):
        return "base"

    def strategy_fingerprint(self, name):
        return f"fp-{name}"


@pytest.fixture(autouse=True)
def _no_real_fingerprints(monkeypatch):
    monkeypatch.setattr(tournament, "CodeGraph", _Graph)


def _stub_result(config_a_name, config_b_name, **_kwargs):
    return tournament.MatchResult(config_a=config_a_name, config_b=config_b_name, score_a=1, score_b=0, stats=None)


def test_vs_champion_pairs_is_every_other_config_against_the_champion_then_champion_vs_first_opponent():
    names = ["b", "champ", "a", "d"]

    assert tournament.vs_champion_pairs(names, "champ", "a") == [
        ("a", "champ"),
        ("b", "champ"),
        ("d", "champ"),
        ("champ", "a"),
    ]


def test_vs_champion_pairs_plays_the_champion_only_as_config_b_except_its_own_match():
    pairs = tournament.vs_champion_pairs(["a", "b", "champ"], "champ", "b")

    assert [a for a, b in pairs if a == "champ"] == ["champ"]
    assert all(b == "champ" for a, b in pairs if a != "champ")


def test_vs_champion_pairs_refuses_a_missing_champion_or_first_opponent():
    with pytest.raises(ValueError, match="champion"):
        tournament.vs_champion_pairs(["a", "b"], "champ", "a")
    with pytest.raises(ValueError, match="first opponent"):
        tournament.vs_champion_pairs(["a", "champ"], "champ", "b")


def test_vs_champion_plays_active_strategies_against_champion_with_reference_seed_one(tmp_path, monkeypatch, capsys):
    calls = []

    def _fake_run_match(config_a_name, config_b_name, **kwargs):
        calls.append((config_a_name, config_b_name, kwargs.get("sim_noise")))
        return _stub_result(config_a_name, config_b_name)

    monkeypatch.setattr(tournament, "run_match", _fake_run_match)
    monkeypatch.setattr(tournament, "REPLAY_BASE_PATH", tmp_path)
    monkeypatch.setattr(tournament, "_CONFIG_NAMES", ["a", "b", "champ"])
    monkeypatch.setattr(tournament, "load_opponents", lambda *_a, **_k: {"champion": "champ", "opponents": ["a", "b"]})
    monkeypatch.setattr("sys.argv", ["round_robin.py", "--sequential", "--vs-champion"])

    tournament.main()

    seeded = sim_noise_from_spec("reference:2")
    assert calls == [("a", "champ", seeded), ("b", "champ", seeded), ("champ", "a", seeded)]
    assert "reference:2" in capsys.readouterr().out
    # Same conditions as evaluate.py's round-2 champion fixtures, so the store keeps these matches as those fixtures.
    stored = [json.loads(p.read_text()) for p in (tmp_path / "match_results").rglob("*.json")]
    assert len(stored) == 3 and {r["seed"] for r in stored} == {2}
    # Output still lands in replays/tournament_<id>, which downstream tools read.
    assert list(tmp_path.glob("tournament_*/summary.json"))


def test_noise_is_not_defaulted_outside_vs_champion_and_an_explicit_spec_wins(tmp_path, monkeypatch):
    calls = []

    def _fake_run_match(config_a_name, config_b_name, **kwargs):
        calls.append(kwargs.get("sim_noise"))
        return _stub_result(config_a_name, config_b_name)

    monkeypatch.setattr(tournament, "run_match", _fake_run_match)
    monkeypatch.setattr(tournament, "REPLAY_BASE_PATH", tmp_path)
    monkeypatch.setattr(tournament, "_CONFIG_NAMES", ["a", "champ"])
    monkeypatch.setattr(tournament, "load_opponents", lambda *_a, **_k: {"champion": "champ", "opponents": ["a"]})

    monkeypatch.setattr("sys.argv", ["round_robin.py", "--sequential", "--no-save", "a", "champ"])
    tournament.main()
    assert calls == [None]

    calls.clear()
    monkeypatch.setattr(
        "sys.argv",
        ["round_robin.py", "--sequential", "--vs-champion", "--sim-noise", "reference:7"],
    )
    tournament.main()
    assert calls == [sim_noise_from_spec("reference:7"), sim_noise_from_spec("reference:7")]


def test_vs_champion_with_pair_is_an_error_and_plays_nothing(tmp_path, monkeypatch):
    def _no_match(*_args, **_kwargs):
        raise AssertionError("--vs-champion with --pair must not play a match")

    monkeypatch.setattr(tournament, "run_match", _no_match)
    monkeypatch.setattr(tournament, "REPLAY_BASE_PATH", tmp_path)
    monkeypatch.setattr("sys.argv", ["round_robin.py", "--sequential", "--vs-champion", "--pair", "a", "b"])

    with pytest.raises(SystemExit, match="--vs-champion and --pair"):
        tournament.main()
