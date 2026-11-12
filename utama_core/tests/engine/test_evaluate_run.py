"""`evaluate.py` end to end with matches, fingerprints and the catalog stubbed: which matches each round
plays, what the match results store saves them from, and what the flags change.

A player's fingerprint is its name plus a version number, so "editing" a strategy is bumping its
version. `tests/replay/test_match_results.py` covers the store itself.
"""

from __future__ import annotations

import json

import pytest

from tools.evaluation import evaluate
from tools.evaluation.match import MatchResult
from utama_core.replay import match_results

CORE = ["a", "b", "d"]
RESERVE = ["r1", "r2", "r3"]


class _Graph:
    def __init__(self, rig):
        self.rig = rig

    def base_fingerprint(self, entries):
        return "base"

    def strategy_fingerprint(self, name):
        return f"{self.rig.version[name]}-{name}"  # version first: the tool keeps 16 characters


@pytest.fixture
def rig(tmp_path, monkeypatch):
    rig = type("Rig", (), {})()
    rig.played, rig.kwargs = [], []
    rig.version = {n: 0 for n in [*CORE, *RESERVE, "c"]}
    rig.stats = {}  # (a, b) -> stats of that match
    rig.on_play = None

    def fake_run_match(a, b, run_dir=None, **kwargs):
        rig.played.append((a, b, kwargs["sim_noise"].seed))
        rig.kwargs.append(kwargs)
        if rig.on_play is not None:
            rig.on_play()
        return MatchResult(config_a=a, config_b=b, score_a=1, score_b=0, stats=rig.stats.get((a, b), {"x": 1}))

    players = tmp_path / "players.json"
    players.write_text(json.dumps({"champion": "a", "core": CORE}))
    monkeypatch.setattr(evaluate, "run_match", fake_run_match)
    monkeypatch.setattr(evaluate, "REPLAY_BASE_PATH", tmp_path)
    monkeypatch.setattr(evaluate, "CodeGraph", lambda: _Graph(rig))
    monkeypatch.setattr(evaluate, "_CONFIG_NAMES", [*CORE, *RESERVE, "c"])
    monkeypatch.setattr(evaluate, "WORKERS", 1)  # in this process, so the stubs apply
    monkeypatch.setattr(evaluate, "at_ref", lambda name, ref: name)  # the reference: the core's own "a"
    monkeypatch.setattr(evaluate, "latest_release", lambda: "v0")

    def run(*argv):
        rig.played.clear()
        rig.kwargs.clear()
        evaluate.main(["--players", str(players), *argv])
        (run_dir,) = sorted(tmp_path.glob("eval_*"))[-1:]
        return run_dir

    rig.run = run
    rig.store = match_results.open_store(tmp_path)
    return rig


def test_with_no_candidate_one_round_plays_every_pair_of_the_core(rig):
    run_dir = rig.run()

    # the core a, b, d; a is also the reference; the reserve r1, r2, r3 and c play only with --all
    assert sorted((x, y) for x, y, _ in rig.played) == [("a", "b"), ("a", "d"), ("b", "d")]
    assert {seed for *_, seed in rig.played} == {1}
    summary = json.loads((run_dir / "round_1" / "summary.json").read_text())
    assert len(summary["results"]) == 3 and "strategies" in summary
    assert not (run_dir / "round_2").exists()
    assert json.loads((run_dir / "report.json").read_text())["candidates"] == {}


def test_all_adds_every_other_strategy_in_the_catalog(rig):
    rig.run("--all")
    assert len(rig.played) == 7 * 6 // 2  # a, b, d, r1, r2, r3, c: every pair once


def test_a_second_run_plays_nothing_and_after_an_edit_only_that_strategys_matches_play(rig):
    rig.run()
    rig.run()
    assert rig.played == []

    rig.version["d"] += 1
    rig.run()
    assert sorted((x, y) for x, y, _ in rig.played) == [("a", "d"), ("b", "d")]


def test_a_later_round_is_the_same_round_robin_under_the_next_seed_with_colours_swapped(rig, monkeypatch):
    monkeypatch.setattr(evaluate, "MAX_ROUNDS", 2)
    run_dir = rig.run("c")

    round_2 = [(x, y) for x, y, seed in rig.played if seed == 2]  # equal results: undecided after round 1
    assert sorted(round_2) == [("a", "c"), ("b", "a"), ("b", "c"), ("d", "a"), ("d", "b"), ("d", "c")]

    rig.version["c"] += 1  # edit the candidate: only its matches play again, in both rounds
    rig.run("c")
    assert all("c" in (x, y) for x, y, _ in rig.played) and len(rig.played) == 6
    report = json.loads((run_dir / "report.json").read_text())
    assert report["candidates"]["c"]["verdict"] == "undecided"
    assert report["candidates"]["c"]["rounds"] == 2


def test_a_match_is_stored_only_if_its_code_is_unchanged_when_it_finishes(rig):
    def edit():
        rig.version["d"] = 9

    rig.on_play = edit
    rig.run()

    stored = {(r["a"], r["b"]) for r in map(json.loads, map(lambda p: p.read_text(), rig.store.root.rglob("*.json")))}
    assert ("a", "b") in stored
    assert not any("d" in pair for pair in stored)


def test_at_ref_falls_back_to_the_name_a_strategy_had_at_that_commit(monkeypatch, tmp_path):
    built = []

    def fake_frozen(name, ref, directory):
        built.append(name)
        if name != "split_shape":
            raise ValueError(f"no strategy {name}")
        return tmp_path / f"{name}@abc.strategy"

    monkeypatch.setattr(evaluate, "frozen", fake_frozen)
    assert evaluate.at_ref("pendulum", "v2.5.0").endswith("split_shape@abc.strategy")
    assert built == ["pendulum", "split_shape"]


def test_a_reused_match_is_linked_into_the_new_rounds_folder_from_the_run_that_played_it(rig, monkeypatch):
    def run_match_writing_files(a, b, run_dir=None, **kwargs):
        rig.played.append((a, b, kwargs["sim_noise"].seed))
        for suffix in (".npz", ".stats.json"):
            (run_dir / f"{evaluate.tag(a, b)}{suffix}").write_text(a + b)
        return MatchResult(config_a=a, config_b=b, score_a=1, score_b=0, stats={"x": 1})

    monkeypatch.setattr(evaluate, "run_match", run_match_writing_files)
    monkeypatch.setattr(evaluate, "analyse_replay", lambda npz: (None, None))
    first = rig.run()
    rig.version["d"] += 1
    second = rig.run()

    assert sorted((x, y) for x, y, _ in rig.played) == [("a", "d"), ("b", "d")]
    linked = second / "round_1" / "a_vs_b.npz"
    assert linked.is_symlink() and linked.resolve() == (first / "round_1" / "a_vs_b.npz").resolve()
    assert (second / "round_1" / "a_vs_b.stats.json").read_text() == "ab"
    assert len(list((second / "round_1").glob("*.npz"))) == 3  # every match of the round, played or reused
