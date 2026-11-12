"""`evaluate.py` end to end with matches, fingerprints and the catalog stubbed: which matches each round
plays, what the match results store saves them from, and what the flags change.

A player's fingerprint is its name plus a version number, so "editing" a strategy is bumping its
version. `tests/replay/test_match_results.py` covers the store itself.
"""

from __future__ import annotations

import json
from pathlib import Path

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
        if name.endswith(".strategy"):  # a frozen file: the code of the strategy it was built from, then
            base = Path(name).stem.split("@")[0]
            return f"{self.rig.frozen_version}-{base}"
        return f"{self.rig.version[name]}-{name}"  # version first: the tool keeps 16 characters


@pytest.fixture
def rig(tmp_path, monkeypatch):
    rig = type("Rig", (), {})()
    rig.played, rig.kwargs = [], []
    rig.version = {n: 0 for n in [*CORE, *RESERVE, "c"]}
    rig.stats = {}  # (a, b) -> stats of that match
    rig.on_play = None
    rig.frozen_version = 0  # a frozen file's code is its strategy at version 0
    rig.score = lambda a, b, seed: (1, 0)
    rig.crash = set()  # (a, b) pairs whose match raises

    def fake_run_match(a, b, run_dir=None, **kwargs):
        rig.played.append((a, b, kwargs["sim_noise"].seed))
        rig.kwargs.append(kwargs)
        if rig.on_play is not None:
            rig.on_play()
        if (a, b) in rig.crash:
            raise RuntimeError("rsim died")
        score_a, score_b = rig.score(a, b, kwargs["sim_noise"].seed)
        return MatchResult(
            config_a=a, config_b=b, score_a=score_a, score_b=score_b, stats=rig.stats.get((a, b), {"x": 1})
        )

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


FROZEN_A = "replays/eval/frozen/a@v0.strategy"


def test_the_frozen_champion_and_the_unchanged_champion_are_one_player(rig, monkeypatch):
    # Right after a release the frozen champion is the working-tree champion's code: two names, one
    # fingerprint. Both playing made the reference play itself and its matches go by either name.
    monkeypatch.setattr(evaluate, "at_ref", lambda name, ref: FROZEN_A)
    run_dir = rig.run("c")

    assert not any("a" in (x, y) for x, y, _ in rig.played)  # the working-tree "a" is left out
    assert ("c", FROZEN_A) in {(x, y) for x, y, _ in rig.played}
    report = json.loads((run_dir / "report.json").read_text())
    assert report["core"] == ["b", "d"]
    assert report["candidates"]["c"]["per_opponent"]["b"]["matches"] >= 1  # paired against the reference

    rig.version["a"] += 1  # the champion edited since the release: two players again
    rig.run()
    assert {("a", "b"), ("a", "d")} <= {(x, y) for x, y, _ in rig.played}


def test_the_reference_is_the_champion_named_at_the_release_not_one_promoted_since(rig, monkeypatch):
    real_git = evaluate._git
    released = json.dumps({"champion": "b", "core": CORE})
    monkeypatch.setattr(evaluate, "_git", lambda *a: released if a[0] == "show" else real_git(*a))
    run_dir = rig.run()  # today's players.json names "a"
    assert json.loads((run_dir / "report.json").read_text())["reference"] == "b"


def test_the_champion_at_a_release_without_a_players_file_is_todays(monkeypatch):
    monkeypatch.setattr(evaluate, "_git", lambda *a: None)
    assert evaluate.champion_at("v2.5.0", "pendulum") == "pendulum"


def test_no_release_tag_stops_instead_of_using_todays_code(monkeypatch):
    monkeypatch.setattr(evaluate, "_git", lambda *a: None)
    with pytest.raises(SystemExit, match="No release tag"):
        evaluate.latest_release()


def test_the_verdict_does_not_depend_on_later_rounds_already_in_the_store(rig, monkeypatch):
    monkeypatch.setattr(evaluate, "MAX_ROUNDS", 1)
    first = json.loads((rig.run("c") / "report.json").read_text())["candidates"]["c"]

    # a later run leaves round 2 in the store, with very different scores
    monkeypatch.setattr(evaluate, "MAX_ROUNDS", 2)
    rig.score = lambda a, b, seed: (1, 0) if seed == 1 else (9, 0)
    rig.run("c")

    monkeypatch.setattr(evaluate, "MAX_ROUNDS", 1)
    again = json.loads((rig.run("c") / "report.json").read_text())["candidates"]["c"]
    assert rig.played == []
    assert (again["p_better"], again["margin"], again["margin_sd"]) == (
        first["p_better"],
        first["margin"],
        first["margin_sd"],
    )


def test_a_crashed_match_is_reported_not_stored_and_the_run_finishes(rig):
    rig.crash = {("b", "d")}
    run_dir = rig.run()

    summary = json.loads((run_dir / "round_1" / "summary.json").read_text())
    assert [(f["config_a"], f["config_b"]) for f in summary["failed"]] == [("b", "d")]
    assert len(summary["results"]) == 2
    assert json.loads((run_dir / "report.json").read_text())["failed"][0]["error"] == "RuntimeError('rsim died')"
    rig.crash = set()
    rig.run()
    assert [(x, y) for x, y, _ in rig.played] == [("b", "d")]  # not stored, so played next time


def test_same_code_keeps_the_first_name_of_each_fingerprint():
    fps = {"ref": "1", "cand": "2", "a": "1", "b": "3", "c": "2"}
    assert evaluate.same_code(["ref", "cand", "a", "b", "c"], fps) == {"a": "ref", "c": "cand"}


def test_confirm_plays_fresh_seeds_and_ignores_the_usual_ones(rig, monkeypatch):
    monkeypatch.setattr(evaluate, "MAX_ROUNDS", 2)
    rig.run("c")
    run_dir = rig.run("c", "--confirm")

    assert rig.played and {seed for *_, seed in rig.played} <= {101, 102}  # nothing reused from seeds 1-2
    report = json.loads((run_dir / "report.json").read_text())
    assert report["seeds"][0] == 101
    assert json.loads((run_dir / "round_1" / "summary.json").read_text())["seed"] == 101


def test_the_candidate_is_also_measured_against_todays_champion_when_shared_code_moved(rig, tmp_path, monkeypatch):
    # today's champion "d" sits last in the core; the release froze it before a shared-code change
    players = tmp_path / "players_d.json"
    players.write_text(json.dumps({"champion": "d", "core": CORE}))
    monkeypatch.setattr(evaluate, "at_ref", lambda name, ref: f"replays/eval/frozen/{name}@v0.strategy")
    rig.version["d"] = 1  # differs from its frozen copy (version 0)
    monkeypatch.setattr(evaluate, "MAX_ROUNDS", 2)
    run_dir = rig.run("c", "--players", str(players))

    r = json.loads((run_dir / "report.json").read_text())["candidates"]["c"]
    vs = r["vs_todays_champion"]
    # paired against "a" and "b" in every round: "d" meets them in the candidate's colour
    assert vs["name"] == "d" and vs["pairs"] == 2 * r["rounds"] and vs["head_to_head"] == r["rounds"]


def test_no_second_estimate_when_todays_champion_is_the_reference(rig):
    run_dir = rig.run("c")  # the stub reference is "a" itself
    assert "vs_todays_champion" not in json.loads((run_dir / "report.json").read_text())["candidates"]["c"]
