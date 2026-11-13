"""`utama_core.dashboard.views.strategies`: the Strategies tab's payloads, from a small fake `replays/`."""

from __future__ import annotations

import json

import pytest

from utama_core.dashboard.views import strategies as view


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _row(a, b, score_a, score_b, **extra):
    return {"config_a": a, "config_b": b, "score_a": score_a, "score_b": score_b, "seed": 1, **extra}


@pytest.fixture
def replays(tmp_path):
    """One finished evaluate.py run: mirage the candidate (dropped), the reference a frozen pendulum."""
    run = tmp_path / "eval_20261112_100000"
    frozen = "replays/eval/frozen/pendulum@ab12cd34.strategy"
    _write(
        run / "round_1" / "summary.json",
        {
            "run_id": "eval_20261112_100000/round_1",
            "round": 1,
            "seed": 1,
            "results": [
                _row("mirage", "wolfpack", 3, 1, stats={"stall_events": [{"t": 40.0}], "shots": 9}),
                _row(frozen, "mirage", 2, 0),
                _row("wolfpack", "mirage", 1, 1, reused=True),  # played by an earlier run
            ],
            "strategies": {
                "mirage": {"matches": 2, "goals_for": 3, "shots": 10, "stalled": 1},
                "wolfpack": {"matches": 1, "goals_for": 1, "shots": 4, "stalled": 0},
                frozen: {"matches": 1, "goals_for": 2, "shots": 8, "stalled": 0},
            },
        },
    )
    (run / "round_1" / "mirage_vs_wolfpack.npz").write_bytes(b"")
    _write(
        run / "report.json",
        {
            "run_id": "eval_20261112_100000",
            "reference": frozen,
            "candidates": {
                "mirage": {
                    "verdict": "dropped",
                    "decided_in": "screen",
                    "rounds": 2,
                    "screen": {"n": 8, "mean": -1.2, "se": 0.4, "p": 0.99},
                    "stalls": {"too": "big"},
                }
            },
        },
    )
    return tmp_path


def _strategy(payload, name):
    (s,) = [s for s in payload["strategies"] if s["name"] == name]
    return s


def test_frozen_files_and_old_names_show_as_the_catalog_name():
    assert view.display_name("replays/eval/frozen/pendulum@ab12cd34.strategy") == "pendulum"
    assert view.display_name("build_split_shape_kernel_strategy") == "pendulum"
    assert view.display_name("high_press") == "wolfpack"
    assert view.display_name("split_shape@cffe3a28") == "pendulum"  # how evaluate.py records the reference


def test_the_summary_drops_the_name_and_the_boilerplate():
    doc = "`wolfpack` (until 2026-11-12 `high_press`): hunts the ball high up. The pickers only it uses are here."
    assert view.summary_of(doc) == "Hunts the ball high up."
    assert view.summary_of("Helpers shared by more than one strategy.") == "Helpers shared by more than one strategy."


def test_signals_come_from_a_real_round_not_a_newer_exhibition_match(replays, monkeypatch):
    monkeypatch.setattr(view, "MIN_SIGNAL_MATCHES", 2)  # mirage plays 2 in the fixture's round
    _write(
        replays / "tournament_20261113_000000" / "summary.json",
        {"run_id": "tournament_20261113_000000", "results": [], "strategies": {"mirage": {"matches": 1, "shots": 99}}},
    )
    sig = _strategy(view.load_strategies(replays), "mirage")["signals"]
    assert sig["run_id"] == "eval_20261112_100000/round_1"


def test_runs_are_newest_first_by_date_whatever_their_prefix(replays):
    _write(replays / "tournament_20261110_000000" / "summary.json", {"run_id": "tournament_20261110_000000"})
    assert [r["run_id"] for r in view.load_runs(replays)] == ["eval_20261112_100000", "tournament_20261110_000000"]


def test_every_catalog_strategy_is_listed_once_with_its_role(replays):
    payload = view.load_strategies(replays)
    names = [s["name"] for s in payload["strategies"]]
    assert sorted(names) == sorted(set(names)) and "pendulum" in names
    assert _strategy(payload, payload["champion"])["role"] == "champion"
    assert {_strategy(payload, n)["role"] for n in payload["core"] if n != payload["champion"]} == {"core"}
    assert not any(n.endswith(".strategy") for n in names)


def test_a_strategy_carries_its_description_file_and_tactics(replays):
    wolfpack = _strategy(view.load_strategies(replays), "wolfpack")
    assert wolfpack["summary"] and wolfpack["file"] == "utama_core/strategy/wolfpack.py"
    assert {"GiveAndGoTactic", "PressAndContainTactic"} <= {t["name"] for t in wolfpack["tactics"]}
    assert "high_press" in wolfpack["old_names"]


def test_a_candidates_verdict_is_shown_against_the_champion_by_name(replays):
    (v,) = _strategy(view.load_strategies(replays), "mirage")["verdicts"]
    assert (v["verdict"], v["decided_in"], v["rounds"], v["mean"], v["n"]) == ("dropped", "screen", 2, -1.2, 8)
    assert v["reference"] == "pendulum" and v["date"] == "2026-11-12"


def test_matches_are_from_the_strategys_side_with_their_replays_and_reused_ones_once(replays):
    matches = _strategy(view.load_strategies(replays), "mirage")["matches"]
    assert sorted((m["opponent"], tuple(m["score"])) for m in matches) == [
        ("pendulum (last release)", (0, 2)),
        ("wolfpack", (3, 1)),
    ]
    by_opponent = {m["opponent"]: m["replay"] for m in matches}
    assert by_opponent["wolfpack"] == "eval_20261112_100000/round_1/mirage_vs_wolfpack.npz"
    assert by_opponent["pendulum (last release)"] is None


def test_the_frozen_files_matches_are_not_todays_strategys(replays):
    assert _strategy(view.load_strategies(replays), "pendulum")["matches"] == []


def test_signals_are_per_match_beside_the_other_strategies_average(replays):
    sig = _strategy(view.load_strategies(replays), "mirage")["signals"]
    rows = {r["key"]: r for r in sig["rows"]}
    assert rows["shots"]["value"] == 5.0  # 10 shots in 2 matches
    assert rows["shots"]["field"] == 4.0  # wolfpack; the frozen pendulum file is not in the field
    assert rows["stalled"]["value"] == 0.5
    assert rows["fouls"]["value"] is None  # not in this round's table


def test_a_run_groups_its_rounds_and_its_report_without_the_bulky_stalls(replays):
    (run,) = view.load_runs(replays)
    assert run["run_id"] == "eval_20261112_100000" and run["reference"] == "pendulum"
    assert run["report"]["candidates"]["mirage"]["verdict"] == "dropped"
    assert "stalls" not in run["report"]["candidates"]["mirage"]
    results = run["rounds"][0]["results"]
    assert [r["display_a"] for r in results] == ["mirage", "pendulum", "wolfpack"]
    # only the stall flag of each match's stats is sent: the full stats are megabytes over all runs
    assert [r["stalled"] for r in results] == [True, False, False]
    assert not any("stats" in r for r in results)


def test_an_older_round_robin_is_one_round_and_finds_side_tagged_replays(tmp_path):
    run = tmp_path / "tournament_1"
    _write(
        run / "summary.json",
        {"run_id": "tournament_1", "results": [_row("a", "b", 1, 0, a_is_right=False, a_kicks_off=True)]},
    )
    (run / "a_vs_b_LK.pkl").write_bytes(b"")
    (loaded,) = view.load_runs(tmp_path)
    assert loaded["report"] is None
    assert loaded["rounds"][0]["results"][0]["replay"] == "tournament_1/a_vs_b_LK.pkl"


def test_no_replays_folder_gives_empty_pages(tmp_path):
    assert view.load_runs(tmp_path / "missing") == []
    assert all(not s["matches"] for s in view.load_strategies(tmp_path / "missing")["strategies"])


class _Job:
    def __init__(self, done=False):
        self.done = done

    def poll(self):
        return 0 if self.done else None


def test_maps_start_once_in_the_background_then_come_back_by_catalog_name(replays, monkeypatch):
    monkeypatch.setattr(view, "_jobs", {})
    started = []
    start = lambda cmd, **kw: started.append(cmd) or _Job()  # noqa: E731
    round_id = "eval_20261112_100000/round_1"
    assert view.maps_of(round_id, replays, start)["status"] == "computing"
    assert view.maps_of(round_id, replays, start)["status"] == "computing"
    assert len(started) == 1 and "--data" in started[0]
    _write(
        replays / round_id / view.MAPS_FILE,
        {"vmax": {"attack": 0.1}, "strategies": {"high_press": {"shots": {}}, "pendulum@ab12cd34": {"shots": {}}}},
    )
    maps = view.maps_of(round_id, replays, start)
    assert maps["status"] == "ready" and maps["vmax"] == {"attack": 0.1}
    assert list(maps["strategies"]) == ["wolfpack"]  # renamed; the frozen file left out


def test_maps_report_a_failed_job_once_then_retry(replays, monkeypatch):
    round_dir = (replays / "eval_20261112_100000" / "round_1").resolve()
    monkeypatch.setattr(view, "_jobs", {round_dir: _Job(done=True)})
    assert view.maps_of("eval_20261112_100000/round_1", replays, lambda cmd, **kw: _Job())["status"] == "failed"
    assert view.maps_of("eval_20261112_100000/round_1", replays, lambda cmd, **kw: _Job())["status"] == "computing"


def test_maps_never_leave_the_replays_folder(replays):
    assert view.maps_of("../../etc", replays)["status"] == "missing"
    assert view.maps_of("eval_20261112_100000/round_9", replays)["status"] == "missing"
