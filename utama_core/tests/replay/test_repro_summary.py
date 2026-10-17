"""`tools/repro_from_replay.py --summary`: per-robot rows from match_log events, and their text."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

_TOOL = Path(__file__).resolve().parents[3] / "tools" / "repro_from_replay.py"
_spec = importlib.util.spec_from_file_location("repro_from_replay", _TOOL)
repro = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(repro)


def _sample(t, robots, ball=(0.0, 0.0), ref="NORMAL_START"):
    return {"t": t, "referee": ref, "ball": ball, "robots": robots}


def _intention(t, slot, robots, note=None):
    return SimpleNamespace(sim_time=t, tactic_id=slot, robot_ids=tuple(robots), note=note)


def _trace(t, key, value):
    return SimpleNamespace(sim_time=t, key=key, value=value)


def test_rows_are_every_quarter_second_with_tactic_target_and_distance():
    samples = [
        _sample(0.0, {1: (3.0, 4.0), 3: (0.3, 0.0)}),
        _sample(0.1, {1: (3.0, 4.0), 3: (0.3, 0.0)}),  # skipped: under 0.25 s after the last row
        _sample(0.25, {1: (3.0, 4.0), 3: (0.3, 0.0)}, ref="STOP"),
        _sample(0.5, {1: (3.0, 4.0), 3: (0.3, 0.0)}),
    ]
    events = [
        _intention(0.2, "overload", [1], note="DecoyOverloadTactic"),
        _intention(0.4, "overload", [], note="released (was committed=False, held 0 committed ticks)"),
        _trace(0.1, "give_and_go.pass_target", {"receiver_id": 3, "x": 1.0, "y": 2.0}),
        _trace(0.1, "decoy_and_overload.lure_target", {"x": -4.0, "y": 2.5}),  # no robot id: not attributed
    ]
    rows = repro.summary_rows(samples, [1, 3], events)

    assert [(r["t"], r["robot"]) for r in rows] == [(0.0, 1), (0.0, 3), (0.25, 1), (0.25, 3), (0.5, 1), (0.5, 3)]
    by = {(r["t"], r["robot"]): r for r in rows}
    assert by[(0.0, 1)]["tactic"] == "-"
    assert by[(0.25, 1)]["tactic"] == "DecoyOverloadTactic"
    assert by[(0.5, 1)]["tactic"] == "-"  # released at 0.4
    assert by[(0.25, 3)]["target"] == "give_and_go.pass_target (1.00,2.00)"
    assert by[(0.25, 1)]["target"] == "-"
    assert by[(0.25, 1)]["dist_ball"] == pytest.approx(5.0)
    assert by[(0.25, 1)]["ref"] == "STOP" and by[(0.5, 1)]["ref"] == "NORMAL_START"


def test_a_slot_reassigned_to_fewer_robots_drops_the_ones_it_lost():
    samples = [_sample(0.0, {3: (0.0, 0.0), 4: (1.0, 0.0)}), _sample(0.5, {3: (0.0, 0.0), 4: (1.0, 0.0)})]
    events = [
        _intention(0.0, "attack", [3, 4], note="GiveAndGoTactic"),
        _intention(0.3, "attack", [4]),  # robot 3 moved to another slot in the same tick, not logged here
    ]
    rows = repro.summary_rows(samples, [3, 4], events)
    late = {r["robot"]: r["tactic"] for r in rows if r["t"] == 0.5}
    assert late == {3: "-", 4: "GiveAndGoTactic"}


def test_ball_missing_gives_no_distance():
    rows = repro.summary_rows([_sample(0.0, {1: (1.0, 1.0)}, ball=None)], [1], [])
    assert rows[0]["dist_ball"] is None
    assert "ball    - m" in repro.format_summary(rows)[0]


def test_format_caps_the_lines_and_counts_the_rest():
    rows = [
        {"t": i / 4, "robot": 1, "tactic": "-", "pos": (0.0, 0.0), "dist_ball": 1.0, "ref": "STOP", "target": "-"}
        for i in range(45)
    ]
    lines = repro.format_summary(rows)
    assert len(lines) == repro.SUMMARY_MAX_LINES + 1
    assert lines[-1] == "... 5 more"
    assert "robot 1" in lines[0] and "ref STOP" in lines[0]
