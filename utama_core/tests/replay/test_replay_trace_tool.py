import importlib.util
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from utama_core.entities.referee.referee_command import RefereeCommand as C

_TOOL = Path(__file__).resolve().parents[3] / "tools" / "replay_trace.py"
_spec = importlib.util.spec_from_file_location("replay_trace", _TOOL)
replay_trace = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(replay_trace)


def _replay(tmp_path, n=60):
    commands = np.array([C.NORMAL_START.value] * 30 + [C.STOP.value] * 30, dtype=np.int8)
    friendly = np.zeros((n, 2, 2))
    friendly[:, 0] = (1.3, 0.0)
    friendly[:, 1] = (3.0, 0.0)
    enemy = np.full((n, 1, 2), (0.0, 2.0))
    path = tmp_path / "a_vs_b.npz"
    np.savez(
        path,
        ts=50.0 + np.arange(n) / 60.0,
        has_referee=np.ones(n, bool),
        referee_command=commands,
        ball_p=np.tile([1.0, 0.0, 0.0], (n, 1)),
        ball_v=np.zeros((n, 3)),
        friendly_p=friendly,
        friendly_ids=np.array([4, 5]),
        friendly_has_ball=np.zeros((n, 2), bool),
        enemy_p=enemy,
        enemy_ids=np.array([2]),
        enemy_has_ball=np.zeros((n, 1), bool),
    )
    statuses = [(i, SimpleNamespace(status_message="Keep-out circle violation" if i >= 30 else None)) for i in range(n)]
    with open(tmp_path / "a_vs_b.sparse_referee.pkl", "wb") as f:
        pickle.dump(statuses, f)
    return path


def test_trace_prints_each_referee_change_with_its_reason_and_the_nearest_robots(tmp_path):
    lines = replay_trace.trace(_replay(tmp_path), 0.0, 1.0, every=1000)

    assert len(lines) == 2
    assert lines[0].split()[:2] == ["0.00", "NORMAL_START"]
    assert "near F 4@0.30" in lines[0] and "E 2@2.24" in lines[0]
    assert lines[1].split()[:2] == ["0.50", "STOP"]
    assert lines[1].endswith("| Keep-out circle violation")


def test_trace_keeps_to_the_window(tmp_path):
    lines = replay_trace.trace(_replay(tmp_path), 0.6, 0.8, every=6)
    assert [line.split()[0] for line in lines] == ["0.60", "0.70", "0.80"]


def _frames(n=120):
    """Synthetic arrays in the npz's layout: 60 Hz, ball at the origin and at rest, friendly 4 at
    0.3 m, friendly 5 at 3 m, enemy 2 at 3 m; no one has the ball, NORMAL_START throughout."""
    return {
        "ts": 1.0 + np.arange(n) / 60.0,
        "has_referee": np.ones(n, bool),
        "referee_command": np.full(n, C.NORMAL_START.value, dtype=np.int8),
        "ball_p": np.zeros((n, 3)),
        "ball_v": np.zeros((n, 3)),
        "friendly_p": np.tile([[0.3, 0.0], [-3.0, 0.0]], (n, 1, 1)),
        "friendly_ids": np.array([4, 5]),
        "friendly_has_ball": np.zeros((n, 2), bool),
        "enemy_p": np.tile([[0.0, 3.0]], (n, 1, 1)),
        "enemy_ids": np.array([2]),
        "enemy_has_ball": np.zeros((n, 1), bool),
    }


def _no_status(n):
    return [None] * n


def test_a_kick_is_a_sharp_speed_rise_named_by_the_nearest_robot():
    z = _frames()
    z["ball_v"][60:, 0] = 2.0  # at rest, then 2 m/s from tick 60 (t = 1.0 s from the first frame)
    evs = replay_trace.events(z, _no_status(120), [], 0.0, 3.0)

    kicks = [e for e in evs if e["kind"] == "KICK"]
    assert len(kicks) == 1
    assert kicks[0]["t"] == 1.0
    assert (kicks[0]["side"], kicks[0]["robot_id"]) == ("ours", 4)
    assert abs(kicks[0]["dist"] - 0.3) < 1e-9


def test_a_kick_is_counted_once_within_the_cooldown():
    z = _frames()
    z["ball_v"][60, 0] = 0.6  # rise of 0.6 m/s in one tick
    z["ball_v"][61:, 0] = 1.5  # rises again one tick later: the same kick
    kicks = [e for e in replay_trace.events(z, _no_status(120), [], 0.0, 3.0) if e["kind"] == "KICK"]
    assert len(kicks) == 1


def test_a_slow_roll_is_not_a_kick():
    z = _frames()
    z["ball_v"][:, 0] = np.linspace(0.0, 0.4, 120)  # 0.4 m/s over 2 s: no tick rises 0.5 m/s
    assert not [e for e in replay_trace.events(z, _no_status(120), [], 0.0, 3.0) if e["kind"] == "KICK"]


def test_possession_and_referee_changes_are_events_with_the_status_reason():
    z = _frames()
    z["friendly_has_ball"][30:, 0] = True  # F from t = 1.5
    z["referee_command"][40:] = C.STOP.value  # STOP from t = 1.67
    statuses = [None] * 40 + ["Ball out of bounds"] * 80
    evs = replay_trace.events(z, statuses, [], 0.0, 3.0)

    hold = [e for e in evs if e["kind"] == "HOLD"]
    assert [(e["from"], e["to"]) for e in hold] == [("-", "F")]
    ref = [e for e in evs if e["kind"] == "REF"]
    assert [(e["command"], e["status"]) for e in ref] == [("STOP", "Ball out of bounds")]
    assert ref[0]["text"].endswith("| Ball out of bounds")


def test_fouls_are_events_by_side_and_only_inside_the_window():
    z = _frames()
    fouls = [
        {"rule": "keeps_clear", "sim_time": 1.5, "side": "friendly", "robot_id": 4, "tactic": "T"},
        {"rule": "out_of_bounds", "sim_time": 2.5, "side": "enemy", "robot_id": 2, "tactic": None},
        {"rule": "keeps_clear", "sim_time": 9.0, "side": "friendly", "robot_id": 5, "tactic": None},
    ]
    foul_evs = [e for e in replay_trace.events(z, _no_status(120), fouls, 1.0, 3.0) if e["kind"] == "FOUL"]
    assert [(e["side"], e["rule"], e["robot_id"]) for e in foul_evs] == [
        ("ours", "keeps_clear", 4),
        ("theirs", "out_of_bounds", 2),
    ]


def test_event_lines_are_capped_with_a_count_of_the_rest(tmp_path):
    import json

    path = _replay(tmp_path)
    fouls = [{"rule": "keeps_clear", "sim_time": i / 100, "side": "friendly", "robot_id": 4} for i in range(45)]
    path.with_name("a_vs_b.stats.json").write_text(json.dumps({"fouls": fouls}))
    lines = replay_trace.event_lines(path, 0.0, 1.0)
    assert len(lines) == replay_trace.MAX_EVENT_LINES + 1
    assert lines[-1] == "... 6 more"  # 45 fouls and the 0.5 s referee change, 40 shown


def test_rows_carry_the_nearest_robot_ids_and_distances_as_data(tmp_path):
    z, statuses, _ = replay_trace._load(_replay(tmp_path))
    first = replay_trace.rows(z, statuses, 0.0, 0.0)[0]
    assert first["referee"] == "NORMAL_START"
    assert first["near_F"] == {"id": 4, "dist": pytest.approx(0.3, abs=1e-6)}
    assert first["near_E"] == {"id": 2, "dist": pytest.approx(2.24, abs=0.01)}
