"""Tests for `utama_core.dashboard.views.replay`'s `/replay/frames` payload."""

from __future__ import annotations

import json
import pickle

from utama_core.dashboard.views import replay as replay_view
from utama_core.engine.match_log import MatchLog
from utama_core.engine.tactic import TacticTag
from utama_core.entities.data.vector import Vector2D, Vector3D
from utama_core.entities.game import Ball, GameFrame, Robot
from utama_core.replay.entities import ReplayMetadata


def _frame(ts: float) -> GameFrame:
    robot = Robot(
        id=1, is_friendly=True, has_ball=False, p=Vector2D(0, 0), v=Vector2D(0, 0), a=Vector2D(0, 0), orientation=0.0
    )
    return GameFrame(
        ts=ts,
        my_team_is_yellow=True,
        my_team_is_right=True,
        friendly_robots={1: robot},
        enemy_robots={},
        ball=Ball(p=Vector3D(0, 0, 0), v=Vector3D(0, 0, 0), a=Vector3D(0, 0, 0)),
    )


def test_frames_payload_tactic_events_carry_each_tactics_tag(tmp_path, monkeypatch):
    # The Replay view colours robots by tag, and it only sees what this payload ships.
    replay = tmp_path / "match.pkl"
    with open(replay, "wb") as f:
        pickle.dump(ReplayMetadata(my_team_is_yellow=True, exp_friendly=1, exp_enemy=0), f)
        pickle.dump(_frame(0.0), f)
    log = MatchLog()
    log.intention(tick=1, sim_time=0.0, tactic_id="attack", robot_ids=(1,), tag=TacticTag.ATTACK)
    log.intention(tick=2, sim_time=0.1, tactic_id="defense", robot_ids=(2, 3), tag=TacticTag.DEFENSE)
    log.to_jsonl(tmp_path / "match.intentions.jsonl")
    monkeypatch.setattr(replay_view, "REPLAY_BASE_PATH", tmp_path)

    payload = json.loads(replay_view._frames_bytes({"path": "match.pkl"}))

    assert [(e["tactic_id"], e["tag"]) for e in payload["tactic_events"]] == [
        ("attack", "attack"),
        ("defense", "defense"),
    ]


def test_frames_payload_marks_goals_fouls_stalls_and_real_ball_losses_in_time_order(tmp_path, monkeypatch):
    # the timeline shows these as markers to jump to; every one is already on disk
    replay = tmp_path / "a_vs_b.pkl"
    with open(replay, "wb") as f:
        pickle.dump(ReplayMetadata(my_team_is_yellow=True, exp_friendly=1, exp_enemy=0), f)
        pickle.dump(_frame(0.0), f)
    log = MatchLog()
    log.referee(tick=1, sim_time=0.0, command="NORMAL_START", stage="NORMAL_FIRST_HALF", yellow_score=0, blue_score=0)
    log.referee(tick=2, sim_time=12.0, command="STOP", stage="NORMAL_FIRST_HALF", yellow_score=1, blue_score=0)
    log.referee(tick=3, sim_time=20.0, command="STOP", stage="NORMAL_FIRST_HALF", yellow_score=1, blue_score=0)
    log.to_jsonl(tmp_path / "a_vs_b.intentions.jsonl")
    foul = {"rule": "crashing", "sim_time": 9.5, "side": "friendly", "robot_id": 2, "tactic": "GiveAndGoTactic"}
    (tmp_path / "a_vs_b.stats.json").write_text(
        json.dumps(
            {
                # a crash is logged once per side at the same instant: one marker
                "fouls": [foul, {**foul, "side": "enemy", "robot_id": 3, "tactic": "ShadowAndMarkTactic"}],
                "stall_events": [{"kind": "COMMITTED_FROZEN", "sim_time": 30.0, "duration_s": 11.0, "diagnosis": ""}],
            }
        )
    )
    turnovers = [
        {"kind": "tackled", "t": 5.0, "tactic": "GiveAndGoTactic", "regained_after_s": None},
        {"kind": "tackled", "t": 6.0, "tactic": "GiveAndGoTactic", "regained_after_s": 0.1},  # flicker, not real
    ]
    real = [t for t in turnovers if replay_view.is_real(t)]
    monkeypatch.setattr(replay_view, "_analysis", lambda path: {"losses": real, "shots": []})
    monkeypatch.setattr(replay_view, "REPLAY_BASE_PATH", tmp_path)

    payload = json.loads(replay_view._frames_bytes({"path": "a_vs_b.pkl"}))

    assert [(m["sim_time"], m["kind"]) for m in payload["markers"]] == [
        (5.0, "loss"),
        (9.5, "foul"),
        (12.0, "goal"),
        (30.0, "stall"),
    ]
    assert "yellow" in payload["markers"][2]["label"] and "1-0" in payload["markers"][2]["label"]


def test_the_slow_analysis_is_kept_beside_the_replay_until_the_replay_changes(tmp_path, monkeypatch):
    replay = tmp_path / "a_vs_b.npz"
    replay.write_bytes(b"first")
    calls = []

    def analyse(path):
        calls.append(path)
        return {"turnovers": [], "chances": {"shots": [{"t": 3.0, "side": "enemy", "xg": 0.1}]}}

    monkeypatch.setattr(replay_view, "analyse_match", analyse)
    first = replay_view._analysis(replay)
    assert replay_view._analysis(replay) == first == {"losses": [], "shots": [{"t": 3.0, "side": "enemy"}]}
    assert len(calls) == 1
    replay.write_bytes(b"second, longer")  # a rerun match
    replay_view._analysis(replay)
    assert len(calls) == 2


def test_play_rows_end_each_situation_at_a_stoppage_or_the_next_side_winning_the_ball():
    from utama_core.engine.match_log import RefereeEvent

    def referee(t, command, note=None):
        return RefereeEvent(tick=0, sim_time=t, command=command, stage="", yellow_score=0, blue_score=0, note=note)

    starts = [
        {"t": 0.0, "kind": "kickoff", "side": "friendly", "third": "middle"},
        {"t": 4.0, "kind": "ball won", "side": "enemy", "third": "defensive"},
        {"t": 9.0, "kind": "ball won", "side": "friendly", "third": "middle"},
        {"t": 20.0, "kind": "free kick", "side": "friendly", "third": "attacking"},
    ]
    events = [referee(0.0, "NORMAL_START"), referee(15.0, "STOP", "Goal by Yellow"), referee(20.0, "NORMAL_START")]
    losses = [{"kind": "tackled", "t": 4.1}]
    rows = replay_view.play_rows(starts, events, losses, end_t=30.0)
    assert [(r["t"], r["t_end"], r["ended"]) for r in rows] == [
        (0.0, 4.0, "won by enemy (tackled)"),
        (4.0, 9.0, "won by friendly"),
        (9.0, 15.0, "Goal by Yellow"),
        (20.0, 30.0, None),
    ]


def _frames_from_columns(c: dict) -> list[dict]:
    """`framesFromColumns` in replay.js, line for line."""

    def team(t, i):
        return [
            {"id": rid, "x": t["x"][i][k] / 1000, "y": t["y"][i][k] / 1000}
            | {"vx": t["vx"][i][k] / 100, "vy": t["vy"][i][k] / 100, "orientation": t["o"][i][k] / 1000}
            | {"has_ball": bool(t["has_ball"][i][k])}
            for k, rid in enumerate(t["ids"])
            if t["seen"][i][k]
        ]

    b = c["ball"]
    return [
        {
            "ts": ts,
            "my_team_is_right": bool(c["right"][i]),
            "robots": {"friendly": team(c["friendly"], i), "enemy": team(c["enemy"], i)},
            "ball": (
                {"x": b["x"][i] / 1000, "y": b["y"][i] / 1000, "vx": b["vx"][i] / 100, "vy": b["vy"][i] / 100}
                if b["seen"][i]
                else None
            ),
        }
        for i, ts in enumerate(c["ts"])
    ]


def test_the_columns_rebuild_exactly_what_serialising_each_game_frame_gives(tmp_path):
    from utama_core.dashboard.views.referee import _serialise_ball, _serialise_robots
    from utama_core.replay.columnar_reader import load_columnar_replay
    from utama_core.replay.columnar_writer import (
        ColumnarReplayWriter,
        ColumnarReplayWriterConfig,
    )

    def robot(rid, x, has_ball=False):
        return Robot(
            id=rid,
            is_friendly=True,
            has_ball=has_ball,
            p=Vector2D(x, -0.12345),
            v=Vector2D(0.987, 0.0),
            a=Vector2D(0, 0),
            orientation=1.23456,
        )

    ball = Ball(p=Vector3D(0.5, 0.25, 0), v=Vector3D(1.234, 0, 0), a=Vector3D(0, 0, 0))
    frames = [  # robot 1 leaves, robot 3 joins: the array has a NaN row for each absent robot
        GameFrame(0.0, True, True, {1: robot(1, 1.0)}, {}, ball),
        GameFrame(0.1, True, False, {3: robot(3, 2.0, has_ball=True)}, {7: robot(7, -1.0)}, ball),
    ]
    config = ColumnarReplayWriterConfig(replay_name="m", overwrite_existing=True, checkpoint_every_s=0)
    writer = ColumnarReplayWriter(config, my_team_is_yellow=True, exp_friendly=6, exp_enemy=6, path=tmp_path / "m.npz")
    for frame in frames:
        writer.write_frame(frame)
    writer.close()
    replay = load_columnar_replay(writer.path)

    columns = json.loads(json.dumps(replay_view.columnar_payload(replay)))  # as the page receives it
    for tick, row in enumerate(_frames_from_columns(columns)):
        frame = replay.frame_at(tick)
        assert row == {
            "ts": round(frame.ts, 4),
            "my_team_is_right": bool(frame.my_team_is_right),
            "robots": _serialise_robots(frame),
            "ball": _serialise_ball(frame),
        }
