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
    monkeypatch.setattr(replay_view, "_ball_losses", lambda path: [t for t in turnovers if replay_view._is_real(t)])
    monkeypatch.setattr(replay_view, "REPLAY_BASE_PATH", tmp_path)

    payload = json.loads(replay_view._frames_bytes({"path": "a_vs_b.pkl"}))

    assert [(m["sim_time"], m["kind"]) for m in payload["markers"]] == [
        (5.0, "loss"),
        (9.5, "foul"),
        (12.0, "goal"),
        (30.0, "stall"),
    ]
    assert "yellow" in payload["markers"][2]["label"] and "1-0" in payload["markers"][2]["label"]
