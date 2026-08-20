"""Tests for `utama_core.engine.match_log.MatchLog` — the structured intention/trace log."""

from __future__ import annotations

import json

from utama_core.engine.match_log import IntentionEvent, MatchLog, TraceEvent, load_jsonl
from utama_core.engine.tactic import TacticTag


def test_to_jsonl_round_trips(tmp_path):
    log = MatchLog()
    log.intention(tick=1, sim_time=0.0, tactic_id="pass_and_shoot", robot_ids=(3, 1, 2), tag=TacticTag.ATTACK)
    log.intention(tick=42, sim_time=0.7, tactic_id="defense", robot_ids=(4,), tag=TacticTag.DEFENSE, note="fallback")

    out = tmp_path / "match.jsonl"
    log.to_jsonl(out)

    lines = out.read_text().strip().split("\n")
    assert len(lines) == 2

    row0 = json.loads(lines[0])
    assert row0 == {
        "event": "intention",
        "tick": 1,
        "sim_time": 0.0,
        "tactic_id": "pass_and_shoot",
        "robot_ids": [1, 2, 3],  # sorted, not insertion order
        "tag": "attack",
        "note": None,
    }

    row1 = json.loads(lines[1])
    assert row1["tactic_id"] == "defense"
    assert row1["note"] == "fallback"
    assert row1["tag"] == "defense"


def test_events_returns_recorded_events_in_order():
    log = MatchLog()
    log.intention(tick=1, sim_time=0.0, tactic_id="a", robot_ids=(1,), tag=TacticTag.MIXED)
    log.intention(tick=2, sim_time=0.1, tactic_id="b", robot_ids=(2,), tag=TacticTag.MIXED)

    events = log.events()
    assert [e.tactic_id for e in events] == ["a", "b"]


def test_trace_round_trips_alongside_intentions(tmp_path):
    log = MatchLog()
    log.intention(tick=1, sim_time=0.0, tactic_id="a", robot_ids=(1,), tag=TacticTag.MIXED)
    log.trace(tick=1, sim_time=0.0, key="go_to_ball[1].approach", value="shield")
    log.trace(tick=2, sim_time=0.1, key="has_ball", value=True)

    out = tmp_path / "match.jsonl"
    log.to_jsonl(out)

    events = load_jsonl(out)
    assert [type(e) for e in events] == [IntentionEvent, TraceEvent, TraceEvent]
    assert events[1] == TraceEvent(tick=1, sim_time=0.0, key="go_to_ball[1].approach", value="shield")
    assert events[2] == TraceEvent(tick=2, sim_time=0.1, key="has_ball", value=True)


def test_load_jsonl_reads_intention_rows_without_event_key(tmp_path):
    """Replays written before `event` existed have no such key — must still load as intentions."""
    out = tmp_path / "old.jsonl"
    out.write_text(
        json.dumps(
            {
                "tick": 1,
                "sim_time": 0.0,
                "tactic_id": "a",
                "robot_ids": [1],
                "tag": "mixed",
                "note": None,
            }
        )
        + "\n"
    )

    events = load_jsonl(out)
    assert events == [IntentionEvent(tick=1, sim_time=0.0, tactic_id="a", robot_ids=(1,), tag=TacticTag.MIXED)]
