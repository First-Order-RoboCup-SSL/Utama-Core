"""Tests for `utama_core.kernel.match_log.MatchLog` — the structured intention trace."""

from __future__ import annotations

import json

from utama_core.kernel.match_log import MatchLog
from utama_core.kernel.tactic import TacticTag


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
