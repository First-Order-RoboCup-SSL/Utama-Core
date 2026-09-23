"""Tests for `utama_core.replay.scenario_harvester`.

Uses a synthetic `.npz` replay (`ColumnarReplayWriter`, same writer
`smoke_tournament.py` uses) plus a hand-written `.intentions.jsonl` sidecar —
referee-command info lives entirely in the sidecar here (frames carry
`referee=None`), matching `scenario_from_replay`'s documented fallback path,
so no real `RefereeData`/`TeamInfo`/`Stage` construction is needed to test
the harvester's transition-detection and trust-gate logic.
"""

from __future__ import annotations

import json

from utama_core.entities.data.vector import Vector2D, Vector3D
from utama_core.entities.game import Ball, GameFrame, Robot
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.replay.columnar_writer import (
    ColumnarReplayWriter,
    ColumnarReplayWriterConfig,
)
from utama_core.replay.scenario_harvester import (
    find_restart_transitions,
    harvest_replay,
    harvest_run_dir,
    match_is_trustworthy,
)


def _robot(id_: int, x: float, y: float, *, friendly: bool) -> Robot:
    return Robot(
        id=id_,
        is_friendly=friendly,
        has_ball=False,
        p=Vector2D(x, y),
        v=Vector2D(0, 0),
        a=Vector2D(0, 0),
        orientation=0.0,
    )


def _frame(ts: float) -> GameFrame:
    return GameFrame(
        ts=ts,
        my_team_is_yellow=True,
        my_team_is_right=True,
        friendly_robots={0: _robot(0, -4.0, 0.0, friendly=True), 1: _robot(1, -1.0, 0.5, friendly=True)},
        enemy_robots={0: _robot(0, 4.0, 0.0, friendly=False), 1: _robot(1, 1.0, -0.5, friendly=False)},
        ball=Ball(p=Vector3D(0.0, 0.0, 0.0), v=Vector3D(0.0, 0.0, 0.0), a=Vector3D(0.0, 0.0, 0.0)),
    )


def _write_replay(tmp_path, name: str, ts_values: list[float]):
    path = tmp_path / f"{name}.npz"
    writer = ColumnarReplayWriter(
        ColumnarReplayWriterConfig(replay_name=name, overwrite_existing=True),
        my_team_is_yellow=True,
        exp_friendly=2,
        exp_enemy=2,
        path=path,
    )
    for ts in ts_values:
        writer.write_frame(_frame(ts))
    writer.close()
    return path


def _write_sidecar(replay_path, rows: list[dict]) -> None:
    sidecar_path = replay_path.with_name(f"{replay_path.stem}.intentions.jsonl")
    with open(sidecar_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _write_stats(replay_path, stall_events: list) -> None:
    stats_path = replay_path.with_name(f"{replay_path.stem}.stats.json")
    stats_path.write_text(json.dumps({"stall_events": stall_events}))


# --- find_restart_transitions -------------------------------------------------


def test_finds_kickoff_transition(tmp_path):
    sidecar = tmp_path / "match.intentions.jsonl"
    sidecar.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"event": "referee", "sim_time": 0.0, "command": "PREPARE_KICKOFF_YELLOW"},
                {"event": "referee", "sim_time": 3.0, "command": "NORMAL_START"},
            ]
        )
    )
    transitions = find_restart_transitions(sidecar)
    assert len(transitions) == 1
    assert transitions[0].sim_time == 3.0
    assert transitions[0].command == RefereeCommand.NORMAL_START


def test_finds_direct_free_transition(tmp_path):
    sidecar = tmp_path / "match.intentions.jsonl"
    sidecar.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"event": "referee", "sim_time": 10.0, "command": "STOP"},
                {"event": "referee", "sim_time": 15.0, "command": "DIRECT_FREE_BLUE"},
            ]
        )
    )
    transitions = find_restart_transitions(sidecar)
    assert len(transitions) == 1
    assert transitions[0].command == RefereeCommand.DIRECT_FREE_BLUE
    assert transitions[0].kicking_is_yellow is False


def test_prepare_states_alone_produce_no_transition(tmp_path):
    sidecar = tmp_path / "match.intentions.jsonl"
    sidecar.write_text(json.dumps({"event": "referee", "sim_time": 0.0, "command": "PREPARE_KICKOFF_YELLOW"}))
    assert find_restart_transitions(sidecar) == []


def test_missing_sidecar_returns_empty(tmp_path):
    assert find_restart_transitions(tmp_path / "nonexistent.intentions.jsonl") == []


def test_duplicate_transitions_within_gap_are_deduplicated(tmp_path):
    sidecar = tmp_path / "match.intentions.jsonl"
    sidecar.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"event": "referee", "sim_time": 0.0, "command": "PREPARE_KICKOFF_YELLOW"},
                {"event": "referee", "sim_time": 3.0, "command": "NORMAL_START"},
                {"event": "referee", "sim_time": 3.05, "command": "NORMAL_START"},
            ]
        )
    )
    # Second NORMAL_START row has the same prev_command as itself (NORMAL_START
    # -> NORMAL_START is not a transition at all, so this also checks that
    # re-emitting the same command isn't treated as a new restart).
    transitions = find_restart_transitions(sidecar)
    assert len(transitions) == 1


# --- match_is_trustworthy -----------------------------------------------------


def test_trustworthy_when_stats_file_has_zero_stalls(tmp_path):
    stats_path = tmp_path / "match.stats.json"
    stats_path.write_text(json.dumps({"stall_events": []}))
    assert match_is_trustworthy(stats_path) is True


def test_untrustworthy_when_stats_file_has_stalls(tmp_path):
    stats_path = tmp_path / "match.stats.json"
    stats_path.write_text(json.dumps({"stall_events": [{"kind": "RESTART_STALL"}]}))
    assert match_is_trustworthy(stats_path) is False


def test_untrustworthy_when_stats_file_missing(tmp_path):
    assert match_is_trustworthy(tmp_path / "nonexistent.stats.json") is False


def test_untrustworthy_when_stats_file_unparseable(tmp_path):
    stats_path = tmp_path / "match.stats.json"
    stats_path.write_text("not json")
    assert match_is_trustworthy(stats_path) is False


# --- harvest_replay / harvest_run_dir (synthetic .npz + sidecar) --------------


def test_harvest_replay_extracts_scenario_at_transition(tmp_path):
    replay_path = _write_replay(tmp_path, "clean_match", [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    _write_sidecar(
        replay_path,
        [
            {"event": "referee", "sim_time": 0.0, "command": "PREPARE_KICKOFF_YELLOW"},
            {"event": "referee", "sim_time": 3.0, "command": "NORMAL_START"},
        ],
    )

    scenarios = harvest_replay(replay_path, source_run_id="test_run", evaluator_version="abc123")

    assert len(scenarios) == 1
    bench_scenario = scenarios[0]
    assert bench_scenario.provenance.source_run_id == "test_run"
    assert bench_scenario.provenance.evaluator_version == "abc123"
    assert bench_scenario.provenance.anchor_tick == 3.0
    assert bench_scenario.scenario.friendly_robots
    assert bench_scenario.scenario.enemy_robots


def test_harvest_run_dir_skips_untrusted_matches(tmp_path):
    trusted_replay = _write_replay(tmp_path, "clean_match", [0.0, 1.0, 2.0, 3.0])
    _write_sidecar(
        trusted_replay,
        [
            {"event": "referee", "sim_time": 0.0, "command": "PREPARE_KICKOFF_YELLOW"},
            {"event": "referee", "sim_time": 2.0, "command": "NORMAL_START"},
        ],
    )
    _write_stats(trusted_replay, [])

    untrusted_replay = _write_replay(tmp_path, "buggy_match", [0.0, 1.0, 2.0, 3.0])
    _write_sidecar(
        untrusted_replay,
        [
            {"event": "referee", "sim_time": 0.0, "command": "PREPARE_KICKOFF_YELLOW"},
            {"event": "referee", "sim_time": 2.0, "command": "NORMAL_START"},
        ],
    )
    _write_stats(untrusted_replay, [{"kind": "RESTART_STALL"}])

    no_stats_replay = _write_replay(tmp_path, "no_stats_match", [0.0, 1.0, 2.0, 3.0])
    _write_sidecar(
        no_stats_replay,
        [
            {"event": "referee", "sim_time": 0.0, "command": "PREPARE_KICKOFF_YELLOW"},
            {"event": "referee", "sim_time": 2.0, "command": "NORMAL_START"},
        ],
    )
    # Deliberately no .stats.json written for this one.

    scenarios, report = harvest_run_dir(tmp_path, evaluator_version="abc123")

    assert report["matches_seen"] == 3
    assert report["matches_trusted"] == 1
    assert report["matches_untrusted"] == 2
    assert len(scenarios) == 1
    assert scenarios[0].provenance.source_replay == trusted_replay
