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
from utama_core.replay import scenario_harvester
from utama_core.replay.bench_scenario import ScenarioFamily, ScenarioTrigger
from utama_core.replay.columnar_writer import (
    ColumnarReplayWriter,
    ColumnarReplayWriterConfig,
)
from utama_core.replay.scenario_harvester import (
    _PASS_LEAD_S,
    OpenPlayEvent,
    find_restart_transitions,
    harvest_replay,
    harvest_run_dir,
    match_is_trustworthy,
    open_play_events,
    pick_events,
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
            {"event": "referee", "sim_time": 0.5, "command": "NORMAL_START"},  # opening kickoff: skipped
            {"event": "referee", "sim_time": 1.0, "command": "STOP"},
            {"event": "referee", "sim_time": 2.0, "command": "PREPARE_KICKOFF_BLUE"},
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
            {"event": "referee", "sim_time": 0.0, "command": "STOP"},
            {"event": "referee", "sim_time": 2.0, "command": "FORCE_START"},
        ],
    )
    _write_stats(trusted_replay, [])

    untrusted_replay = _write_replay(tmp_path, "buggy_match", [0.0, 1.0, 2.0, 3.0])
    _write_sidecar(
        untrusted_replay,
        [
            {"event": "referee", "sim_time": 0.0, "command": "STOP"},
            {"event": "referee", "sim_time": 2.0, "command": "FORCE_START"},
        ],
    )
    _write_stats(untrusted_replay, [{"kind": "RESTART_STALL"}])

    no_stats_replay = _write_replay(tmp_path, "no_stats_match", [0.0, 1.0, 2.0, 3.0])
    _write_sidecar(
        no_stats_replay,
        [
            {"event": "referee", "sim_time": 0.0, "command": "STOP"},
            {"event": "referee", "sim_time": 2.0, "command": "FORCE_START"},
        ],
    )
    # Deliberately no .stats.json written for this one.

    # each match's referee log is not a replay of its own
    (tmp_path / "another_match.sparse_referee.pkl").write_bytes(b"")

    scenarios, report = harvest_run_dir(tmp_path, evaluator_version="abc123")

    assert report["matches_seen"] == 3
    assert report["matches_trusted"] == 1
    assert report["matches_untrusted"] == 2
    assert len(scenarios) == 1
    assert scenarios[0].provenance.source_replay == trusted_replay


def test_finds_free_kick_after_ball_placement(tmp_path):
    """`CustomReferee` places the ball before most free kicks; only STOP -> DIRECT_FREE
    was matched, so a whole round-robin yielded 14 free kicks."""
    sidecar = tmp_path / "match.intentions.jsonl"
    sidecar.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"event": "referee", "sim_time": 10.0, "command": "STOP"},
                {"event": "referee", "sim_time": 11.0, "command": "BALL_PLACEMENT_YELLOW"},
                {"event": "referee", "sim_time": 15.0, "command": "DIRECT_FREE_YELLOW"},
            ]
        )
    )
    transitions = find_restart_transitions(sidecar)
    assert [t.command for t in transitions] == [RefereeCommand.DIRECT_FREE_YELLOW]


def test_harvest_replay_skips_the_opening_kickoff(tmp_path):
    """Every match opens with the same kickoff; from a 231-match run it was 222 of 317
    harvested scenarios. A kickoff after a goal is kept."""
    replay_path = _write_replay(tmp_path, "m", [0.0, 1.0, 2.0, 3.0, 4.0])
    _write_sidecar(
        replay_path,
        [
            {"event": "referee", "sim_time": 0.0, "command": "PREPARE_KICKOFF_YELLOW"},
            {"event": "referee", "sim_time": 1.0, "command": "NORMAL_START"},
            {"event": "referee", "sim_time": 2.0, "command": "PREPARE_KICKOFF_BLUE"},
            {"event": "referee", "sim_time": 3.0, "command": "NORMAL_START"},
        ],
    )
    scenarios = harvest_replay(replay_path, source_run_id="r", evaluator_version="abc")
    assert [s.provenance.anchor_tick for s in scenarios] == [3.0]


# --- open play ----------------------------------------------------------------

_LIVE_FROM_5 = [
    {"event": "referee", "sim_time": 0.0, "command": "PREPARE_KICKOFF_YELLOW"},
    {"event": "referee", "sim_time": 5.0, "command": "NORMAL_START"},
    {"event": "referee", "sim_time": 30.0, "command": "STOP"},
]


def _analysis(passes=(), turnovers=()) -> dict:
    return {"passes": [{"t": t} for t in passes], "turnovers": list(turnovers), "restarts": []}


def _loss(t: float, kind: str = "tackled", regained_after_s=None) -> dict:
    return {"t": t, "kind": kind, "regained_after_s": regained_after_s}


def test_a_pass_gives_a_possession_start_before_the_release():
    events = open_play_events(_analysis(passes=[12.0]), _LIVE_FROM_5)

    assert events == [
        OpenPlayEvent(12.0 - _PASS_LEAD_S, ScenarioFamily.OPEN_PLAY_POSSESSION, "candidate_in_possession")
    ]


def test_a_pass_whose_lead_in_is_not_all_live_play_is_skipped():
    # the start would fall before NORMAL_START (5.0), or span the STOP at 30.0
    assert open_play_events(_analysis(passes=[5.0 + _PASS_LEAD_S - 0.01, 30.5]), _LIVE_FROM_5) == []
    assert len(open_play_events(_analysis(passes=[5.0 + _PASS_LEAD_S]), _LIVE_FROM_5)) == 1


def test_only_real_live_losses_give_counter_starts():
    losses = [
        _loss(10.0),
        _loss(11.0, regained_after_s=0.5),  # flicker: won straight back
        _loss(12.0, kind="during_stoppage"),
        _loss(31.0),  # after the STOP
    ]
    events = open_play_events(_analysis(turnovers=losses), _LIVE_FROM_5)

    assert events == [OpenPlayEvent(10.0, ScenarioFamily.OPEN_PLAY_COUNTER, "candidate_defending")]


def test_pick_events_caps_each_family_and_is_reproducible():
    events = [
        OpenPlayEvent(float(t), ScenarioFamily.OPEN_PLAY_POSSESSION, "candidate_in_possession") for t in range(10)
    ]
    events += [OpenPlayEvent(20.0, ScenarioFamily.OPEN_PLAY_COUNTER, "candidate_defending")]

    picked = pick_events(events, 2, seed="match_a")

    assert len(picked) == 3
    assert sum(e.family == ScenarioFamily.OPEN_PLAY_POSSESSION for e in picked) == 2
    assert picked == pick_events(events, 2, seed="match_a")
    assert [e.sim_time for e in picked] == sorted(e.sim_time for e in picked)


def test_harvest_replay_adds_open_play_scenarios_when_asked(tmp_path, monkeypatch):
    replay = _write_replay(tmp_path, "match", [t / 10 for t in range(0, 200)])
    _write_sidecar(replay, _LIVE_FROM_5)
    monkeypatch.setattr(
        scenario_harvester, "analyse_match", lambda path: _analysis(passes=[12.0], turnovers=[_loss(15.0)])
    )

    assert harvest_replay(replay, source_run_id="run", evaluator_version="x") == []
    scenarios = harvest_replay(replay, source_run_id="run", evaluator_version="x", open_play_per_match=1)

    assert [(s.provenance.family, s.provenance.trigger) for s in scenarios] == [
        (ScenarioFamily.OPEN_PLAY_POSSESSION, ScenarioTrigger.EVENT),
        (ScenarioFamily.OPEN_PLAY_COUNTER, ScenarioTrigger.EVENT),
    ]
    assert scenarios[0].scenario.sim_time == 12.0 - _PASS_LEAD_S
    assert scenarios[0].scenario_id == f"match_t{12.0 - _PASS_LEAD_S:.1f}_open_play_possession"
