"""Tests for `utama_core.scenario_bench.scenario_harvester`.

Uses a synthetic `.npz` replay (`ColumnarReplayWriter`, same writer
`round_robin.py` uses) plus a hand-written `.intentions.jsonl` sidecar —
referee-command info lives entirely in the sidecar here (frames carry
`referee=None`), matching `scenario_from_replay`'s documented fallback path,
except in the free-kick tick tests, whose frames carry the command one tick
after the sidecar's time for it, as a real replay's do.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest

from utama_core.entities.data.referee import RefereeData
from utama_core.entities.data.vector import Vector2D, Vector3D
from utama_core.entities.game import Ball, GameFrame, Robot
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.entities.referee.stage import Stage
from utama_core.replay.columnar_writer import (
    ColumnarReplayWriter,
    ColumnarReplayWriterConfig,
)
from utama_core.scenario_bench import scenario_harvester
from utama_core.scenario_bench.hand_authored_scenarios import (
    all_hand_authored_scenarios,
)
from utama_core.scenario_bench.scenario_harvester import (
    RestartTransition,
    _kicking_is_yellow,
    find_restart_transitions,
    harvest_replay,
    harvest_run_dir,
    match_is_trustworthy,
    restart_situation,
    sample_by_situation,
)
from utama_core.scenario_bench.start import ScenarioFamily, ScenarioTrigger


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


def _referee(command: RefereeCommand) -> RefereeData:
    return RefereeData(
        source_identifier=None,
        time_sent=0.0,
        time_received=0.0,
        referee_command=command,
        referee_command_timestamp=0.0,
        stage=Stage.NORMAL_FIRST_HALF,
        stage_time_left=0.0,
        blue_team=None,
        yellow_team=None,
    )


def _frame(ts: float, command: Optional[RefereeCommand] = None, ball_x: float = 0.0) -> GameFrame:
    return GameFrame(
        ts=ts,
        my_team_is_yellow=True,
        my_team_is_right=True,
        friendly_robots={0: _robot(0, -4.0, 0.0, friendly=True), 1: _robot(1, -1.0, 0.5, friendly=True)},
        enemy_robots={0: _robot(0, 4.0, 0.0, friendly=False), 1: _robot(1, 1.0, -0.5, friendly=False)},
        ball=Ball(p=Vector3D(ball_x, 0.0, 0.0), v=Vector3D(0.0, 0.0, 0.0), a=Vector3D(0.0, 0.0, 0.0)),
        referee=_referee(command) if command is not None else None,
    )


def _write_replay(tmp_path, name: str, ts_values: list[float], frame=_frame):
    path = tmp_path / f"{name}.npz"
    writer = ColumnarReplayWriter(
        ColumnarReplayWriterConfig(replay_name=name, overwrite_existing=True),
        my_team_is_yellow=True,
        exp_friendly=2,
        exp_enemy=2,
        path=path,
    )
    for ts in ts_values:
        writer.write_frame(frame(ts))
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


def test_a_kickoff_or_penalty_is_found_where_it_is_awarded_not_where_it_goes_live(tmp_path):
    """The start is the restart's own command, as for a free kick. The NORMAL_START after it
    could only be replayed as live play, the penalty or kickoff already over."""
    sidecar = tmp_path / "match.intentions.jsonl"
    sidecar.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"event": "referee", "sim_time": 0.0, "command": "PREPARE_KICKOFF_YELLOW"},
                {"event": "referee", "sim_time": 3.0, "command": "NORMAL_START"},
                {"event": "referee", "sim_time": 10.0, "command": "STOP"},
                {"event": "referee", "sim_time": 11.0, "command": "BALL_PLACEMENT_YELLOW"},
                {"event": "referee", "sim_time": 13.0, "command": "PREPARE_PENALTY_BLUE"},
                {"event": "referee", "sim_time": 18.0, "command": "NORMAL_START"},
            ]
        )
    )
    transitions = find_restart_transitions(sidecar)
    assert [(t.sim_time, t.command, t.kicking_is_yellow) for t in transitions] == [
        (0.0, RefereeCommand.PREPARE_KICKOFF_YELLOW, True),
        (13.0, RefereeCommand.PREPARE_PENALTY_BLUE, False),
    ]


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


def test_live_play_and_stoppages_alone_produce_no_transition(tmp_path):
    sidecar = tmp_path / "match.intentions.jsonl"
    sidecar.write_text(
        "\n".join(
            json.dumps({"event": "referee", "sim_time": t, "command": c})
            for t, c in [(0.0, "NORMAL_START"), (5.0, "STOP"), (6.0, "BALL_PLACEMENT_BLUE"), (9.0, "NORMAL_START")]
        )
    )
    assert find_restart_transitions(sidecar) == []


def test_missing_sidecar_returns_empty(tmp_path):
    assert find_restart_transitions(tmp_path / "nonexistent.intentions.jsonl") == []


def test_duplicate_transitions_within_gap_are_deduplicated(tmp_path):
    sidecar = tmp_path / "match.intentions.jsonl"
    sidecar.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"event": "referee", "sim_time": 0.0, "command": "STOP"},
                {"event": "referee", "sim_time": 3.0, "command": "DIRECT_FREE_YELLOW"},
                {"event": "referee", "sim_time": 3.05, "command": "DIRECT_FREE_YELLOW"},
                {"event": "referee", "sim_time": 3.1, "command": "STOP"},
                {"event": "referee", "sim_time": 3.2, "command": "DIRECT_FREE_YELLOW"},
            ]
        )
    )
    # The repeated row (a score or stage change logs the same command again) is not a
    # transition at all, and the free kick given again 0.2 s later is within the gap.
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
            {"event": "referee", "sim_time": 0.0, "command": "PREPARE_KICKOFF_YELLOW"},  # opening kickoff: skipped
            {"event": "referee", "sim_time": 0.5, "command": "NORMAL_START"},
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
    assert bench_scenario.provenance.anchor_tick == 2.0
    assert bench_scenario.provenance.family == ScenarioFamily.KICKOFF
    assert bench_scenario.provenance.perspective == "candidate_defending"  # blue kicks off; the candidate is yellow
    assert bench_scenario.scenario.referee_command == RefereeCommand.PREPARE_KICKOFF_BLUE
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
    assert [s.provenance.anchor_tick for s in scenarios] == [2.0]


_TICK = 1 / 60


def _placement_then_free_kick(ts: float) -> GameFrame:
    """A replay frame shows the referee's decision on the frame before it: the match log puts
    DIRECT_FREE_YELLOW at t=15.0, the replay first shows it one tick later. The ball moves
    1 m a tick so the test can tell which tick a start was taken from."""
    awarded = ts > 15.0 + _TICK / 2
    command = RefereeCommand.DIRECT_FREE_YELLOW if awarded else RefereeCommand.BALL_PLACEMENT_YELLOW
    return _frame(ts, command, ball_x=round((ts - 15.0) / _TICK))


def test_a_free_kick_start_is_the_first_replay_tick_that_shows_the_free_kick(tmp_path):
    """The nearest tick to the match log's time still showed BALL_PLACEMENT_*: replayed, the
    free kick sat in ball placement and was never taken (bank_v7, 241 starts)."""
    ticks = [15.0 + k * _TICK for k in range(-2, 4)]
    replay_path = _write_replay(tmp_path, "m", ticks, _placement_then_free_kick)
    _write_sidecar(
        replay_path,
        [
            {"event": "referee", "sim_time": 10.0, "command": "STOP"},
            {"event": "referee", "sim_time": 11.0, "command": "BALL_PLACEMENT_YELLOW"},
            {"event": "referee", "sim_time": 15.0, "command": "DIRECT_FREE_YELLOW"},
        ],
    )

    [start] = harvest_replay(replay_path, source_run_id="r", evaluator_version="abc")

    assert start.scenario.referee_command == RefereeCommand.DIRECT_FREE_YELLOW
    assert start.scenario.frame_ts == pytest.approx(15.0 + _TICK, abs=1e-6)
    assert start.scenario.ball_x == pytest.approx(1.0)
    assert start.provenance.anchor_tick == 15.0


def test_a_start_whose_command_the_replay_never_shows_is_dropped(tmp_path):
    ticks = [15.0 + k * _TICK for k in range(-2, 4)]
    replay_path = _write_replay(tmp_path, "m", ticks, lambda ts: _frame(ts, RefereeCommand.STOP))
    _write_sidecar(
        replay_path,
        [
            {"event": "referee", "sim_time": 10.0, "command": "STOP"},
            {"event": "referee", "sim_time": 15.0, "command": "DIRECT_FREE_YELLOW"},
        ],
    )

    assert harvest_replay(replay_path, source_run_id="r", evaluator_version="abc") == []


# --- situation types and the per-situation sample --------------------------------


@pytest.mark.parametrize(
    "command, ball_x, situation",
    [
        # the candidate is yellow, defending the right goal: it attacks -x
        (RefereeCommand.DIRECT_FREE_YELLOW, -3.0, "own free kick, attacking third"),
        (RefereeCommand.DIRECT_FREE_YELLOW, 3.0, "own free kick, defensive third"),
        (RefereeCommand.DIRECT_FREE_BLUE, 3.0, "opponent free kick, attacking third"),
        (RefereeCommand.DIRECT_FREE_BLUE, 0.0, "opponent free kick, middle third"),
        (RefereeCommand.PREPARE_KICKOFF_BLUE, 0.0, "opponent kickoff"),
        (RefereeCommand.PREPARE_PENALTY_YELLOW, -3.0, "own penalty"),
        (RefereeCommand.FORCE_START, 3.0, "force start"),
    ],
)
def test_a_restart_start_is_typed_from_the_candidate_s_view(command, ball_x, situation):
    transition = RestartTransition(10.0, RefereeCommand.STOP, command, _kicking_is_yellow(command))
    assert restart_situation(transition, True, SimpleNamespace(ball_x=ball_x)) == situation


def test_a_harvested_restart_carries_its_situation(tmp_path):
    ticks = [15.0 + k * _TICK for k in range(-2, 4)]
    replay_path = _write_replay(tmp_path, "m", ticks, _placement_then_free_kick)
    _write_sidecar(
        replay_path,
        [
            {"event": "referee", "sim_time": 11.0, "command": "BALL_PLACEMENT_YELLOW"},
            {"event": "referee", "sim_time": 15.0, "command": "DIRECT_FREE_YELLOW"},
        ],
    )

    [start] = harvest_replay(replay_path, source_run_id="r", evaluator_version="abc")

    assert start.scenario.ball_x == pytest.approx(1.0)  # 3.5 m from the goal the candidate attacks
    assert start.provenance.situation == "own free kick, middle third"


def test_a_ball_won_start_is_each_ledger_spell_not_taking_a_restart_in_live_play(tmp_path, monkeypatch):
    replay = _write_replay(tmp_path, "m", [t / 10 for t in range(60)])
    _write_sidecar(
        replay,
        [
            {"event": "referee", "sim_time": 0.0, "command": "PREPARE_KICKOFF_YELLOW"},
            {"event": "referee", "sim_time": 1.0, "command": "NORMAL_START"},
        ],
    )

    def spell(t, side, third, takes_restart=False):
        return {"t": t, "t_end": t + 0.5, "side": side, "third": third, "takes_restart": takes_restart}

    spells = [
        spell(0.5, "friendly", "middle"),  # before play went live
        spell(1.2, "friendly", "middle", takes_restart=True),  # the kickoff being taken
        spell(2.0, "friendly", "attacking"),
        spell(3.0, "enemy", "defensive"),
    ]
    monkeypatch.setattr(scenario_harvester, "match_ledger", lambda path: {"spells": spells})

    assert harvest_replay(replay, source_run_id="r", evaluator_version="x") == []  # the opening kickoff
    starts = harvest_replay(replay, source_run_id="r", evaluator_version="x", ball_won=True)

    assert [(s.provenance.anchor_tick, s.provenance.situation, s.provenance.family) for s in starts] == [
        (2.0, "own ball won, attacking third", ScenarioFamily.OPEN_PLAY_POSSESSION),
        (3.0, "opponent ball won, defensive third", ScenarioFamily.OPEN_PLAY_COUNTER),
    ]
    assert all(s.provenance.trigger == ScenarioTrigger.EVENT for s in starts)
    assert starts[0].scenario_id == "m_t2.0_own_ball_won_attacking"


def _typed(situation: str, match: str, k: int):
    base = list(all_hand_authored_scenarios())[0]
    return dataclasses.replace(
        base,
        scenario_id=f"{match}_{situation}_{k}",
        # 1 m apart: no two are near-duplicates
        scenario=dataclasses.replace(base.scenario, ball_x=-4.0 + k, ball_y=-2.0 + ord(match[-1]) - ord("a")),
        provenance=dataclasses.replace(base.provenance, situation=situation, source_replay=Path(f"{match}.npz")),
    )


def test_the_sample_keeps_up_to_n_of_each_situation_spread_over_matches_and_half_as_many_kickoffs():
    starts = [
        _typed(situation, match, k)
        for situation in ("own ball won, middle third", "own kickoff")
        for match in ("match_a", "match_b", "match_c")
        for k in range(5)
    ]

    sample = sample_by_situation(starts, 4, seed=0)

    won = [s for s in sample if s.provenance.situation == "own ball won, middle third"]
    assert len(won) == 4
    assert {s.provenance.source_replay for s in won} == {Path(f"match_{m}.npz") for m in "abc"}
    assert len([s for s in sample if s.provenance.situation == "own kickoff"]) == 2
    assert [s.scenario_id for s in sample_by_situation(starts, 4, seed=0)] == [s.scenario_id for s in sample]
    assert [s.scenario_id for s in sample_by_situation(starts, 4, seed=1)] != [s.scenario_id for s in sample]


def test_the_sample_skips_a_near_duplicate_of_a_start_already_drawn():
    first = _typed("own kickoff", "match_a", 0)
    again = dataclasses.replace(
        first, scenario_id="again", provenance=dataclasses.replace(first.provenance, source_replay=Path("match_b.npz"))
    )

    assert len(sample_by_situation([first, again], 40)) == 1
