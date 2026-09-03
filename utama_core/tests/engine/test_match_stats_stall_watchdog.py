"""Tests for the in-match stall watchdog in `utama_core.engine.match_stats`.

Feeds `MatchStatsAccumulator.record_tick()` a synthetic sequence of
`GameFrame`s (ball position/velocity, referee command, sim `ts`) and checks
`MatchStats.stall_events` — see that module's `StallEvent`/
`_maybe_record_stalls` for the two watchdogs under test:

- RESTART_STALL: a non-live referee command (STOP/DIRECT_FREE_*/
  PREPARE_KICKOFF_*/BALL_PLACEMENT_*/...) held continuously for more than
  `_RESTART_STALL_SECONDS` (15s) without advancing back to
  NORMAL_START/FORCE_START.
- COMMITTED_FROZEN: the ball not moving more than `_STALL_BALL_STILL_TOL_M`
  (5cm) for more than `_COMMITTED_FROZEN_SECONDS` (10s) during live play
  (NORMAL_START/FORCE_START) while at least one tactic slot is committed
  (or, when slot-commitment info isn't supplied, during live play at all —
  the documented fallback).
"""

from __future__ import annotations

import pytest

from utama_core.engine.match_stats import MatchStatsAccumulator
from utama_core.entities.data.referee import RefereeData
from utama_core.entities.data.vector import Vector2D, Vector3D
from utama_core.entities.game.ball import Ball
from utama_core.entities.game.game_frame import GameFrame
from utama_core.entities.game.robot import Robot
from utama_core.entities.game.team_info import TeamInfo
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.entities.referee.stage import Stage

TICK_DT = 1.0 / 60.0


def _referee(command: RefereeCommand, ts: float) -> RefereeData:
    return RefereeData(
        source_identifier=None,
        time_sent=ts,
        time_received=ts,
        referee_command=command,
        referee_command_timestamp=ts,
        stage=Stage.NORMAL_FIRST_HALF,
        stage_time_left=0.0,
        blue_team=TeamInfo(name="blue"),
        yellow_team=TeamInfo(name="yellow"),
    )


def _frame(ts: float, command: RefereeCommand, ball_xy=(0.0, 0.0)) -> GameFrame:
    friendly = {1: Robot(id=1, is_friendly=True, has_ball=False, p=Vector2D(0.0, 0.0), v=None, a=None, orientation=0)}
    enemy = {2: Robot(id=2, is_friendly=False, has_ball=False, p=Vector2D(4.0, 0.0), v=None, a=None, orientation=0)}
    ball = Ball(Vector3D(ball_xy[0], ball_xy[1], 0), Vector3D(0, 0, 0), None)
    return GameFrame(
        ts=ts,
        my_team_is_yellow=True,
        my_team_is_right=True,
        friendly_robots=friendly,
        enemy_robots=enemy,
        ball=ball,
        referee=_referee(command, ts),
    )


def _run_ticks(acc: MatchStatsAccumulator, n_ticks: int, command, ball_xy=(0.0, 0.0), committed_tactics=None):
    """Feed `n_ticks` frames 1/60s apart at a fixed command/ball position."""
    for i in range(n_ticks):
        ts = (i + 1) * TICK_DT
        acc.record_tick(_frame(ts, command, ball_xy), committed_tactics=committed_tactics)


# ---------------------------------------------------------------------------
# RESTART_STALL
# ---------------------------------------------------------------------------


def test_restart_persisting_20s_produces_restart_stall():
    acc = MatchStatsAccumulator()
    _run_ticks(acc, n_ticks=20 * 60, command=RefereeCommand.DIRECT_FREE_BLUE)

    stats = acc.finalize()
    restart_events = [e for e in stats.stall_events if e.kind == "RESTART_STALL"]
    assert len(restart_events) == 1
    event = restart_events[0]
    assert event.referee_command == "DIRECT_FREE_BLUE"
    assert event.sim_time == pytest.approx(15.0, abs=2 * TICK_DT)
    assert event.duration_s >= 5.0  # kept updating past onset, up to ~20s - 15s


def test_restart_advancing_within_5s_does_not_stall():
    acc = MatchStatsAccumulator()
    # 5s stuck in a restart, then back to live play for the rest.
    _run_ticks(acc, n_ticks=5 * 60, command=RefereeCommand.DIRECT_FREE_BLUE)
    _run_ticks(acc, n_ticks=5 * 60, command=RefereeCommand.NORMAL_START)

    stats = acc.finalize()
    assert [e for e in stats.stall_events if e.kind == "RESTART_STALL"] == []


def test_restart_stall_produces_one_event_not_one_per_tick():
    acc = MatchStatsAccumulator()
    _run_ticks(acc, n_ticks=25 * 60, command=RefereeCommand.PREPARE_KICKOFF_YELLOW)

    stats = acc.finalize()
    restart_events = [e for e in stats.stall_events if e.kind == "RESTART_STALL"]
    assert len(restart_events) == 1


# ---------------------------------------------------------------------------
# COMMITTED_FROZEN
# ---------------------------------------------------------------------------


def test_frozen_ball_with_committed_slot_for_12s_produces_committed_frozen():
    acc = MatchStatsAccumulator()
    _run_ticks(
        acc,
        n_ticks=12 * 60,
        command=RefereeCommand.NORMAL_START,
        ball_xy=(0.0, 0.0),
        committed_tactics={"give_and_go": (1,)},
    )

    stats = acc.finalize()
    frozen_events = [e for e in stats.stall_events if e.kind == "COMMITTED_FROZEN"]
    assert len(frozen_events) == 1
    event = frozen_events[0]
    assert event.referee_command == "NORMAL_START"
    assert event.tactic_ids == ("give_and_go",)
    assert event.robot_ids == (1,)
    assert event.sim_time == pytest.approx(10.0, abs=2 * TICK_DT)


def test_frozen_ball_during_stop_does_not_stall():
    acc = MatchStatsAccumulator()
    _run_ticks(
        acc,
        n_ticks=12 * 60,
        command=RefereeCommand.STOP,
        ball_xy=(0.0, 0.0),
        committed_tactics={"give_and_go": (1,)},
    )

    stats = acc.finalize()
    assert [e for e in stats.stall_events if e.kind == "COMMITTED_FROZEN"] == []
    # STOP is a non-live command held the whole 12s, well under the 15s
    # RESTART_STALL threshold -- neither watchdog should fire.
    assert stats.stall_events == []


def test_frozen_ball_without_any_committed_slot_does_not_stall():
    """A frozen ball during live play with committed_tactics supplied but
    empty (nothing committed) is not the bug this watchdog targets."""
    acc = MatchStatsAccumulator()
    _run_ticks(
        acc,
        n_ticks=12 * 60,
        command=RefereeCommand.NORMAL_START,
        ball_xy=(0.0, 0.0),
        committed_tactics={},
    )

    stats = acc.finalize()
    assert [e for e in stats.stall_events if e.kind == "COMMITTED_FROZEN"] == []


def test_frozen_ball_falls_back_to_live_play_only_when_commitment_unknown():
    """When `committed_tactics` isn't supplied at all (None -- the documented
    fallback for a caller without cheap access to slot-commitment info), a
    frozen ball during live play alone is enough to flag COMMITTED_FROZEN,
    with empty tactic/robot ids."""
    acc = MatchStatsAccumulator()
    _run_ticks(
        acc,
        n_ticks=12 * 60,
        command=RefereeCommand.NORMAL_START,
        ball_xy=(0.0, 0.0),
        committed_tactics=None,
    )

    stats = acc.finalize()
    frozen_events = [e for e in stats.stall_events if e.kind == "COMMITTED_FROZEN"]
    assert len(frozen_events) == 1
    assert frozen_events[0].tactic_ids == ()
    assert frozen_events[0].robot_ids == ()


def test_ball_moving_more_than_5cm_resets_the_frozen_clock():
    acc = MatchStatsAccumulator()
    # 9s frozen, then a >5cm move, then 9s frozen again -- neither span alone
    # crosses the 10s threshold, so no stall should be recorded.
    _run_ticks(
        acc, n_ticks=9 * 60, command=RefereeCommand.NORMAL_START, ball_xy=(0.0, 0.0), committed_tactics={"t": (1,)}
    )
    _run_ticks(acc, n_ticks=1, command=RefereeCommand.NORMAL_START, ball_xy=(0.5, 0.0), committed_tactics={"t": (1,)})
    _run_ticks(
        acc, n_ticks=9 * 60, command=RefereeCommand.NORMAL_START, ball_xy=(0.5, 0.0), committed_tactics={"t": (1,)}
    )

    stats = acc.finalize()
    assert [e for e in stats.stall_events if e.kind == "COMMITTED_FROZEN"] == []


def test_to_json_serializes_stall_events(tmp_path):
    acc = MatchStatsAccumulator()
    _run_ticks(
        acc,
        n_ticks=11 * 60,
        command=RefereeCommand.NORMAL_START,
        ball_xy=(0.0, 0.0),
        committed_tactics={"give_and_go": (1, 2)},
    )

    out = tmp_path / "stats.json"
    acc.finalize().to_json(out)

    import json

    data = json.loads(out.read_text())
    assert len(data["stall_events"]) == 1
    event = data["stall_events"][0]
    assert event["kind"] == "COMMITTED_FROZEN"
    assert event["tactic_ids"] == ["give_and_go"]
    assert event["robot_ids"] == [1, 2]
