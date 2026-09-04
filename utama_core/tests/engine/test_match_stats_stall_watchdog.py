"""Tests for the in-match stall watchdog in `utama_core.engine.match_stats`.

Feeds `MatchStatsAccumulator.record_tick()` a synthetic sequence of
`GameFrame`s (ball position/velocity, referee command, sim `ts`) and checks
`MatchStats.stall_events` — see that module's `StallEvent`/
`_maybe_record_stalls`/`_maybe_record_no_progress_possession` for the three
watchdogs under test:

- RESTART_STALL: a non-live referee command (STOP/DIRECT_FREE_*/
  PREPARE_KICKOFF_*/BALL_PLACEMENT_*/...) held continuously for more than
  `_RESTART_STALL_SECONDS` (15s) without advancing back to
  NORMAL_START/FORCE_START.
- COMMITTED_FROZEN: the ball not moving more than `_STALL_BALL_STILL_TOL_M`
  (5cm) for more than `_COMMITTED_FROZEN_SECONDS` (10s) during live play
  (NORMAL_START/FORCE_START) while at least one tactic slot is committed
  (or, when slot-commitment info isn't supplied, during live play at all —
  the documented fallback).
- NO_PROGRESS_POSSESSION: the same single robot stays the nearest-to-ball
  robot, within `_NO_PROGRESS_RADIUS_M` (0.35m -- wider than the true
  possession radius, since the failure mode is a robot repeatedly bumping
  the ball a bit farther than true possession range and re-chasing it, not
  one sitting exactly on top of it), for more than `_NO_PROGRESS_SECONDS`
  (8s) of live play without that robot's side ever actually registering
  possession (`_poss_side`/`_poss_robot_id`) of it. Unlike COMMITTED_FROZEN,
  this does not require the ball to be still -- it's built to catch exactly
  the case COMMITTED_FROZEN misses: a robot that keeps bumping the ball a
  few centimetres and turning away, never latching possession, which keeps
  the ball moving just enough each bump to dodge COMMITTED_FROZEN's 5cm
  stillness tolerance. See `_maybe_record_no_progress_possession`'s
  docstring for the live match (shadow_switch_vs_zone_fluid, 2026-09-04)
  this was built and validated against.
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


def _custom_frame(
    ts: float,
    command,
    ball_xy=(0.0, 0.0),
    friendly_xy=(0.0, 0.0),
    friendly_has_ball=False,
    enemy_xy=(4.0, 0.0),
    enemy_has_ball=False,
) -> GameFrame:
    """Like `_frame`, but lets a caller position each robot independently and
    control `has_ball` per robot -- needed for NO_PROGRESS_POSSESSION tests,
    which must place a robot near-but-not-on the ball and simulate an actual
    pickup (`has_ball=True`) partway through a run."""
    friendly = {
        1: Robot(
            id=1, is_friendly=True, has_ball=friendly_has_ball, p=Vector2D(*friendly_xy), v=None, a=None, orientation=0
        )
    }
    enemy = {
        2: Robot(id=2, is_friendly=False, has_ball=enemy_has_ball, p=Vector2D(*enemy_xy), v=None, a=None, orientation=0)
    }
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


# ---------------------------------------------------------------------------
# NO_PROGRESS_POSSESSION
# ---------------------------------------------------------------------------
#
# `_maybe_record_no_progress_possession` gates "actually in control" via
# `_poss_side`/`_poss_robot_id` (the possession-radius state machine driven
# by `_update_possession_events`: within `_POSSESSION_RADIUS_M` (2*0.09+0.05
# = 0.23m) with the ball slower than `_PASS_RELEASE_MPS`), NOT via
# `Robot.has_ball` -- that field is never read by this watchdog. So "the
# robot stays near the ball without gaining control" is modeled by keeping
# the robot within `_NO_PROGRESS_RADIUS_M` (0.35m) but *outside*
# `_POSSESSION_RADIUS_M` (0.23m); "an actual pickup" is modeled by bringing
# the robot inside 0.23m so the possession state machine latches onto it,
# which is what resets this watchdog's clock (`controlled_by_holder` in the
# implementation).
_NO_PROGRESS_BUMP_XY = (0.30, 0.0)  # in (0.23, 0.35): "near", never "controlling"
_NO_PROGRESS_BUMP_XY_JITTER = (0.28, 0.02)  # a second near point, still in that band
_NO_PROGRESS_PICKUP_XY = (0.10, 0.0)  # inside 0.23m: a real pickup


def _run_no_progress_ticks(
    acc: MatchStatsAccumulator,
    n_ticks: int,
    command,
    start_ts: float = 0.0,
    ball_xy=_NO_PROGRESS_BUMP_XY,
    friendly_xy=(0.0, 0.0),
    enemy_xy=(4.0, 0.0),
):
    """Feed `n_ticks` frames 1/60s apart, alternating the ball between two
    close-together points within the no-progress band (a "ball-sized jitter"
    -- a robot bumping the ball a few centimetres, never a stationary ball)
    so the scenario isn't accidentally indistinguishable from a perfectly
    frozen ball."""
    for i in range(n_ticks):
        ts = start_ts + (i + 1) * TICK_DT
        xy = _NO_PROGRESS_BUMP_XY_JITTER if (i % 2 == 0) else ball_xy
        acc.record_tick(_custom_frame(ts, command, ball_xy=xy, friendly_xy=friendly_xy, enemy_xy=enemy_xy))


def test_robot_bumping_ball_for_9s_without_control_produces_no_progress_event():
    acc = MatchStatsAccumulator()
    _run_no_progress_ticks(acc, n_ticks=9 * 60, command=RefereeCommand.NORMAL_START)

    stats = acc.finalize()
    events = [e for e in stats.stall_events if e.kind == "NO_PROGRESS_POSSESSION"]
    assert len(events) == 1
    event = events[0]
    assert event.referee_command == "NORMAL_START"
    assert event.robot_ids == (1,)
    assert event.sim_time == pytest.approx(8.0, abs=2 * TICK_DT)
    assert event.duration_s >= 1.0  # kept updating past onset, up to ~9s - 8s


def test_no_progress_event_produces_one_event_not_one_per_tick():
    acc = MatchStatsAccumulator()
    _run_no_progress_ticks(acc, n_ticks=12 * 60, command=RefereeCommand.NORMAL_START)

    stats = acc.finalize()
    events = [e for e in stats.stall_events if e.kind == "NO_PROGRESS_POSSESSION"]
    assert len(events) == 1
    assert events[0].duration_s >= 3.0


def test_robot_actually_gaining_possession_cancels_no_progress():
    """The robot bumps the ball for 5s (no control), then actually picks it
    up (within true possession radius, ball slow) for the rest of the
    9s+ run -- the state machine latches possession, which must reset the
    no-progress clock so it never crosses the threshold."""
    acc = MatchStatsAccumulator()
    _run_no_progress_ticks(acc, n_ticks=5 * 60, command=RefereeCommand.NORMAL_START)
    # Now actually gains control: inside _POSSESSION_RADIUS_M, ball slow
    # (already 0 velocity in _custom_frame).
    for i in range(6 * 60):
        ts = 5.0 + (i + 1) * TICK_DT
        acc.record_tick(
            _custom_frame(ts, RefereeCommand.NORMAL_START, ball_xy=_NO_PROGRESS_PICKUP_XY, friendly_xy=(0.0, 0.0))
        )

    stats = acc.finalize()
    assert [e for e in stats.stall_events if e.kind == "NO_PROGRESS_POSSESSION"] == []


def test_no_progress_does_not_fire_outside_live_play():
    acc = MatchStatsAccumulator()
    _run_no_progress_ticks(acc, n_ticks=12 * 60, command=RefereeCommand.STOP)

    stats = acc.finalize()
    assert [e for e in stats.stall_events if e.kind == "NO_PROGRESS_POSSESSION"] == []


def test_no_progress_gap_in_the_middle_resets_the_clock():
    """The robot bumps the ball for 5s, the ball rolls out of
    `_NO_PROGRESS_RADIUS_M` briefly, then comes back and the robot resumes
    bumping it for another 5s. Total "near" time is 10s (> the 8s
    threshold), but per the implementation (`_maybe_record_no_progress_possession`:
    `holder != self._no_progress_holder` triggers a full reset, including
    when the ball leaves and `holder` becomes `None`), a gap resets the
    clock entirely -- neither 5s stretch alone crosses 8s, so this must NOT
    produce an event."""
    acc = MatchStatsAccumulator()
    _run_no_progress_ticks(acc, n_ticks=5 * 60, command=RefereeCommand.NORMAL_START, start_ts=0.0)
    # Ball rolls far away -- out of the 0.35m radius entirely.
    _run_no_progress_ticks(acc, n_ticks=1 * 60, command=RefereeCommand.NORMAL_START, start_ts=5.0, ball_xy=(3.0, 0.0))
    # Ball comes back within bump range; robot resumes bumping it.
    _run_no_progress_ticks(acc, n_ticks=5 * 60, command=RefereeCommand.NORMAL_START, start_ts=6.0)

    stats = acc.finalize()
    assert [e for e in stats.stall_events if e.kind == "NO_PROGRESS_POSSESSION"] == []


def test_no_progress_second_continuous_stretch_fires_once_it_alone_exceeds_threshold():
    """Same gap-reset shape as above, but the second stretch alone is long
    enough (9s > 8s threshold) to cross the threshold on its own -- confirms
    the clock genuinely restarted at the resume point rather than merely
    pausing, and that the watchdog still fires once that second stretch
    alone is long enough."""
    acc = MatchStatsAccumulator()
    _run_no_progress_ticks(acc, n_ticks=5 * 60, command=RefereeCommand.NORMAL_START, start_ts=0.0)
    _run_no_progress_ticks(acc, n_ticks=1 * 60, command=RefereeCommand.NORMAL_START, start_ts=5.0, ball_xy=(3.0, 0.0))
    # Resumed stretch alone is 9s > _NO_PROGRESS_SECONDS (8s).
    _run_no_progress_ticks(acc, n_ticks=9 * 60, command=RefereeCommand.NORMAL_START, start_ts=6.0)

    stats = acc.finalize()
    events = [e for e in stats.stall_events if e.kind == "NO_PROGRESS_POSSESSION"]
    assert len(events) == 1
    # Onset is 8s into the *second* stretch, i.e. at absolute sim time 6+8=14s.
    assert events[0].sim_time == pytest.approx(14.0, abs=2 * TICK_DT)


def test_short_stretch_near_ball_without_possession_does_not_stall():
    acc = MatchStatsAccumulator()
    _run_no_progress_ticks(acc, n_ticks=4 * 60, command=RefereeCommand.NORMAL_START)  # well under 8s

    stats = acc.finalize()
    assert [e for e in stats.stall_events if e.kind == "NO_PROGRESS_POSSESSION"] == []


def test_no_progress_attributes_to_whichever_robot_is_nearest_even_across_teams():
    """Two robots from different sides are both near the ball at different
    times -- confirm the watchdog attributes to whichever specific robot was
    nearest each tick, and that a change in the *nearest* identity (even
    across teams) resets the clock rather than attributing a mixed stretch
    to either robot."""
    acc = MatchStatsAccumulator()
    # First 5s: only the friendly robot is near the ball (enemy far away).
    for i in range(5 * 60):
        ts = (i + 1) * TICK_DT
        xy = _NO_PROGRESS_BUMP_XY_JITTER if (i % 2 == 0) else _NO_PROGRESS_BUMP_XY
        acc.record_tick(
            _custom_frame(ts, RefereeCommand.NORMAL_START, ball_xy=xy, friendly_xy=(0.0, 0.0), enemy_xy=(4.0, 0.0))
        )
    # Next 9s: the enemy robot becomes nearest instead (friendly moves away)
    # -- a different (side, id) identity, so the clock must restart, not
    # accumulate onto the friendly robot's earlier 5s.
    for i in range(9 * 60):
        ts = 5.0 + (i + 1) * TICK_DT
        xy = _NO_PROGRESS_BUMP_XY_JITTER if (i % 2 == 0) else _NO_PROGRESS_BUMP_XY
        acc.record_tick(
            _custom_frame(ts, RefereeCommand.NORMAL_START, ball_xy=xy, friendly_xy=(10.0, 10.0), enemy_xy=(0.0, 0.0))
        )

    stats = acc.finalize()
    events = [e for e in stats.stall_events if e.kind == "NO_PROGRESS_POSSESSION"]
    assert len(events) == 1
    # Attributed to the enemy robot (id=2), whose stretch alone crossed 8s;
    # the earlier friendly stretch (id=1) never reached the threshold and
    # must not contribute to this event's robot_ids or onset time.
    assert events[0].robot_ids == (2,)
    assert events[0].sim_time == pytest.approx(5.0 + 8.0, abs=2 * TICK_DT)


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
