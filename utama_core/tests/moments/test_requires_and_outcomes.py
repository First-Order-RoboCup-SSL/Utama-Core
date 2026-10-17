"""Pure checks (no rsim) of the `requires` conditions and the outcome checks `pass_received` and
`placement_completes` in `tools/moments.py`: each fires on the case it guards and not on the one
it must allow, over synthetic ticks."""

import dataclasses

import pytest

from tools import moments
from utama_core.entities.referee.referee_command import RefereeCommand

DT = 1 / moments.TICKS_PER_SECOND
FREE_KICK = RefereeCommand.DIRECT_FREE_BLUE
PLACING = RefereeCommand.BALL_PLACEMENT_YELLOW


def _tick(t, command, ball, ours, speed=0.0, designated=None, tactic_robots=frozenset()):
    return moments.Tick(
        t=t,
        command=command,
        ball=ball,
        ours=ours,
        our_speeds={},
        our_foul=None,
        tactic_robots=frozenset(tactic_robots),
        ball_speed=speed,
        designated=designated,
    )


def _probe(**changes):
    return dataclasses.replace(moments.template("probe", "a note", "NORMAL_START", ["*"]), **changes)


# --- restart_lasts -------------------------------------------------------------------------------


def test_restart_lasts_holds_when_the_restart_outlasts_its_second():
    ticks = [_tick(i * DT, FREE_KICK, (0.0, 0.0), {1: (3.0, 0.0)}) for i in range(int(2.0 / DT))]
    assert moments.REQUIRES["restart_lasts"](ticks) is None


def test_restart_lasts_fires_when_the_restart_ends_early():
    ticks = [_tick(i * DT, FREE_KICK, (0.0, 0.0), {1: (3.0, 0.0)}) for i in range(int(0.4 / DT))]
    ticks += [_tick(0.4 + i * DT, RefereeCommand.NORMAL_START, (0.0, 0.0), {1: (3.0, 0.0)}) for i in range(60)]
    why = moments.REQUIRES["restart_lasts"](ticks)
    assert why is not None and "DIRECT_FREE_BLUE ended at 0.40 s" in why


def test_restart_lasts_fires_when_the_moment_does_not_start_from_a_restart():
    ticks = [_tick(i * DT, RefereeCommand.NORMAL_START, (0.0, 0.0), {1: (3.0, 0.0)}) for i in range(120)]
    assert "does not start from a restart" in moments.REQUIRES["restart_lasts"](ticks)


def test_restart_lasts_counts_a_restart_still_on_at_the_end_only_once_the_play_was_long_enough():
    short = [_tick(i * DT, FREE_KICK, (0.0, 0.0), {1: (3.0, 0.0)}) for i in range(int(0.5 / DT))]
    assert "still on at the end" in moments.REQUIRES["restart_lasts"](short)


# --- robot_near_ball -----------------------------------------------------------------------------


def test_robot_near_ball_holds_when_a_robot_of_ours_comes_within_a_metre():
    ticks = [_tick(0.0, FREE_KICK, (0.0, 0.0), {1: (3.0, 0.0)}), _tick(DT, FREE_KICK, (0.0, 0.0), {1: (0.9, 0.0)})]
    assert moments.REQUIRES["robot_near_ball"](ticks) is None


def test_robot_near_ball_fires_when_none_of_ours_gets_near():
    ticks = [_tick(i * DT, FREE_KICK, (0.0, 0.0), {1: (1.5, 0.0)}) for i in range(10)]
    assert "closest robot of ours to the ball was 1.50 m" in moments.REQUIRES["robot_near_ball"](ticks)


# --- ball_kicked ---------------------------------------------------------------------------------


def test_ball_kicked_holds_at_kick_speed_and_fires_below_it():
    kicked = [_tick(0.0, None, (0.0, 0.0), {}, speed=0.0), _tick(DT, None, (0.1, 0.0), {}, speed=2.5)]
    assert moments.REQUIRES["ball_kicked"](kicked) is None
    rolled = [_tick(0.0, None, (0.0, 0.0), {}, speed=1.9), _tick(DT, None, (0.1, 0.0), {}, speed=1.0)]
    assert "top speed was 1.90 m/s" in moments.REQUIRES["ball_kicked"](rolled)


def test_a_moving_ball_at_the_start_is_not_a_kick():
    """free_kick_robot_drives_in_at_start starts with the ball at 1.04 m/s."""
    ticks = [_tick(i * DT, FREE_KICK, (0.0, 0.0), {1: (3.0, 0.0)}, speed=1.04) for i in range(5)]
    assert moments.REQUIRES["ball_kicked"](ticks) is not None


def test_a_failed_requirement_is_reported_as_did_not_arise():
    m = _probe(requires=["ball_kicked"], checks=[])
    ticks = [_tick(0.0, None, (0.0, 0.0), {1: (3.0, 0.0)}, speed=0.2)]
    assert moments.judge(m, ticks) == ["did not arise: ball_kicked (the ball's top speed was 0.20 m/s, need 2.0 m/s)"]
    kicked = [_tick(0.0, None, (0.0, 0.0), {1: (3.0, 0.0)}, speed=3.0)]
    assert moments.judge(m, kicked) == []


def test_requires_round_trip_through_json_and_unknown_ones_are_named():
    m = _probe(requires=["restart_lasts", "robot_near_ball"], checks=["*"])
    assert moments.from_dict(moments.to_dict(m)).requires == ["restart_lasts", "robot_near_ball"]
    with pytest.raises(moments.MomentError, match=r"unknown requires 'ball_kickd'"):
        moments.validate(_probe(requires=["ball_kickd"]))


def test_requires_and_checks_are_registered_by_name():
    assert {"restart_lasts", "robot_near_ball", "ball_kicked"} == set(moments.REQUIRES)
    assert {"pass_received", "placement_completes"} <= set(moments.CHECKS)


# --- pass_received -------------------------------------------------------------------------------


def _pass_ticks(end_at, end_speed, end_ball=(1.0, 0.0)):
    """Robot 1 (the carrier, at the ball) kicks at t=0.1 at 4 m/s; robot 2 stands 1 m on. The ball
    then reaches `end_ball` at `end_speed` at t=0.2 (`end_at` lets a case place it elsewhere)."""
    ours = {1: (0.0, 0.0), 2: (1.0, 0.0)}
    tactic = {1, 2}
    return [
        _tick(0.0, None, (0.0, 0.0), ours, speed=0.0, tactic_robots=tactic),
        _tick(0.1, None, (0.5, 0.0), ours, speed=4.0, tactic_robots=tactic),
        _tick(0.2, None, end_ball, ours, speed=end_speed, tactic_robots=tactic),
    ][:end_at]


def test_pass_received_holds_when_the_receiver_takes_the_ball():
    assert moments.pass_received(_pass_ticks(3, end_speed=0.5, end_ball=(0.98, 0.0))) == []


def test_pass_received_fails_a_graze_at_pass_speed():
    """The ball passes the receiver at 4 m/s, 0.11 m off its centre: a miss, not a reception."""
    ticks = _pass_ticks(3, end_speed=4.0, end_ball=(1.0, 0.11))
    assert moments.pass_received(ticks) == ["pass_received: no robot of [2] got the ball from robot 1 by the end"]


def test_pass_received_fails_when_the_ball_never_reaches_kick_speed():
    ticks = [_tick(i * DT, None, (0.0, 0.0), {1: (0.0, 0.0), 2: (1.0, 0.0)}, tactic_robots={1, 2}) for i in range(5)]
    assert moments.pass_received(ticks) == ["pass_received: the ball never reached kick speed (2.0 m/s), so no pass"]


def test_pass_received_fails_when_only_the_carrier_has_the_ball():
    ours = {1: (0.0, 0.0), 2: (1.0, 0.0)}
    ticks = [
        _tick(0.0, None, (0.0, 0.0), ours, speed=0.0, tactic_robots={1}),
        _tick(0.1, None, (0.2, 0.0), ours, speed=0.3, tactic_robots={1}),
        _tick(0.2, None, (0.0, 0.0), ours, speed=4.0, tactic_robots={1}),
        _tick(0.3, None, (0.0, 0.0), ours, speed=0.3, tactic_robots={1}),
    ]
    assert moments.pass_received(ticks) == ["pass_received: no robot of [] got the ball from robot 1 by the end"]


def test_pass_received_counts_only_the_tactics_robots_when_one_ticks():
    """A robot of ours outside the tactic that takes the ball is not the tactic's reception."""
    ours = {1: (0.0, 0.0), 2: (1.0, 0.0), 3: (1.0, 0.0)}
    ticks = [
        _tick(0.0, None, (0.0, 0.0), ours, speed=0.0, tactic_robots={1, 2}),
        _tick(0.1, None, (0.5, 0.0), ours, speed=4.0, tactic_robots={1, 2}),
        _tick(0.2, None, (1.0, 0.0), ours, speed=0.5, tactic_robots={1, 2}),
    ]
    assert moments.pass_received(ticks) == []
    outside = {1: (0.0, 0.0), 2: (5.0, 5.0), 3: (1.0, 0.0)}
    ticks = [dataclasses.replace(t, ours=outside) for t in ticks]
    assert moments.pass_received(ticks) != []


# --- placement_completes -------------------------------------------------------------------------


def _placing(end_ball, designated=(2.0, -3.0), then=RefereeCommand.DIRECT_FREE_YELLOW, ticks_placing=3):
    ours = {4: (1.0, -2.0)}
    placing = [
        _tick(i * DT, PLACING, (2.0, -3.0) if i == ticks_placing - 1 else (0.0, -3.0), ours, designated=designated)
        for i in range(ticks_placing)
    ]
    placing[-1] = dataclasses.replace(placing[-1], ball=end_ball)
    after = [_tick(ticks_placing * DT, then, end_ball, ours, designated=designated)]
    return placing + after


def test_placement_completes_when_the_ball_is_within_tolerance_and_the_referee_moves_on():
    assert moments.placement_completes(_placing(end_ball=(2.05, -3.0))) == []


def test_placement_fails_when_the_placement_times_out_away_from_the_target():
    failures = moments.placement_completes(_placing(end_ball=(-2.0, -3.0)))
    assert len(failures) == 1 and "ended at t=0.03s with the ball 4.00 m" in failures[0]
    assert "tolerance 0.15 m" in failures[0]


def test_placement_fails_when_the_placement_is_still_on_at_the_end():
    ours = {4: (1.0, -2.0)}
    ticks = [_tick(i * DT, PLACING, (2.0, -3.0), ours, designated=(2.0, -3.0)) for i in range(5)]
    assert moments.placement_completes(ticks) == [
        "placement_completes: still BALL_PLACEMENT_YELLOW at the end, the ball not placed"
    ]


def test_placement_fails_when_the_moment_does_not_start_from_our_placement():
    ours = {4: (1.0, -2.0)}
    ticks = [
        _tick(i * DT, RefereeCommand.BALL_PLACEMENT_BLUE, (2.0, -3.0), ours, designated=(2.0, -3.0)) for i in range(5)
    ]
    assert moments.placement_completes(ticks) == [
        "placement_completes: the moment does not start from our BALL_PLACEMENT_YELLOW"
    ]


def test_placement_completes_is_the_check_named_in_a_moment():
    m = _probe(checks=["placement_completes", "*"], requires=[])
    failures = moments.judge(m, _placing(end_ball=(-2.0, -3.0)), we_are_yellow=True)
    assert any(f.startswith("placement_completes:") for f in failures)
