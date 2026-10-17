"""Pure checks (no rsim) of the `requires` conditions in `tools/moments.py`: each fires on the case it guards and not on the one it must allow, over
synthetic ticks."""

import dataclasses

import pytest

from tools import moments
from utama_core.entities.referee.referee_command import RefereeCommand

DT = 1 / moments.TICKS_PER_SECOND
FREE_KICK = RefereeCommand.DIRECT_FREE_BLUE


def _tick(t, command, ball, ours):
    return moments.Tick(
        t=t,
        command=command,
        ball=ball,
        ours=ours,
        our_speeds={},
        our_foul=None,
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


def test_a_failed_requirement_is_reported_as_did_not_arise():
    m = _probe(requires=["robot_near_ball"], checks=[])
    far = [_tick(0.0, None, (0.0, 0.0), {1: (3.0, 0.0)})]
    assert moments.judge(m, far) == [
        "did not arise: robot_near_ball (the closest robot of ours to the ball was 3.00 m, need 1.0 m)"
    ]
    near = [_tick(0.0, None, (0.0, 0.0), {1: (0.5, 0.0)})]
    assert moments.judge(m, near) == []


def test_requires_round_trip_through_json_and_unknown_ones_are_named():
    m = _probe(requires=["restart_lasts", "robot_near_ball"], checks=["*"])
    assert moments.from_dict(moments.to_dict(m)).requires == ["restart_lasts", "robot_near_ball"]
    with pytest.raises(moments.MomentError, match=r"unknown requires 'ball_kickd'"):
        moments.validate(_probe(requires=["ball_kickd"]))


def test_requires_and_checks_are_registered_by_name():
    assert {"restart_lasts", "robot_near_ball"} == set(moments.REQUIRES)
