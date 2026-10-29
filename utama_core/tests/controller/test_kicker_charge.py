"""A real robot that has not kicked for `KEEP_S` fires an empty kick, unless the ball is on or
near it or play is halted; its own kicks reset the wait."""

from types import SimpleNamespace

import pytest

from utama_core.entities.data.command import RobotCommand
from utama_core.entities.data.vector import Vector2D, Vector3D
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.team_controller.src.controllers.real.kicker_charge import (
    CLEAR_M,
    KEEP_S,
    KickerCharge,
)

STILL = RobotCommand(0.0, 0.0, 0.0, kick=False, chip=False, dribble=False)


def game(ts, ball_at=(2.0, 0.0), has_ball=False, command=RefereeCommand.FORCE_START):
    robot = SimpleNamespace(p=Vector2D(0.0, 0.0), has_ball=has_ball)
    return SimpleNamespace(
        ts=ts,
        referee=SimpleNamespace(referee_command=command),
        ball=None if ball_at is None else SimpleNamespace(p=Vector3D(*ball_at, 0.0)),
        friendly_robots={0: robot},
    )


def kicks(charge, g):
    return charge.apply(g, {0: STILL})[0].kick


def test_it_kicks_at_the_start_then_again_only_once_keep_s_has_passed():
    charge = KickerCharge()
    assert kicks(charge, game(0.0))
    assert not kicks(charge, game(KEEP_S - 0.01))
    assert kicks(charge, game(KEEP_S + 0.01))


def test_a_kick_of_its_own_restarts_the_wait():
    charge = KickerCharge()
    charge.apply(game(0.0), {0: STILL._replace(kick=True)})
    assert not kicks(charge, game(KEEP_S - 0.01))


@pytest.mark.parametrize(
    "kw",
    [
        {"has_ball": True},
        {"ball_at": (CLEAR_M - 0.01, 0.0)},
        {"ball_at": None},  # ball unseen: perhaps hidden under the robot
        {"command": RefereeCommand.HALT},
    ],
)
def test_no_empty_kick_with_the_ball_on_or_near_it_unseen_or_in_a_halt(kw):
    assert not kicks(KickerCharge(), game(100.0, **kw))
    assert kicks(KickerCharge(), game(100.0, ball_at=(CLEAR_M + 0.01, 0.0)))
