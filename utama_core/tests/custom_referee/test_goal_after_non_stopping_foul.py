"""SSL rulebook §7: a goal counts only if the scoring team "did not commit any non
stopping foul in the last two seconds before the ball entered the goal". With ball
speed, keep-out and attacker-in-defense-area now non-stopping, a shot over 6.5 m/s
would otherwise score.
"""

import pytest

from utama_core.custom_referee.custom_referee import CustomReferee
from utama_core.custom_referee.rules.base_rule import BaseRule, RuleViolation
from utama_core.entities.data.vector import Vector3D
from utama_core.entities.game.ball import Ball
from utama_core.entities.game.game_frame import GameFrame
from utama_core.entities.referee.referee_command import RefereeCommand


class _FoulOnce(BaseRule):
    """A non-stopping foul by yellow at `at`."""

    def __init__(self, at: float) -> None:
        self._at = at

    def check(self, game_frame, geometry, current_command, designated_position=None):
        if game_frame.ts != self._at:
            return None
        return RuleViolation(
            rule_name="stub_non_stopping",
            suggested_command=current_command,
            next_command=None,
            status_message="stub",
            offending_teams=(True,),
            is_stopping=False,
        )

    def reset(self) -> None:
        pass


def _frame(ts: float, x: float) -> GameFrame:
    zero = Vector3D(0, 0, 0)
    return GameFrame(
        ts=ts,
        my_team_is_yellow=True,
        my_team_is_right=False,
        friendly_robots={},
        enemy_robots={},
        ball=Ball(p=Vector3D(x, -0.1, 0), v=zero, a=zero),
        referee=None,
    )


def _shoot(foul_at: float):
    referee = CustomReferee.from_profile_name("simulation", n_robots_yellow=1, n_robots_blue=1)
    referee.seed_clock(0.0, initial_command=RefereeCommand.FORCE_START)
    referee._rules.insert(0, _FoulOnce(foul_at))
    referee.step(_frame(foul_at, 3.0), current_time=foul_at)
    # Yellow (on the left) scores in the right goal at t=3.
    result = referee.step(_frame(3.0, referee._geometry.half_length + 0.05), current_time=3.0)
    return referee, result


def test_goal_after_a_foul_within_two_seconds_is_a_goal_kick():
    referee, result = _shoot(foul_at=1.5)
    assert result.yellow_team.score == 0
    assert referee.last_violation.rule_name == "invalid_goal"
    # Blue places the ball for its goal kick.
    assert result.referee_command == RefereeCommand.BALL_PLACEMENT_BLUE
    assert result.next_command == RefereeCommand.DIRECT_FREE_BLUE
    geo = referee._geometry
    assert result.designated_position == pytest.approx((geo.half_length - 1.0, -(geo.half_width - 0.5)))


def test_goal_more_than_two_seconds_after_the_foul_counts():
    _, result = _shoot(foul_at=0.5)
    assert result.yellow_team.score == 1
