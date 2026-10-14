"""SSL rulebook §5.3.5: a penalty still in play 10 s after its normal start is
stopped, no goal, and continued by a goal kick for the defending team (§6.2.1:
1 m from the goal line, next to the closest touch line).

The referee had no such limit: a penalty's NORMAL_START turned into open play.
"""

import pytest

from utama_core.config.field_params import STANDARD_FIELD_DIMS
from utama_core.custom_referee.geometry import RefereeGeometry
from utama_core.custom_referee.rules.penalty_time_limit_rule import PenaltyTimeLimitRule
from utama_core.entities.data.vector import Vector3D
from utama_core.entities.game.ball import Ball
from utama_core.entities.game.game_frame import GameFrame
from utama_core.entities.referee.referee_command import RefereeCommand

GEO = RefereeGeometry.from_field_dims(STANDARD_FIELD_DIMS)


def _frame(ts: float, y: float = -0.4) -> GameFrame:
    zero = Vector3D(0, 0, 0)
    return GameFrame(
        ts=ts,
        my_team_is_yellow=True,
        my_team_is_right=False,
        friendly_robots={},
        enemy_robots={},
        ball=Ball(p=Vector3D(3.5, y, 0), v=zero, a=zero),
        referee=None,
    )


def _penalty(rule, prepare=RefereeCommand.PREPARE_PENALTY_YELLOW):
    rule.check(_frame(0.0), GEO, prepare)
    rule.reset()
    return rule.check(_frame(0.0), GEO, RefereeCommand.NORMAL_START)


def test_penalty_in_play_at_ten_seconds_becomes_a_goal_kick_for_the_defenders():
    rule = PenaltyTimeLimitRule(max_seconds=10.0)
    assert _penalty(rule) is None
    assert rule.check(_frame(9.99), GEO, RefereeCommand.NORMAL_START) is None
    v = rule.check(_frame(10.0), GEO, RefereeCommand.NORMAL_START)
    assert v is not None
    assert v.suggested_command == RefereeCommand.STOP
    assert v.next_command == RefereeCommand.DIRECT_FREE_BLUE
    # Yellow on the left attacks the right goal; the ball is on the -y side.
    assert v.designated_position == pytest.approx((GEO.half_length - 1.0, -(GEO.half_width - 0.5)))


def test_blue_penalty_goal_kick_is_in_front_of_the_left_goal():
    rule = PenaltyTimeLimitRule(max_seconds=10.0)
    _penalty(rule, RefereeCommand.PREPARE_PENALTY_BLUE)
    v = rule.check(_frame(10.0, y=0.3), GEO, RefereeCommand.NORMAL_START)
    assert v.next_command == RefereeCommand.DIRECT_FREE_YELLOW
    assert v.designated_position == pytest.approx((-(GEO.half_length - 1.0), GEO.half_width - 0.5))


def test_clock_survives_the_force_start_auto_advance():
    rule = PenaltyTimeLimitRule(max_seconds=10.0)
    _penalty(rule)
    rule.reset()
    assert rule.check(_frame(10.0), GEO, RefereeCommand.FORCE_START) is not None


def test_stopping_play_ends_the_penalty():
    rule = PenaltyTimeLimitRule(max_seconds=10.0)
    _penalty(rule)
    rule.check(_frame(3.0), GEO, RefereeCommand.STOP)
    rule.reset()
    assert rule.check(_frame(20.0), GEO, RefereeCommand.FORCE_START) is None


def test_ordinary_kickoff_has_no_time_limit():
    rule = PenaltyTimeLimitRule(max_seconds=10.0)
    _penalty(rule, RefereeCommand.PREPARE_KICKOFF_YELLOW)
    assert rule.check(_frame(20.0), GEO, RefereeCommand.NORMAL_START) is None


def test_simulation_profile_stops_a_penalty_after_ten_seconds():
    from utama_core.custom_referee.custom_referee import CustomReferee

    referee = CustomReferee.from_profile_name("simulation", n_robots_yellow=1, n_robots_blue=1)
    referee.seed_clock(0.0, initial_command=RefereeCommand.FORCE_START)
    referee.force_command(RefereeCommand.PREPARE_PENALTY_YELLOW, 0.0)
    referee.step(_frame(0.0), current_time=0.0)
    referee.force_command(RefereeCommand.NORMAL_START, 0.0)
    for k, ts in enumerate((0.0, 4.0, 8.0, 10.0)):
        # The ball keeps moving, so no-progress never fires first.
        frame = _frame(ts, y=-0.4 + 0.1 * k)
        referee.step(frame, current_time=ts)
    assert referee.last_violation is not None
    assert referee.last_violation.rule_name == "penalty_time_limit"
