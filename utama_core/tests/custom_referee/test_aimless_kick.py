"""SSL rulebook §6.2.3 (Division B only): "A kick is aimless when after the ball
touched a robot, it subsequently crossed the halfline and then its opponent's goal
line outside the goal without touching another robot." The ball is placed where it
was kicked and the opponent gets the free kick.

The referee gave a goal kick in the opponent's corner instead, so a blind clearance
from our own half cost nothing.
"""

import pytest

from utama_core.config.field_params import STANDARD_FIELD_DIMS
from utama_core.custom_referee.geometry import RefereeGeometry
from utama_core.custom_referee.profiles.profile_loader import load_profile
from utama_core.custom_referee.rules.out_of_bounds_rule import OutOfBoundsRule
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.tests.custom_referee.helpers import ball as _ball
from utama_core.tests.custom_referee.helpers import frame as _frame
from utama_core.tests.custom_referee.helpers import robot as _robot

GEO = RefereeGeometry.from_field_dims(STANDARD_FIELD_DIMS)
# Yellow (friendly) defends the left goal, so it attacks the right goal line.
_EXIT_OVER_BLUES_GOAL_LINE = (GEO.half_length + 0.05, 1.0)


def _touch(rule, x, y, is_friendly=True, ts=9.0):
    robots = {0: _robot(0, x, y, is_friendly=is_friendly, has_ball=True)}
    key = "friendly_robots" if is_friendly else "enemy_robots"
    assert rule.check(_frame(ball=_ball(x, y), ts=ts, **{key: robots}), GEO, RefereeCommand.NORMAL_START) is None


def _exit(rule, x, y, command=RefereeCommand.NORMAL_START):
    return rule.check(_frame(ball=_ball(x, y), ts=10.0), GEO, command)


def test_clearance_from_own_half_over_the_far_goal_line_restarts_where_it_was_kicked():
    rule = OutOfBoundsRule()
    _touch(rule, -1.0, 0.5)
    v = _exit(rule, *_EXIT_OVER_BLUES_GOAL_LINE)
    assert v is not None
    assert v.next_command == RefereeCommand.DIRECT_FREE_BLUE
    assert v.status_message == "Aimless kick"
    assert v.designated_position == pytest.approx((-1.0, 0.5))


@pytest.mark.parametrize("touch_x", [0.0, 0.01])
def test_a_kick_from_the_halfway_line_or_beyond_is_a_goal_kick(touch_x):
    # On the line itself the ball doesn't cross it: a kick-off "cannot be aimless".
    rule = OutOfBoundsRule()
    _touch(rule, touch_x, 0.5)
    v = _exit(rule, *_EXIT_OVER_BLUES_GOAL_LINE)
    assert v.status_message == "Ball out of bounds"
    assert v.designated_position == pytest.approx(GEO.goal_kick_position(1.0, 1.0))


def test_a_teammate_touch_past_the_halfway_line_makes_it_a_goal_kick():
    rule = OutOfBoundsRule()
    _touch(rule, -1.0, 0.5, ts=8.0)
    _touch(rule, 1.0, 0.5, ts=9.0)
    v = _exit(rule, *_EXIT_OVER_BLUES_GOAL_LINE)
    assert v.status_message == "Ball out of bounds"
    assert v.next_command == RefereeCommand.DIRECT_FREE_BLUE


def test_over_the_kickers_own_goal_line_is_a_corner_kick():
    # Blue clears from its own half (right) all the way over yellow's goal line
    # on the left: aimless for blue, a free kick for yellow at the kick point.
    rule = OutOfBoundsRule()
    _touch(rule, 1.5, -0.5, is_friendly=False)
    v = _exit(rule, -(GEO.half_length + 0.05), 1.0)
    assert v.status_message == "Aimless kick"
    assert v.next_command == RefereeCommand.DIRECT_FREE_YELLOW
    # Yellow then puts it back over its own goal line from its own half: a corner kick.
    rule = OutOfBoundsRule()
    _touch(rule, 1.5, -0.5, is_friendly=False, ts=8.0)
    _touch(rule, -2.0, 0.5, ts=9.0)
    v = _exit(rule, -(GEO.half_length + 0.05), 1.0)
    assert v.status_message == "Ball out of bounds"
    assert v.designated_position == pytest.approx((-(GEO.half_length - 0.5), GEO.half_width - 0.5))


def test_a_keeper_clearance_restarts_outside_the_defense_area():
    rule = OutOfBoundsRule()
    _touch(rule, -GEO.half_length + 0.3, 0.2)
    v = _exit(rule, *_EXIT_OVER_BLUES_GOAL_LINE)
    assert v.status_message == "Aimless kick"
    x, y = v.designated_position
    assert y == pytest.approx(0.2)
    assert GEO.distance_to_left_defense_area(x, y) >= 1.0 - 1e-9


def test_a_touch_before_a_stoppage_does_not_count():
    rule = OutOfBoundsRule()
    _touch(rule, -1.0, 0.5)
    assert _exit(rule, 0.0, 0.0, command=RefereeCommand.STOP) is None
    v = _exit(rule, *_EXIT_OVER_BLUES_GOAL_LINE)
    assert v.status_message == "Ball out of bounds"


def test_off_without_the_rule():
    rule = OutOfBoundsRule(aimless_kick=False)
    _touch(rule, -1.0, 0.5)
    v = _exit(rule, *_EXIT_OVER_BLUES_GOAL_LINE)
    assert v.status_message == "Ball out of bounds"


def test_simulation_profile_plays_division_b():
    assert load_profile("simulation").rules.out_of_bounds.aimless_kick
