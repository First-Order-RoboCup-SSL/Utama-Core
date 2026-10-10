"""A ball out of bounds is charged to the robot that last touched it.

The foul log used to charge it to that team's robot nearest the ball when it crossed
the line: a full-power clearance rolls ~17 m, so that was usually a robot standing near
the touch line that never played the ball, and the per-tactic out-of-bounds counts
(e.g. `PressAndContainTactic` 356 in tournament_20261005_170958) named the wrong tactic.
"""

from utama_core.config.field_params import STANDARD_FIELD_DIMS
from utama_core.custom_referee.geometry import RefereeGeometry
from utama_core.custom_referee.rules.out_of_bounds_rule import OutOfBoundsRule
from utama_core.engine.match_stats import MatchStatsAccumulator
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.tests.custom_referee.helpers import ball as _ball
from utama_core.tests.custom_referee.helpers import frame as _frame
from utama_core.tests.custom_referee.helpers import robot as _robot

GEO = RefereeGeometry.from_field_dims(STANDARD_FIELD_DIMS)
_OVER_THE_TOUCH_LINE = (1.0, GEO.half_width + 0.05)


def _kick_then_exit(my_team_is_yellow=True):
    """Friendly robot 2 kicks at (-1, 0.5); when the ball crosses the touch line,
    friendly robot 4 stands next to it and robot 2 is far away."""
    rule = OutOfBoundsRule()
    kicker = _robot(2, -1.0, 0.5, is_friendly=True, has_ball=True)
    kick = _frame(ball=_ball(-1.0, 0.5), friendly_robots={2: kicker}, my_team_is_yellow=my_team_is_yellow, ts=9.0)
    assert rule.check(kick, GEO, RefereeCommand.NORMAL_START) is None
    robots = {
        2: _robot(2, -1.0, 0.5, is_friendly=True),
        4: _robot(4, _OVER_THE_TOUCH_LINE[0], _OVER_THE_TOUCH_LINE[1] - 0.3, is_friendly=True),
    }
    exit_frame = _frame(
        ball=_ball(*_OVER_THE_TOUCH_LINE), friendly_robots=robots, my_team_is_yellow=my_team_is_yellow, ts=10.0
    )
    return rule.check(exit_frame, GEO, RefereeCommand.NORMAL_START), exit_frame


def test_the_rule_names_the_robot_that_kicked_the_ball_out():
    v, _ = _kick_then_exit()
    assert v.offending_robots == ((True, 2),)
    v, _ = _kick_then_exit(my_team_is_yellow=False)
    assert v.offending_robots == ((False, 2),)


def test_the_foul_log_charges_the_kicker_not_the_robot_nearest_the_line():
    v, exit_frame = _kick_then_exit()
    acc = MatchStatsAccumulator()
    acc.record_rule_violation(v, True, exit_frame, {"friendly": {2: "clear", 4: "defense"}})
    (foul,) = acc._fouls
    assert (foul.robot_id, foul.tactic, foul.inferred) == (2, "clear", False)


def test_no_touch_seen_since_the_restart_leaves_it_to_the_nearest_robot():
    rule = OutOfBoundsRule()
    robots = {4: _robot(4, _OVER_THE_TOUCH_LINE[0], _OVER_THE_TOUCH_LINE[1] - 0.3, is_friendly=True)}
    v = rule.check(_frame(ball=_ball(*_OVER_THE_TOUCH_LINE), friendly_robots=robots), GEO, RefereeCommand.NORMAL_START)
    assert v.offending_robots == ()
