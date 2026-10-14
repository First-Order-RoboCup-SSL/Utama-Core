"""SSL rulebook §8.1 "No Progress In Game": "If there is no progress in the game
for 5 seconds (Division A) or 10 seconds (Division B) while both teams are allowed
to manipulate the ball, the game is stopped and continued by a forced start."

The referee had no such rule: every COMMITTED_FROZEN stall in the 2026-09-28
round-robins was a ball that sat still for the rest of the match.
"""

import pytest

from utama_core.config.field_params import STANDARD_FIELD_DIMS
from utama_core.custom_referee.geometry import RefereeGeometry
from utama_core.custom_referee.rules.no_progress_rule import NoProgressRule
from utama_core.custom_referee.state_machine import GameStateMachine
from utama_core.entities.data.vector import Vector2D, Vector3D
from utama_core.entities.game.ball import Ball
from utama_core.entities.game.game_frame import GameFrame
from utama_core.entities.game.robot import Robot
from utama_core.entities.referee.referee_command import RefereeCommand

GEO = RefereeGeometry.from_field_dims(STANDARD_FIELD_DIMS)


def _frame(ts: float, x: float = 1.0, y: float = 0.5, friendly_robots=None) -> GameFrame:
    zero = Vector3D(0, 0, 0)
    return GameFrame(
        ts=ts,
        my_team_is_yellow=True,
        my_team_is_right=False,
        friendly_robots=friendly_robots or {},
        enemy_robots={},
        ball=Ball(p=Vector3D(x, y, 0), v=zero, a=zero),
        referee=None,
    )


def _run(rule, times, command=RefereeCommand.FORCE_START, xs=None):
    result = None
    for k, ts in enumerate(times):
        x = xs[k] if xs is not None else 1.0
        result = rule.check(_frame(ts, x=x), GEO, command)
    return result


def test_stopped_and_force_started_after_ten_seconds_without_progress():
    rule = NoProgressRule(max_seconds=10.0)
    assert _run(rule, [0.0, 5.0, 9.99]) is None
    v = rule.check(_frame(10.0), GEO, RefereeCommand.FORCE_START)
    assert v is not None
    assert v.suggested_command == RefereeCommand.STOP
    assert v.next_command == RefereeCommand.FORCE_START
    assert v.designated_position == pytest.approx((1.0, 0.5))
    assert v.offending_teams == ()


def test_ball_moving_resets_the_clock():
    rule = NoProgressRule(max_seconds=10.0)
    assert _run(rule, [0.0, 6.0, 12.0], xs=[1.0, 1.1, 1.1]) is None


@pytest.mark.parametrize("command", [RefereeCommand.STOP, RefereeCommand.DIRECT_FREE_YELLOW])
def test_not_counted_while_the_ball_is_out_of_play(command):
    rule = NoProgressRule(max_seconds=10.0)
    assert _run(rule, [0.0, 10.0, 20.0], command=command) is None


def test_state_machine_continues_with_force_start_once_robots_clear():
    sm = GameStateMachine(half_duration_seconds=300.0, kickoff_team="yellow", n_robots_yellow=3, n_robots_blue=3)
    sm.seed_clock(0.0)
    sm.set_command(RefereeCommand.FORCE_START, 0.0)
    rule = NoProgressRule(max_seconds=10.0)
    rule.check(_frame(0.0), GEO, RefereeCommand.FORCE_START)
    v = rule.check(_frame(10.0), GEO, RefereeCommand.FORCE_START)
    near = {
        0: Robot(
            id=0,
            is_friendly=True,
            has_ball=False,
            p=Vector2D(1.2, 0.5),
            v=Vector2D(0, 0),
            a=Vector2D(0, 0),
            orientation=0.0,
        )
    }
    assert sm.step(10.0, v, _frame(10.0, friendly_robots=near)).referee_command == RefereeCommand.STOP
    assert sm.step(10.5, None, _frame(10.5)).referee_command == RefereeCommand.FORCE_START


def test_simulation_profile_stops_a_frozen_game():
    from utama_core.custom_referee.custom_referee import CustomReferee

    referee = CustomReferee.from_profile_name("simulation", n_robots_yellow=1, n_robots_blue=1)
    referee.seed_clock(0.0, initial_command=RefereeCommand.FORCE_START)
    for ts in (0.0, 5.0, 9.9):
        referee.step(_frame(ts), current_time=ts)
    assert referee.last_violation is None
    # A robot beside the ball holds the STOP for this tick.
    near = {
        0: Robot(
            id=0,
            is_friendly=True,
            has_ball=False,
            p=Vector2D(1.2, 0.5),
            v=Vector2D(0, 0),
            a=Vector2D(0, 0),
            orientation=0.0,
        )
    }
    result = referee.step(_frame(10.1, friendly_robots=near), current_time=10.1)
    assert referee.last_violation.rule_name == "no_progress"
    assert result.referee_command == RefereeCommand.STOP


# ---------------------------------------------------------------------------
# The clock starts once the ball is in play (not at a NORMAL_START with the ball unmoved)
# ---------------------------------------------------------------------------


def _frame_at(ts: float, x: float = 1.0, y: float = 0.5) -> GameFrame:
    return _frame(ts, x=x, y=y)


def test_normal_start_with_unmoved_ball_does_not_fire_at_ten_seconds():
    # Kickoff/penalty: the ball is not in play until it moves or auto-starts (10 s).
    # Real matches: the old clock fired exactly 10 s after such a restart, 226 of 422 false alarms.
    rule = NoProgressRule(max_seconds=10.0)
    for ts in (0.0, 5.0, 10.5):
        assert rule.check(_frame_at(ts), GEO, RefereeCommand.NORMAL_START) is None


def test_ball_moved_four_point_nine_cm_is_not_in_play():
    rule = NoProgressRule(max_seconds=10.0)
    assert rule.check(_frame_at(0.0, x=1.0), GEO, RefereeCommand.NORMAL_START) is None
    assert rule.check(_frame_at(2.0, x=1.049), GEO, RefereeCommand.NORMAL_START) is None
    # Still not in play at 10.5 s: the clock starts at the 10 s auto-in-play, so no call yet.
    assert rule.check(_frame_at(10.5, x=1.049), GEO, RefereeCommand.NORMAL_START) is None


def test_ball_moved_five_point_one_cm_starts_the_clock():
    rule = NoProgressRule(max_seconds=10.0)
    assert rule.check(_frame_at(0.0, x=1.0), GEO, RefereeCommand.NORMAL_START) is None
    assert rule.check(_frame_at(2.0, x=1.051), GEO, RefereeCommand.NORMAL_START) is None
    assert rule.check(_frame_at(11.9, x=1.051), GEO, RefereeCommand.NORMAL_START) is None
    v = rule.check(_frame_at(12.0, x=1.051), GEO, RefereeCommand.NORMAL_START)
    assert v is not None and v.rule_name == "no_progress"


def test_force_start_clock_runs_from_the_first_tick():
    # A forced start puts the ball in play at once, so the clock is not gated on movement.
    rule = NoProgressRule(max_seconds=10.0)
    assert _run(rule, [0.0, 10.0]) is not None


# ---------------------------------------------------------------------------
# No progress is not counted while the ball is inside either defense area
# ---------------------------------------------------------------------------

# Standard field: the left box is x <= -3.5, the right box x >= 3.5 (|y| <= 1.0).
_LEFT_BOX_EDGE_X = -GEO.half_length + 2 * GEO.half_defense_depth


@pytest.mark.parametrize("side", [-1.0, 1.0])
def test_ball_just_inside_a_defense_area_never_fires(side):
    rule = NoProgressRule(max_seconds=10.0)
    x = side * (abs(_LEFT_BOX_EDGE_X) + 0.01)  # 1 cm inside the box
    for ts in (0.0, 5.0, 10.0, 15.0):
        assert rule.check(_frame_at(ts, x=x, y=0.0), GEO, RefereeCommand.FORCE_START) is None


@pytest.mark.parametrize("side", [-1.0, 1.0])
def test_ball_just_outside_a_defense_area_still_fires(side):
    rule = NoProgressRule(max_seconds=10.0)
    x = side * (abs(_LEFT_BOX_EDGE_X) - 0.01)  # 1 cm outside the box
    assert rule.check(_frame_at(0.0, x=x, y=0.0), GEO, RefereeCommand.FORCE_START) is None
    v = rule.check(_frame_at(10.0, x=x, y=0.0), GEO, RefereeCommand.FORCE_START)
    assert v is not None and v.rule_name == "no_progress"


def test_leaving_a_defense_area_restarts_the_clock():
    rule = NoProgressRule(max_seconds=10.0)
    inside_x = -(abs(_LEFT_BOX_EDGE_X) + 0.01)
    for ts in (0.0, 5.0, 9.5):
        assert rule.check(_frame_at(ts, x=inside_x, y=0.0), GEO, RefereeCommand.FORCE_START) is None
    # Out at 9.5 s (1 cm outside the box edge), then unmoved: the 10 s run starts from there.
    outside_x = -(abs(_LEFT_BOX_EDGE_X) - 0.01)
    assert rule.check(_frame_at(9.6, x=outside_x, y=0.0), GEO, RefereeCommand.FORCE_START) is None
    assert rule.check(_frame_at(19.5, x=outside_x, y=0.0), GEO, RefereeCommand.FORCE_START) is None
    v = rule.check(_frame_at(19.6, x=outside_x, y=0.0), GEO, RefereeCommand.FORCE_START)
    assert v is not None and v.rule_name == "no_progress"


def test_ball_in_play_stays_in_play_after_leaving_a_defense_area():
    # Live play after a kickoff stays under NORMAL_START. Once the ball has moved it is in
    # play; a pass through a defense area must not send it back to "waiting for the
    # kickoff", which would hold the clock another 10 s (a 20 s no-progress).
    rule = NoProgressRule(max_seconds=10.0)
    inside_x = -(abs(_LEFT_BOX_EDGE_X) + 0.01)
    outside_x = -(abs(_LEFT_BOX_EDGE_X) - 0.01)
    assert rule.check(_frame_at(0.0, x=0.0, y=0.0), GEO, RefereeCommand.NORMAL_START) is None
    assert rule.check(_frame_at(1.0, x=-1.0, y=0.0), GEO, RefereeCommand.NORMAL_START) is None
    assert rule.check(_frame_at(2.0, x=inside_x, y=0.0), GEO, RefereeCommand.NORMAL_START) is None
    assert rule.check(_frame_at(3.0, x=outside_x, y=0.0), GEO, RefereeCommand.NORMAL_START) is None
    assert rule.check(_frame_at(12.9, x=outside_x, y=0.0), GEO, RefereeCommand.NORMAL_START) is None
    v = rule.check(_frame_at(13.0, x=outside_x, y=0.0), GEO, RefereeCommand.NORMAL_START)
    assert v is not None and v.rule_name == "no_progress"
