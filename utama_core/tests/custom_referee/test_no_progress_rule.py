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
