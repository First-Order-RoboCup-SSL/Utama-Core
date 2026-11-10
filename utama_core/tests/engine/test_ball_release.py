"""The backstop that kicks before a held ball costs a foul or jams against an enemy."""

import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from utama_core.engine.ball_release import (
    AIM_LIMIT_S,
    CARRY_AIM_M,
    CONTACT_M,
    HELD_STILL_S,
    PINNED_RELEASE_S,
    PINNED_WITH_LANE_S,
    BallRelease,
)
from utama_core.engine.context import TickContext
from utama_core.entities.data.command import RobotCommand
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.shared.pass_and_score_geometry import (
    ANNOUNCE_PASS_RAD,
    MAX_OPEN_TEAMMATE_M,
    open_teammate,
    reaches_own_goal,
    release_heading,
    upfield_heading,
)
from utama_core.tactics import _pass_and_score as pas

_STRAFE = RobotCommand(local_forward_vel=0, local_left_vel=0.3, angular_vel=0, kick=0, chip=0, dribble=1)
_TURN = RobotCommand(local_forward_vel=0, local_left_vel=0, angular_vel=2.0, kick=0, chip=0, dribble=1)
_ROBOT = 1


def _game(
    ts=0.0,
    ball=(0.0, 0.0),
    v=(0.0, 0.0),
    has_ball=True,
    heading=math.pi,
    enemy=None,
    cmd=RefereeCommand.NORMAL_START,
    mates=(),
):
    """Our goal at +x, the robot just behind the ball, facing the enemy goal (-x) by default."""
    robot = SimpleNamespace(p=Vector2D(ball[0] + 0.09, ball[1]), orientation=heading, has_ball=has_ball)
    enemies = {} if enemy is None else {0: SimpleNamespace(p=Vector2D(*enemy))}
    friends = {_ROBOT: robot, **{i + 2: SimpleNamespace(p=Vector2D(*m), has_ball=False) for i, m in enumerate(mates)}}
    return SimpleNamespace(
        ts=ts,
        ball=SimpleNamespace(p=Vector2D(*ball), v=Vector2D(*v)),
        referee=SimpleNamespace(referee_command=cmd),
        friendly_robots=friends,
        enemy_robots=enemies,
        field=SimpleNamespace(
            my_goal_line=np.array([[4.5, 0.5], [4.5, -0.5]]),
            enemy_goal_line=np.array([[-4.5, 0.5], [-4.5, -0.5]]),
            my_defense_area=np.array([[4.5, 1.0], [3.5, 1.0], [3.5, -1.0], [4.5, -1.0]]),
            half_length=4.5,
            half_width=3.0,
        ),
    )


@pytest.fixture(autouse=True)
def _turn(monkeypatch):
    monkeypatch.setattr("utama_core.engine.ball_release.turn_on_spot", lambda **kw: _TURN)


@pytest.fixture
def no_held_limit(monkeypatch):
    monkeypatch.setattr("utama_core.engine.ball_release.HELD_STILL_S", 100.0)


def _step(release, game, command=_STRAFE):
    return release.apply(game, MagicMock(), {_ROBOT: command})[_ROBOT]


def test_carry_is_left_alone_just_short_of_the_limit():
    release = BallRelease()
    _step(release, _game())
    assert _step(release, _game(ts=1.0, ball=(0.0, CARRY_AIM_M - 0.01))) is _STRAFE


def test_carry_is_released_at_the_limit():
    release = BallRelease()
    _step(release, _game())
    assert _step(release, _game(ts=1.0, ball=(0.0, 0.97))).kick


def test_a_fast_carry_is_released_early_enough_for_the_kick_to_land():
    release = BallRelease()
    _step(release, _game())
    assert _step(release, _game(ts=1.0, ball=(0.0, 0.85), v=(0.0, 1.0))).kick


def test_losing_the_ball_restarts_the_carry():
    release = BallRelease()
    _step(release, _game())
    _step(release, _game(ts=1.0, ball=(0.0, 0.5), has_ball=False))
    _step(release, _game(ts=1.1, ball=(0.0, 0.5)))
    assert not _step(release, _game(ts=2.0, ball=(0.0, 1.2))).kick


def test_pinned_with_a_lane_open_is_released_early(monkeypatch):
    enemy = (-CONTACT_M + 0.01, 0.0)
    monkeypatch.setattr("utama_core.engine.ball_release.open_teammate", lambda *a: 2)
    release = BallRelease()
    _step(release, _game(enemy=enemy))
    assert _step(release, _game(ts=PINNED_WITH_LANE_S - 0.01, enemy=enemy)) is _STRAFE
    assert _step(release, _game(ts=PINNED_WITH_LANE_S, enemy=enemy)) is not _STRAFE


def test_pinned_against_an_enemy_is_released_only_after_the_limit(monkeypatch):
    monkeypatch.setattr("utama_core.engine.ball_release.open_teammate", lambda *a: None)
    enemy = (-CONTACT_M + 0.01, 0.0)
    release = BallRelease()
    _step(release, _game(enemy=enemy))
    assert _step(release, _game(ts=PINNED_RELEASE_S - 0.01, enemy=enemy)) is _STRAFE
    assert _step(release, _game(ts=PINNED_RELEASE_S, enemy=enemy)) is not _STRAFE


def test_an_enemy_leaving_the_ball_restarts_the_pinned_clock(no_held_limit, monkeypatch):
    monkeypatch.setattr("utama_core.engine.ball_release.open_teammate", lambda *a: None)
    enemy = (-CONTACT_M + 0.01, 0.0)
    release = BallRelease()
    _step(release, _game(enemy=enemy))
    _step(release, _game(ts=1.0, enemy=(-1.0, 0.0)))
    _step(release, _game(ts=1.1, enemy=enemy))
    assert _step(release, _game(ts=1.1 + PINNED_RELEASE_S - 0.01, enemy=enemy)) is _STRAFE


def test_an_enemy_out_of_contact_does_not_pin(no_held_limit):
    enemy = (-CONTACT_M - 0.01, 0.0)
    release = BallRelease()
    _step(release, _game(enemy=enemy))
    assert _step(release, _game(ts=5.0, enemy=enemy)) is _STRAFE


def test_never_kicks_at_our_own_goal():
    release = BallRelease()
    _step(release, _game(heading=0.0))
    assert not _step(release, _game(ts=1.0, ball=(0.0, 0.1), heading=0.0)).kick
    # past the carry limit but still dead on our goal (the ball is 1.0 up, the goal line 4.5 away)
    heading_at_goal = math.atan2(-1.0, 4.41)
    assert not _step(release, _game(ts=2.0, ball=(0.0, 1.0), heading=heading_at_goal)).kick


def test_a_heading_wide_of_our_goal_may_kick():
    release = BallRelease()
    _step(release, _game(heading=math.pi / 2))
    assert _step(release, _game(ts=1.0, ball=(0.0, 0.97), heading=math.pi / 2)).kick


def test_only_in_live_play():
    release = BallRelease()
    _step(release, _game(cmd=RefereeCommand.STOP))
    _step(release, _game(ts=1.0, ball=(0.0, 0.5), cmd=RefereeCommand.STOP))
    assert not _step(release, _game(ts=2.0, ball=(0.0, 0.97), cmd=RefereeCommand.STOP)).kick


def test_no_referee_means_no_release():
    release = BallRelease()
    game = _game()
    game.referee = None
    _step(release, game)
    far = _game(ts=1.0, ball=(0.0, 2.0))
    far.referee = None
    assert not _step(release, far).kick


def test_a_robot_with_no_command_stays_without_one():
    release = BallRelease()
    release.apply(_game(), MagicMock(), {})
    assert release.apply(_game(ts=1.0, ball=(0.0, 0.97)), MagicMock(), {}) == {}


def test_a_release_turns_away_from_the_enemy_it_is_pinned_against(no_held_limit):
    enemy = (-CONTACT_M + 0.01, 0.0)  # dead ahead of a robot facing -x
    release = BallRelease()
    _step(release, _game(enemy=enemy))
    assert _step(release, _game(ts=PINNED_RELEASE_S, enemy=enemy)) is _TURN


def test_a_release_kicks_at_the_aim_limit_even_when_not_aimed(no_held_limit):
    enemy = (-CONTACT_M + 0.01, 0.0)
    release = BallRelease()
    _step(release, _game(enemy=enemy))
    _step(release, _game(ts=PINNED_RELEASE_S, enemy=enemy))
    assert _step(release, _game(ts=PINNED_RELEASE_S + AIM_LIMIT_S - 0.01, enemy=enemy)) is _TURN
    assert _step(release, _game(ts=PINNED_RELEASE_S + AIM_LIMIT_S, enemy=enemy)).kick


def test_a_carry_aims_before_the_hard_limit_and_kicks_without_waiting_at_it():
    release = BallRelease()
    _step(release, _game(heading=0.3))
    assert _step(release, _game(ts=1.0, ball=(0.0, CARRY_AIM_M), heading=0.3)) is _TURN
    assert _step(release, _game(ts=1.1, ball=(0.0, 0.97), heading=0.3)).kick


def test_release_heading_prefers_an_open_teammate():
    game = _game(mates=[(-1.0, 2.0)])
    robot = game.friendly_robots[_ROBOT]
    assert release_heading(game, _ROBOT) == pytest.approx(robot.p.angle_to(Vector2D(-1.0, 2.0)))


def test_a_teammate_behind_an_enemy_is_not_open():
    assert open_teammate(_game(mates=[(-2.0, 0.0)], enemy=(-1.0, 0.0)), _ROBOT) is None


def test_a_teammate_inside_our_own_box_is_not_open():
    assert open_teammate(_game(mates=[(4.0, 0.0)]), _ROBOT) is None


def test_the_most_advanced_open_teammate_is_chosen():
    assert open_teammate(_game(mates=[(-1.0, 2.0), (-2.5, -2.0)]), _ROBOT) == 3


def test_a_teammate_out_of_kicking_range_is_not_open():
    robot_x = 0.09  # `_game` stands the robot just behind the ball
    assert open_teammate(_game(mates=[(robot_x - MAX_OPEN_TEAMMATE_M + 0.01, 0.0)]), _ROBOT) == 2
    assert open_teammate(_game(mates=[(robot_x - MAX_OPEN_TEAMMATE_M - 0.01, 0.0)]), _ROBOT) is None


def test_a_release_goes_round_an_enemy_dead_ahead_and_no_wider_than_it_must():
    game = _game(enemy=(-CONTACT_M + 0.01, 0.0))
    offset = abs(math.remainder(release_heading(game, _ROBOT) - math.pi, 2.0 * math.pi))
    assert offset == pytest.approx(1.05)


def test_release_with_no_open_teammate_goes_at_the_enemy_goal():
    game = _game(ball=(0.0, 2.9), enemy=(-1.0, 2.9), heading=math.pi / 2)
    robot = game.friendly_robots[_ROBOT]
    assert release_heading(game, _ROBOT) == pytest.approx(robot.p.angle_to(Vector2D(-4.5, 0.0)))


def test_a_robot_standing_on_the_ball_releases_it_only_at_the_limit():
    release = BallRelease()
    _step(release, _game())
    assert _step(release, _game(ts=HELD_STILL_S - 0.01)) is _STRAFE
    assert _step(release, _game(ts=HELD_STILL_S)).kick


def test_moving_the_ball_restarts_the_held_still_clock():
    release = BallRelease()
    _step(release, _game())
    _step(release, _game(ts=1.5, ball=(0.0, 0.06)))
    assert _step(release, _game(ts=1.5 + HELD_STILL_S - 0.01, ball=(0.0, 0.06))) is _STRAFE


def _announced(heading):
    release = BallRelease()
    ctx = TickContext(motion_controller=None, expected_receivers=set())
    release.apply(_game(heading=heading, mates=[(-2.0, 0.0)]), ctx, {_ROBOT: _STRAFE})
    release.apply(_game(ts=HELD_STILL_S, heading=heading, mates=[(-2.0, 0.0)]), ctx, {_ROBOT: _STRAFE})
    return ctx.expected_receivers


def test_a_receiver_is_told_only_once_the_passer_is_nearly_aimed():
    assert _announced(math.pi + ANNOUNCE_PASS_RAD - 0.01) == {2}
    assert _announced(math.pi + ANNOUNCE_PASS_RAD + 0.01) == set()


def test_release_heading_at_the_touchline_points_back_into_play():
    game = _game(ball=(0.0, 2.9), heading=math.pi / 2)
    assert math.sin(release_heading(game, _ROBOT)) <= 0.0


def test_release_heading_never_points_at_our_own_goal():
    game = _game(ball=(2.0, 0.0), enemy=(1.0, 0.0), heading=0.0)
    robot = game.friendly_robots[_ROBOT]
    assert not reaches_own_goal(game, robot.p, release_heading(game, _ROBOT))


def _upfield_game(enemy=None, y=0.0):
    game = _game(ball=(0.0, y), enemy=enemy)
    game.my_team_is_right = True
    return game


def test_clearance_goes_straight_when_the_lane_is_clear():
    assert upfield_heading(_upfield_game(), _ROBOT) == pytest.approx(math.pi)


def test_clearance_angles_off_a_blocked_lane():
    heading = upfield_heading(_upfield_game(enemy=(-1.5, 0.0)), _ROBOT)
    assert abs(heading - math.pi) == pytest.approx(0.26)


def test_clearance_angles_towards_the_middle_of_the_field():
    heading = upfield_heading(_upfield_game(enemy=(-0.5, 1.0), y=1.0), _ROBOT)
    assert math.sin(heading) < 0.0  # y falls towards the middle


def test_clearance_does_not_angle_off_the_field():
    # Blocked straight ahead on the touchline: the only angle left aims back into play.
    heading = upfield_heading(_upfield_game(enemy=(-0.5, 2.9), y=2.9), _ROBOT)
    assert math.sin(heading) < 0.0


def _shot_game(mates=(), enemy=None, heading=math.pi):
    game = _game(mates=mates, enemy=enemy, heading=heading)
    game.my_team_is_right = True
    return game


def _ctx():
    ctx = MagicMock()
    ctx.motion_controller.kick_bend = 0.0
    return ctx


def test_no_shot_passes_to_an_open_teammate_once_aimed(monkeypatch):
    monkeypatch.setattr(pas, "kick_bend", lambda *a: 0.0)
    assert pas.move_for_a_shot(_shot_game(mates=[(-2.0, 0.0)]), _ctx(), _ROBOT).kick


def test_no_shot_turns_towards_the_teammate_first(monkeypatch):
    monkeypatch.setattr(pas, "kick_bend", lambda *a: 0.0)
    monkeypatch.setattr(pas, "turn_on_spot", lambda **kw: _TURN)
    assert pas.move_for_a_shot(_shot_game(mates=[(-2.0, 0.0)], heading=0.3), _ctx(), _ROBOT) is _TURN


def test_no_shot_with_nobody_open_keeps_moving_for_one(monkeypatch):
    monkeypatch.setattr(pas, "kick_bend", lambda *a: 0.0)
    monkeypatch.setattr(pas, "open_teammate", lambda *a: None)
    monkeypatch.setattr(pas, "no_shot_reposition_target", lambda *a: Vector2D(0.0, 0.6))
    monkeypatch.setattr(pas, "move", lambda **kw: _STRAFE)
    assert pas.move_for_a_shot(_shot_game(), _ctx(), _ROBOT) is _STRAFE
