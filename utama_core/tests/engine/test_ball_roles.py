"""`BallRoles` and the shared helpers it and the tactics lean on: who deals with the ball."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from utama_core.config.physical_constants import ROBOT_RADIUS
from utama_core.engine.ball_roles import COVER_M, ON_BALL_SWITCH_MARGIN_M, BallRoles
from utama_core.engine.context import TickContext
from utama_core.entities.data.command import RobotCommand
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.geometry.pass_and_score_geometry import (
    _DEEP_NEAR_LINE_M,
    _INTERCEPT_MIN_M,
    _PASS_ROLLING_MPS,
    ball_intercept_point,
    no_shot_reposition_target,
    receiver_of_rolling_ball,
    segment_blocked,
)
from utama_core.referee.actions import (
    _WALL_FROM_BALL_M,
    _is_in_own_defense_area,
    _next_is_their_free_kick,
    their_free_kick_targets,
)
from utama_core.skills.man_mark import MARK_STANDOFF, mark_target
from utama_core.tactics.give_and_go import _relocate_target

_ORIGINAL = RobotCommand(local_forward_vel=1, local_left_vel=0, angular_vel=0, kick=0, chip=0, dribble=0)
_TURN = RobotCommand(local_forward_vel=0, local_left_vel=0, angular_vel=2.0, kick=0, chip=0, dribble=1)


def _robot(x, y, has_ball=False, orientation=0.0):
    return SimpleNamespace(p=Vector2D(x, y), orientation=orientation, has_ball=has_ball)


def _game(friends, enemies=(), ball=(0.0, 0.0), v=(0.0, 0.0), cmd=RefereeCommand.NORMAL_START):
    """Our goal at +x, attacking -x. `friends`/`enemies` are robots or (x, y); friend ids are their indices."""
    make = lambda items: {i: (r if isinstance(r, SimpleNamespace) else _robot(*r)) for i, r in enumerate(items)}  # noqa
    return SimpleNamespace(
        ts=0.0,
        ball=SimpleNamespace(p=SimpleNamespace(x=ball[0], y=ball[1], to_2d=lambda: Vector2D(*ball)), v=Vector2D(*v)),
        referee=SimpleNamespace(referee_command=cmd),
        friendly_robots=make(friends),
        enemy_robots=make(enemies),
        my_team_is_right=True,
        field=SimpleNamespace(
            my_goal_line=np.array([[4.5, 0.5], [4.5, -0.5]]),
            enemy_goal_line=np.array([[-4.5, 0.5], [-4.5, -0.5]]),
            my_defense_area=np.array([[4.5, 1.0], [3.5, 1.0], [3.5, -1.0], [4.5, -1.0]]),
            enemy_defense_area=np.array([[-4.5, 1.0], [-3.5, 1.0], [-3.5, -1.0], [-4.5, -1.0]]),
            half_length=4.5,
            half_width=3.0,
            half_defense_area_width=1.0,
            half_defense_area_depth=0.5,
            defense_area_rules=True,
        ),
    )


@pytest.fixture
def moves(monkeypatch):
    calls = {}

    def _move(**kw):
        calls[kw["robot_id"]] = kw
        return RobotCommand(local_forward_vel=0, local_left_vel=0, angular_vel=0, kick=0, chip=0, dribble=9)

    monkeypatch.setattr("utama_core.engine.ball_roles.move", _move)
    monkeypatch.setattr("utama_core.engine.ball_roles.turn_on_spot", lambda **kw: _TURN)
    return calls


def _ctx(claims=(), expected=()):
    return TickContext(motion_controller=None, ball_claims=set(claims), expected_receivers=set(expected))


def _commands(n):
    return {i: _ORIGINAL for i in range(n)}


# --- one robot on the ball ---------------------------------------------------------------


def test_of_two_robots_driving_at_the_ball_the_further_covers(moves):
    game = _game([(1.0, 0.0), (0.5, 0.0)])
    commands = BallRoles().apply(game, _ctx(claims=[0, 1]), _commands(2), [0, 1])
    assert commands[1] is _ORIGINAL
    assert commands[0] is not _ORIGINAL
    assert moves[0]["target_coords"].distance_to(Vector2D(COVER_M, 0.0)) < 1e-6  # towards our goal at +x


def test_a_single_claim_is_left_alone(moves):
    assert BallRoles().apply(_game([(1.0, 0.0)]), _ctx(claims=[0]), _commands(1), [0])[0] is _ORIGINAL


def test_the_robot_on_the_ball_keeps_it_until_another_is_clearly_nearer(moves):
    roles = BallRoles()
    roles.apply(_game([(0.5, 0.0), (1.0, 0.0)]), _ctx(claims=[0, 1]), _commands(2), [0, 1])
    just_short = 1.0 - ON_BALL_SWITCH_MARGIN_M + 0.01
    assert (
        roles.apply(_game([(1.0, 0.0), (just_short, 0.0)]), _ctx(claims=[0, 1]), _commands(2), [0, 1])[0] is _ORIGINAL
    )
    clearly = 1.0 - ON_BALL_SWITCH_MARGIN_M - 0.01
    assert roles.apply(_game([(1.0, 0.0), (clearly, 0.0)]), _ctx(claims=[0, 1]), _commands(2), [0, 1])[1] is _ORIGINAL


def test_a_teammate_holding_the_ball_sends_a_claimer_to_cover(moves):
    game = _game([_robot(0.1, 0.0, has_ball=True), (1.0, 0.0)])
    assert BallRoles().apply(game, _ctx(claims=[1]), _commands(2), [0, 1])[1] is not _ORIGINAL


def test_a_claimer_near_a_held_ball_steps_straight_out_and_is_not_pulled_back(moves):
    game = _game([_robot(0.1, 0.0, has_ball=True), (-0.5, 0.0)])  # the attacker is on the far side of the ball
    BallRoles().apply(game, _ctx(claims=[1]), _commands(2), [0, 1])
    assert moves[1]["target_coords"].distance_to(Vector2D(-COVER_M, 0.0)) < 1e-6


def test_a_claimer_far_from_a_held_ball_stays_where_it_is(moves):
    game = _game([_robot(0.1, 0.0, has_ball=True), (-2.0, 1.0)])
    BallRoles().apply(game, _ctx(claims=[1]), _commands(2), [0, 1])
    assert moves[1]["target_coords"].distance_to(Vector2D(-2.0, 1.0)) < 1e-6


def test_the_closest_robot_is_not_made_to_cover_a_loose_ball(moves):
    roles = BallRoles()
    roles.apply(_game([(0.5, 0.0), (1.0, 0.0)]), _ctx(claims=[0, 1]), _commands(2), [0, 1])
    # 1 has become closest but not by the margin that moves the ball to it: both carry on
    commands = roles.apply(_game([(0.8, 0.0), (0.7, 0.0)]), _ctx(claims=[0, 1]), _commands(2), [0, 1])
    assert commands[0] is _ORIGINAL and commands[1] is _ORIGINAL


def test_nothing_changes_outside_live_play(moves):
    game = _game([(1.0, 0.0), (0.5, 0.0)], cmd=RefereeCommand.STOP)
    assert BallRoles().apply(game, _ctx(claims=[0, 1]), _commands(2), [0, 1])[0] is _ORIGINAL


# --- receiving ---------------------------------------------------------------------------


def test_the_robot_a_free_ball_rolls_to_meets_it_on_its_path(moves):
    game = _game([(2.0, 0.3), (-2.0, 0.0)], v=(3.0, 0.0))
    commands = BallRoles().apply(game, _ctx(), _commands(2), [0, 1])
    assert commands[0] is not _ORIGINAL and commands[1] is _ORIGINAL
    assert moves[0]["dribbling"] is True
    assert abs(moves[0]["target_coords"].y) < 1e-6


def test_a_passer_chasing_its_own_pass_covers_while_the_receiver_takes_it(moves):
    game = _game([(2.0, 0.0), (-0.5, 0.0)], v=(3.0, 0.0))
    commands = BallRoles().apply(game, _ctx(claims=[1]), _commands(2), [0, 1])
    assert commands[1] is not _ORIGINAL
    assert moves[0]["dribbling"] is True


def test_a_receiver_stays_on_the_ball_when_the_reach_check_drops_it(moves):
    roles = BallRoles()
    roles.apply(_game([(1.0, 0.1)], v=(3.0, 0.0)), _ctx(), _commands(1), [0])
    assert receiver_of_rolling_ball(_game([(1.0, 0.6)], v=(3.0, 0.0)), [0]) is None
    assert roles.apply(_game([(1.0, 0.6)], v=(3.0, 0.0)), _ctx(), _commands(1), [0])[0] is not _ORIGINAL


def test_a_receiver_lets_go_once_the_ball_has_passed(moves):
    roles = BallRoles()
    roles.apply(_game([(1.0, 0.1)], v=(3.0, 0.0)), _ctx(), _commands(1), [0])
    assert roles.apply(_game([(-0.5, 0.1)], v=(3.0, 0.0)), _ctx(), _commands(1), [0])[0] is _ORIGINAL


def test_a_receiver_sprints_onto_the_path(moves):
    BallRoles().apply(_game([(1.0, 0.1)], v=(3.0, 0.0)), _ctx(), _commands(1), [0])
    assert moves[0]["sprint"] is True


def test_a_new_pick_replaces_a_robot_the_first_tick_chose(moves):
    roles = BallRoles()
    roles.apply(_game([(1.0, 0.1), (3.0, 3.0)], v=(3.0, 0.0)), _ctx(), _commands(2), [0, 1])
    assert roles._receiving == 0
    # robot 0 is still within the hold's reach of the path, but robot 1 is the one the ball comes to
    roles.apply(_game([(1.0, 0.6), (0.5, 0.05)], v=(3.0, 0.0)), _ctx(), _commands(2), [0, 1])
    assert roles._receiving == 1


def test_a_ball_an_enemy_reaches_first_is_not_received():
    assert receiver_of_rolling_ball(_game([(2.0, 0.0)], [(1.0, 0.05)], v=(3.0, 0.0)), [0]) is None
    assert receiver_of_rolling_ball(_game([(1.0, 0.0)], [(2.0, 0.05)], v=(3.0, 0.0)), [0]) == 0


def test_only_the_first_robot_along_the_path_receives():
    game = _game([(3.0, 0.0), (1.5, 0.0)], v=(3.0, 0.0))
    assert receiver_of_rolling_ball(game, [0, 1]) == 1


def test_a_shot_at_the_enemy_goal_is_not_received():
    assert receiver_of_rolling_ball(_game([(-2.0, 0.0)], v=(-3.0, 0.0)), [0]) is None


def test_a_slow_ball_is_not_received():
    game = _game([(0.5, 0.0)], v=(_PASS_ROLLING_MPS - 0.01, 0.0))
    assert receiver_of_rolling_ball(game, [0]) is None


def test_a_ball_at_a_robots_feet_is_not_a_free_ball():
    assert receiver_of_rolling_ball(_game([(2.0, 0.0)], [(0.1, 0.0)], v=(3.0, 0.0)), [0]) is None


def test_a_ball_that_stops_short_is_not_received():
    speed = 1.0  # rolls 1 / (2 * 0.3) = 1.67 m
    assert receiver_of_rolling_ball(_game([(1.6, 0.0)], v=(speed, 0.0)), [0]) == 0
    assert receiver_of_rolling_ball(_game([(1.75, 0.0)], v=(speed, 0.0)), [0]) is None


def test_a_receiver_out_of_reach_of_the_path_is_passed_over():
    assert receiver_of_rolling_ball(_game([(0.6, 0.19)], v=(6.0, 0.0)), [0]) == 0  # reach 0.1 + 0.1
    assert receiver_of_rolling_ball(_game([(0.6, 0.21)], v=(6.0, 0.0)), [0]) is None


def test_the_robot_about_to_be_passed_to_stops_and_faces_the_ball(moves):
    commands = BallRoles().apply(_game([(2.0, 0.3)]), _ctx(expected=[0]), _commands(1), [0])
    assert commands[0] is _TURN


# --- going to where the ball will be -----------------------------------------------------


def test_a_chaser_heads_for_where_it_can_catch_the_ball():
    spot = ball_intercept_point(_game([(1.0, 1.0)], v=(1.0, 0.0)), 0)
    assert 0.5 < spot.x < 1.67 and spot.y == pytest.approx(0.0)


def test_a_chaser_that_cannot_catch_it_goes_to_where_it_stops():
    spot = ball_intercept_point(_game([(-3.0, 0.0)], v=(1.0, 0.0)), 0)
    assert spot.x == pytest.approx(1.0 / (2 * 0.3), abs=0.01)


def test_no_intercept_for_a_slow_ball_or_a_robot_beside_it():
    assert ball_intercept_point(_game([(1.0, 1.0)], v=(_PASS_ROLLING_MPS - 0.01, 0.0)), 0) is None
    assert ball_intercept_point(_game([(_INTERCEPT_MIN_M - 0.01, 0.0)], v=(3.0, 0.0)), 0) is None


def test_the_intercept_stays_out_of_our_box():
    spot = ball_intercept_point(_game([(-3.0, 0.0)], ball=(2.5, 0.0), v=(3.0, 0.0)), 0)
    assert spot.x < 3.5


# --- releases, corners, supporters, marking ----------------------------------------------


def _reposition(x, y, enemy):
    return no_shot_reposition_target(Vector2D(x, y), [enemy], -4.5, -0.3, 0.3, 3.0)


def test_deep_and_wide_of_the_goal_a_carrier_strafes_towards_the_middle():
    deep = -4.5 + _DEEP_NEAR_LINE_M - 0.01
    assert _reposition(deep, 2.0, Vector2D(deep, 1.8)).y < 2.0  # the blocker would send it to the corner
    assert _reposition(deep, -2.0, Vector2D(deep, -1.8)).y > -2.0


def test_short_of_the_deep_zone_a_carrier_still_strafes_away_from_the_blocker():
    shallow = -4.5 + _DEEP_NEAR_LINE_M + 0.01
    assert _reposition(shallow, 2.0, Vector2D(shallow, 1.8)).y > 2.0


def test_a_support_run_with_a_lane_beats_a_deeper_one_without():
    open_field = _relocate_target(_game([(-1.0, 1.2), (0.1, 0.0)]), 0, [])
    blocker = (open_field.x * 0.5, open_field.y * 0.5)
    chosen = _relocate_target(_game([(-1.0, 1.2), (0.1, 0.0)], [blocker]), 0, [])
    assert chosen.distance_to(open_field) > 0.1
    assert not segment_blocked(Vector2D(0.0, 0.0), chosen, [Vector2D(*blocker)], ROBOT_RADIUS + 0.1)


def _facing(angle):
    return _robot(0.05, 0.0, orientation=angle)


def test_a_marker_steps_into_the_lane_when_the_passer_faces_its_opponent():
    game = _game([], [_facing(0.0), (2.0, 0.0)])
    target = mark_target(game, 1)
    assert target.x == pytest.approx(2.0 - MARK_STANDOFF)  # between the passer and the opponent


def test_a_marker_stays_goal_side_when_the_passer_looks_elsewhere():
    game = _game([], [_facing(0.21), (2.0, 0.0)])
    assert mark_target(game, 1).x == pytest.approx(2.0 + MARK_STANDOFF)


def test_a_marker_stays_goal_side_when_the_passer_is_too_near_to_step_between():
    game = _game([], [_facing(0.0), (1.2, 0.0)])
    assert mark_target(game, 1).x == pytest.approx(1.2 + MARK_STANDOFF)


# --- an opponent's free kick -------------------------------------------------------------


def _free_kick_game(friends, enemies, ball):
    game = _game(friends, enemies, ball=ball)
    game.my_team_is_yellow = True
    game.referee.yellow_team = SimpleNamespace(goalkeeper=0)
    game.referee.next_command = RefereeCommand.DIRECT_FREE_BLUE
    return game


def test_their_free_kick_puts_a_wall_on_the_line_and_marks_the_receivers():
    game = _free_kick_game([(4.2, 0.0), (1.0, 0.5), (0.0, 2.0)], [(-0.15, 0.0), (0.0, -2.0), (2.5, 2.0)], (0.0, 0.0))
    targets = their_free_kick_targets(game)
    assert 0 not in targets  # the keeper keeps its own job
    assert targets[1].distance_to(Vector2D(_WALL_FROM_BALL_M, 0.0)) < 1e-6
    assert targets[2].distance_to(Vector2D(2.5 + MARK_STANDOFF, 2.0)) < 1e-6  # nearest our goal first, not the taker


def test_the_wall_stays_out_of_our_own_box():
    game = _free_kick_game([(4.2, 0.0), (2.0, 0.5)], [(2.85, 0.0)], (3.0, 0.0))
    assert not _is_in_own_defense_area(game, *their_free_kick_targets(game)[1])


def test_stop_only_prepares_for_a_free_kick_that_is_theirs():
    game = _free_kick_game([(4.2, 0.0)], [], (0.0, 0.0))
    assert _next_is_their_free_kick(game)
    game.referee.next_command = RefereeCommand.DIRECT_FREE_YELLOW
    assert not _next_is_their_free_kick(game)
