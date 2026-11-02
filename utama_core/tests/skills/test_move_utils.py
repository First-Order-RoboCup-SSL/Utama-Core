import math
from unittest.mock import MagicMock

import pytest

from utama_core.config.field_params import STANDARD_FIELD_DIMS
from utama_core.config.physical_constants import BALL_RADIUS, ROBOT_RADIUS
from utama_core.entities.data.vector import Vector2D, Vector3D
from utama_core.entities.game import Ball, Field, Game, GameFrame, GameHistory, Robot
from utama_core.motion_planning.src.common.pose_ahead import PoseAhead
from utama_core.skills.src.utils.move_utils import move, turn_on_spot

PIVOT_RADIUS = ROBOT_RADIUS + BALL_RADIUS


def _robot(rid: int, x: float, y: float, is_friendly: bool, orientation: float = 0.0, has_ball: bool = False) -> Robot:
    return Robot(
        id=rid,
        is_friendly=is_friendly,
        has_ball=has_ball,
        p=Vector2D(x, y),
        v=Vector2D(0, 0),
        a=Vector2D(0, 0),
        orientation=orientation,
    )


def _game(friendly: dict, enemy: dict, ball_xy: tuple) -> Game:
    zv = Vector3D(0, 0, 0)
    frame = GameFrame(
        ts=0.0,
        my_team_is_yellow=True,
        my_team_is_right=True,
        friendly_robots=friendly,
        enemy_robots=enemy,
        ball=Ball(p=Vector3D(ball_xy[0], ball_xy[1], 0), v=zv, a=zv),
    )
    return Game(
        past=GameHistory(10),
        current=frame,
        field=Field(
            my_team_is_right=True, field_dims=STANDARD_FIELD_DIMS, field_bounds=STANDARD_FIELD_DIMS.full_field_bounds
        ),
    )


def _motion_controller(angular_vel: float) -> MagicMock:
    mc = MagicMock()
    mc.min_turn = None
    mc.calculate.return_value = (Vector2D(0.0, 0.0), angular_vel)
    return mc


def test_turn_on_spot_pivots_around_ball_when_clear_of_obstacles():
    # Robot facing +x, holding the ball, turning left (positive angular_vel).
    # Local-left pivot compensation should be the standard -angular_vel * PIVOT_RADIUS,
    # with no nearby enemy to suppress it.
    friendly = {1: _robot(1, 0.0, 0.0, True, orientation=0.0, has_ball=True)}
    enemy = {0: _robot(0, 5.0, 5.0, False)}  # far away
    game = _game(friendly, enemy, (ROBOT_RADIUS + BALL_RADIUS, 0.0))
    mc = _motion_controller(angular_vel=1.0)

    cmd = turn_on_spot(game=game, motion_controller=mc, robot_id=1, target_oren=math.pi / 2)

    assert cmd.local_left_vel == pytest.approx(-1.0 * PIVOT_RADIUS)


def test_turn_on_spot_steers_away_from_a_wedged_enemy():
    """Reproduces the traced COMMITTED_FROZEN deadlock from
    split_shape_vs_switch_of_play_RK: a robot pivoting on the ball while
    body-to-body with an enemy kept commanding a lateral push straight into
    the enemy every tick. Box2D's contact resolution cancelled the motion,
    the geometry never changed, and the same command (and freeze) recurred
    forever.

    Robot faces +x at the origin, holding the ball. For angular_vel=+1.0 the
    pivot compensation (-angular_vel * PIVOT_RADIUS, local-left) is negative,
    i.e. global -y for this orientation. Placing the enemy at (0, -small)
    puts it directly in that push direction and within contact distance.

    Merely zeroing `local_left_vel` here (an earlier fix attempt) does not
    produce a working turn: rotating around the ball inherently requires the
    chassis to sweep an arc through space, and dropping only the model's own
    lateral estimate leaves that physical requirement unmet, so rsim's
    contact solver had to supply/oppose the sweep itself, which measured live
    as ZERO net motion for the rest of the match. The actual fix steers
    `move()`'s own hold-position target away from the blocking enemy instead,
    so the motion controller (mocked here) is asked to plan from a clear
    starting point rather than being told to hold in place while suppressing
    its own reported command -- this test only has visibility into
    `motion_controller.calculate`'s `target_pos` argument, not into rsim's
    actual contact resolution.
    """
    friendly = {1: _robot(1, 0.0, 0.0, True, orientation=0.0, has_ball=True)}
    enemy = {0: _robot(0, 0.0, -0.15, False)}  # within 2*ROBOT_RADIUS, directly along the pivot push
    game = _game(friendly, enemy, (ROBOT_RADIUS + BALL_RADIUS, 0.0))
    mc = _motion_controller(angular_vel=1.0)  # pivot push is -y (into the enemy)

    cmd = turn_on_spot(game=game, motion_controller=mc, robot_id=1, target_oren=math.pi / 2)

    assert cmd.angular_vel == pytest.approx(1.0)  # rotation itself is preserved
    # The final move() call (whose mocked angular_vel drives local_left_vel's
    # re-derivation) was planned from a target steered away from the enemy,
    # not held at the robot's own (blocked) position.
    final_call_kwargs = mc.calculate.call_args_list[-1].kwargs
    assert final_call_kwargs["target_pos"].y > 0.0  # steered away from the enemy at y=-0.15


def test_turn_on_spot_keeps_pivot_push_away_from_a_wedged_enemy():
    """Same wedged geometry, but turning the other way -- the pivot push
    points away from the enemy (global +y), so it should NOT be suppressed.
    """
    friendly = {1: _robot(1, 0.0, 0.0, True, orientation=0.0, has_ball=True)}
    enemy = {0: _robot(0, 0.0, -0.15, False)}
    game = _game(friendly, enemy, (ROBOT_RADIUS + BALL_RADIUS, 0.0))
    mc = _motion_controller(angular_vel=-1.0)  # pivot push is +y (away from the enemy)

    cmd = turn_on_spot(game=game, motion_controller=mc, robot_id=1, target_oren=-math.pi / 2)

    assert cmd.local_left_vel == pytest.approx(1.0 * PIVOT_RADIUS)


def test_turn_on_spot_steers_away_from_a_wedged_teammate():
    """A teammate's body blocks the pivot exactly as an enemy's does. Found in
    high_line_zone_vs_low_block (2026-09-28): the overload decoy pivoting on the
    ball with the switch carrier 0.219 m away in its push direction was commanded
    ~1.4 rad/s for 8 s and never turned, freezing the match."""
    friendly = {
        1: _robot(1, 0.0, 0.0, True, orientation=0.0, has_ball=True),
        2: _robot(2, 0.0, -0.15, True),  # within 2*ROBOT_RADIUS, directly along the pivot push
    }
    enemy = {0: _robot(0, 5.0, 5.0, False)}
    game = _game(friendly, enemy, (ROBOT_RADIUS + BALL_RADIUS, 0.0))
    mc = _motion_controller(angular_vel=1.0)  # pivot push is -y (into the teammate)

    turn_on_spot(game=game, motion_controller=mc, robot_id=1, target_oren=math.pi / 2)

    final_call_kwargs = mc.calculate.call_args_list[-1].kwargs
    assert final_call_kwargs["target_pos"].y > 0.0  # steered away from the teammate at y=-0.15


_CARRY_SPEED, _CARRY_TURN = 0.3, 0.6  # REAL_PARAMS's


@pytest.mark.parametrize(
    "velocity, angular_vel, scale",
    [
        ((1.0, 0.0), 0.2, _CARRY_SPEED / 1.0),  # too fast: the speed binds
        ((0.1, 0.0), 4.0, _CARRY_TURN / 4.0),  # turning too hard: the turn binds (2026-10-28: 4 rad/s lost the ball)
        ((-0.6, 0.8), 3.0, min(_CARRY_SPEED / 1.0, _CARRY_TURN / 3.0)),  # both: the tighter one, for both
        ((0.1, 0.1), 0.5, 1.0),  # within both: unchanged
    ],
)
def test_move_holds_a_robot_with_the_ball_to_the_carry_limits(velocity, angular_vel, scale):
    """Dribbling with the ball, `move` scales the whole command down by one factor, so the
    robot's path and turn keep their shape and only slow down."""
    cmd = move(**_carry_case(velocity, angular_vel, has_ball=True, dribbling=True))
    assert cmd.local_forward_vel == pytest.approx(velocity[0] * scale)
    assert cmd.local_left_vel == pytest.approx(velocity[1] * scale)
    assert cmd.angular_vel == pytest.approx(angular_vel * scale)
    assert math.hypot(cmd.local_forward_vel, cmd.local_left_vel) <= _CARRY_SPEED + 1e-9
    assert abs(cmd.angular_vel) <= _CARRY_TURN + 1e-9


@pytest.mark.parametrize(
    "has_ball, dribbling, limits",
    [
        (False, True, (_CARRY_SPEED, _CARRY_TURN)),
        (True, False, (_CARRY_SPEED, _CARRY_TURN)),
        (True, True, (None, None)),
    ],
)
def test_move_is_not_limited_without_the_ball_without_dribbling_or_in_the_sims(has_ball, dribbling, limits):
    """Fetching the ball (no ball yet), moving without the dribbler, and every sim (no limits in
    `RSIM_PARAMS`/`GRSIM_PARAMS`) keep the full command."""
    cmd = move(**_carry_case((1.0, 0.0), 4.0, has_ball=has_ball, dribbling=dribbling, limits=limits))
    assert (cmd.local_forward_vel, cmd.local_left_vel, cmd.angular_vel) == pytest.approx((1.0, 0.0, 4.0))


def test_a_robot_with_its_own_carry_limits_is_held_to_those_not_the_teams():
    """`RobotProfile.carry_speed`/`carry_turn` (via `robot_carry`) replace the team's for that robot only."""
    case = _carry_case((1.0, 0.0), 0.2, has_ball=True, dribbling=True)
    case["motion_controller"].robot_carry = {1: (0.6, 2.0), 2: (0.1, 0.1)}
    cmd = move(**case)
    assert (cmd.local_forward_vel, cmd.angular_vel) == pytest.approx((0.6, 0.2 * 0.6))


def _carry_case(velocity, angular_vel, has_ball, dribbling, limits=(_CARRY_SPEED, _CARRY_TURN)) -> dict:
    game = _game(
        {1: _robot(1, 0.0, 0.0, True, orientation=0.0, has_ball=has_ball)}, {}, (ROBOT_RADIUS + BALL_RADIUS, 0.0)
    )
    mc = MagicMock()
    mc.pose_ahead = PoseAhead(0.0)
    mc.carry_speed, mc.carry_turn = limits
    mc.robot_carry = {}
    mc.min_turn = None
    mc.calculate.return_value = (Vector2D(*velocity), angular_vel)
    return dict(
        game=game,
        motion_controller=mc,
        robot_id=1,
        target_coords=Vector2D(1.0, 0.0),
        target_oren=0.0,
        dribbling=dribbling,
    )


def test_real_robots_have_carry_limits_and_the_sims_do_not():
    from utama_core.config.robot_params import GRSIM_PARAMS, REAL_PARAMS, RSIM_PARAMS

    assert (REAL_PARAMS.CARRY_SPEED, REAL_PARAMS.CARRY_TURN) == (_CARRY_SPEED, _CARRY_TURN)
    assert (
        RSIM_PARAMS.CARRY_SPEED is RSIM_PARAMS.CARRY_TURN is GRSIM_PARAMS.CARRY_SPEED is GRSIM_PARAMS.CARRY_TURN is None
    )


@pytest.mark.parametrize("controller", ["fpp", "pid"])
def test_a_robot_that_loses_the_ball_ramps_from_the_carry_speed_it_was_sent(controller):
    """Carrying the ball to its right, `move` sends 0.3 m/s although the controller asked for
    ~1 m/s. When the ball is lost and the target swings ahead, the next command ramps from the
    0.3 m/s sent: ramping from the 1 m/s never sent lurched the real robot 0.9 m/s sideways
    (dribble_and_shoot, 2026-10-28)."""
    from utama_core.config.enums import Mode
    from utama_core.motion_planning.src.controllers.fastpathplanning import (
        FastPathPlanningController,
    )
    from utama_core.motion_planning.src.controllers.pid_controller import PIDController

    mc = {"fpp": FastPathPlanningController, "pid": PIDController}[controller](Mode.REAL)
    mc.pose_ahead = PoseAhead(0.0)
    ball = (ROBOT_RADIUS + BALL_RADIUS, 0.0)
    with_ball = _game({1: _robot(1, 0.0, 0.0, True, has_ball=True)}, {}, ball)
    for _ in range(30):  # long enough for the controller's own output to reach full speed
        carried = move(with_ball, mc, 1, Vector2D(0.0, -2.0), 0.0, dribbling=True)
    assert math.hypot(carried.local_forward_vel, carried.local_left_vel) == pytest.approx(0.3)

    lost = _game({1: _robot(1, 0.0, 0.0, True, has_ball=False)}, {}, ball)
    cmd = move(lost, mc, 1, Vector2D(1.0, 0.0), 0.0, dribbling=True)

    assert cmd.local_left_vel >= -0.3 - 1e-9  # no faster to the right than it was going


# --- the slowest turn the real robots make ---

_MIN_TURN = 0.3


def _turn_case(orientation: float, target_oren: float, angular_vel: float, min_turn) -> dict:
    game = _game({1: _robot(1, 0.0, 0.0, True, orientation=orientation)}, {}, (1.0, 1.0))
    mc = MagicMock()
    mc.pose_ahead = PoseAhead(0.0)
    mc.carry_speed = mc.carry_turn = None
    mc.robot_carry = {}
    mc.min_turn = min_turn
    mc.calculate.return_value = (Vector2D(0.0, 0.0), angular_vel)
    return dict(game=game, motion_controller=mc, robot_id=1, target_coords=Vector2D(0.0, 0.0), target_oren=target_oren)


@pytest.mark.parametrize(
    "orientation, target_oren, angular_vel, expected",
    [
        (0.0, 0.05, 0.1, _MIN_TURN),  # 0.05 rad short, asked 0.1 rad/s: the slowest real turn
        (0.0, -0.05, -0.1, -_MIN_TURN),  # the other way
        (0.0, -0.05, 0.0, -_MIN_TURN),  # asked nothing: towards the target all the same
        (3.1, -3.1, 0.0, _MIN_TURN),  # across +-pi: the short way round, anticlockwise
        (0.0, 0.011, 0.0, _MIN_TURN),  # just past the 0.01 rad it may stop within
        (0.0, 0.009, 0.05, 0.05),  # within it: as asked
        (0.0, 0.5, 2.0, 2.0),  # asked faster: as asked
    ],
)
def test_a_real_robot_off_its_orientation_turns_at_least_its_slowest_turn(
    orientation, target_oren, angular_vel, expected
):
    """2026-10-28 passing rallies: asked 0.1 rad/s, a passer stood still 3-4 deg short of its
    aim and kicked at the aim tolerance's edge."""
    case = _turn_case(orientation, target_oren, angular_vel, _MIN_TURN)
    assert move(**case).angular_vel == pytest.approx(expected)


def test_the_raised_turn_is_what_the_controller_ramps_from():
    case = _turn_case(0.0, 0.05, 0.1, _MIN_TURN)
    move(**case)
    case["motion_controller"].sent.assert_called_once()
    assert case["motion_controller"].sent.call_args.args[2] == pytest.approx(_MIN_TURN)


def test_the_sims_turn_as_asked():
    assert move(**_turn_case(0.0, 0.05, 0.1, None)).angular_vel == pytest.approx(0.1)


def test_only_real_robots_have_a_slowest_turn():
    from utama_core.config.robot_params import GRSIM_PARAMS, REAL_PARAMS, RSIM_PARAMS

    assert REAL_PARAMS.MIN_TURN == _MIN_TURN
    assert RSIM_PARAMS.MIN_TURN is GRSIM_PARAMS.MIN_TURN is None
