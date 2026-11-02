"""`trajsample` plans with the robot's weaker sideways acceleration (`Trajectory2D.compute`'s
`a_sideways`, `RobotParams.MAX_SIDEWAYS_ACCELERATION`)."""

import math

import pytest

from utama_core.config.enums import Mode
from utama_core.config.robot_params import REAL_PARAMS, RSIM_PARAMS
from utama_core.motion_planning.src.controllers.trajsampling import (
    TrajectorySamplingController,
)
from utama_core.motion_planning.src.trajsampling.bang_bang import Trajectory2D
from utama_core.motion_planning.src.trajsampling.planner import (
    TrajectorySamplingPlanner,
)
from utama_core.tests.fixtures.game_builder import build_game, make_ball, make_robot

A_MAX, A_SIDEWAYS = 4.0, 1.8


def _accel(angle: float, heading: float = 0.0, a_sideways=A_SIDEWAYS) -> float:
    """The acceleration a move 1 m along `angle` is planned with."""
    end = (math.cos(angle), math.sin(angle))
    return Trajectory2D.compute((0.0, 0.0), (0.0, 0.0), end, 2.0, A_MAX, a_sideways, heading).along.a_max


def test_straight_ahead_and_straight_back_use_the_full_acceleration():
    assert _accel(0.0) == A_MAX
    assert _accel(math.pi) == pytest.approx(A_MAX)


@pytest.mark.parametrize("angle", [math.pi / 2, -math.pi / 2])
def test_straight_sideways_uses_the_sideways_acceleration(angle):
    assert _accel(angle) == pytest.approx(A_SIDEWAYS)


def test_a_slant_is_limited_by_its_share_across_the_robot():
    across = math.asin(A_SIDEWAYS / A_MAX)  # the angle where the sideways limit starts to bind
    assert _accel(across - 0.01) == A_MAX
    assert _accel(across + 0.01) == pytest.approx(A_SIDEWAYS / math.sin(across + 0.01))
    assert _accel(across + 0.01) < A_MAX


def test_the_axis_follows_the_heading():
    """Facing +y, a move along +y is straight ahead and one along +x is sideways."""
    assert _accel(math.pi / 2, heading=math.pi / 2) == pytest.approx(A_MAX)
    assert _accel(0.0, heading=math.pi / 2) == pytest.approx(A_SIDEWAYS)
    assert _accel(0.0, heading=-3 * math.pi / 2) == pytest.approx(A_SIDEWAYS)  # same heading, other wrap


def test_without_a_sideways_limit_every_direction_is_the_same():
    assert _accel(math.pi / 2, a_sideways=None) == A_MAX
    assert _accel(math.pi / 2, a_sideways=A_MAX + 1.0) == A_MAX  # a stronger sideways axis changes nothing


def test_the_planner_plans_with_the_robot_s_own_heading():
    planner = TrajectorySamplingPlanner(v_max=2.0, a_max=A_MAX, a_sideways=A_SIDEWAYS)
    game = build_game({0: make_robot(0, is_friendly=True, x=0.0, y=0.0, orientation=0.0)}, {}, make_ball(x=3.0, y=-2.0))
    sideways = planner.plan(game, 0, (0.0, 1.0), game.field.field_bounds)
    assert sideways.trajectory.along.a_max == pytest.approx(A_SIDEWAYS)
    ahead = planner.plan(game, 0, (1.0, 0.0), game.field.field_bounds)
    assert ahead.trajectory.along.a_max == pytest.approx(A_MAX)


def test_only_the_real_robots_have_a_weaker_sideways_axis():
    assert RSIM_PARAMS.MAX_SIDEWAYS_ACCELERATION is None
    assert TrajectorySamplingController(Mode.RSIM).planner.a_sideways is None
    assert TrajectorySamplingController(Mode.REAL).planner.a_sideways == REAL_PARAMS.MAX_SIDEWAYS_ACCELERATION
    assert REAL_PARAMS.MAX_SIDEWAYS_ACCELERATION < REAL_PARAMS.MAX_ACCELERATION
