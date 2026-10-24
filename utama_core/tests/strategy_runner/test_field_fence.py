"""Robots on camera runs are kept inside the area in play: a tactic chasing a ball out of the
camera's half drove its robot out of sight (a grSim test, 0.9 m past the half-way line)."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from utama_core.config.enums import Mode
from utama_core.config.field_params import STANDARD_FIELD_DIMS
from utama_core.entities.data.command import RobotCommand
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game.field import FieldBounds
from utama_core.run.field_fence import (
    BRAKE,
    DELAY,
    FencedRobotController,
    _max_speed_towards,
    fence,
    inner_edges,
)
from utama_core.run.strategy_runner import StrategyRunner
from utama_core.team_controller.src.controllers.common.robot_profile import RobotProfile

LEFT_HALF = FieldBounds(top_left=(-2.0, 1.5), bottom_right=(0.0, -1.5))


def _cmd(forward=0.0, left=0.0, turn=0.0) -> RobotCommand:
    return RobotCommand(forward, left, turn, kick=False, chip=False, dribble=False)


@pytest.mark.parametrize("distance", [0.01, 0.1, 0.5, 2.0])
def test_the_speed_cap_is_what_stops_a_robot_exactly_at_the_line(distance):
    v = _max_speed_towards(distance)
    assert v * DELAY + v * v / (2 * BRAKE) == pytest.approx(distance)


def test_far_from_the_lines_a_command_is_unchanged():
    command = _cmd(1.0, 0.3, 2.0)
    assert fence(-1.0, 0.0, 0.0, command, LEFT_HALF) is command


def test_towards_the_half_way_line_the_robot_slows_and_on_it_stops():
    near = fence(-0.05, 0.0, 0.0, _cmd(1.0), LEFT_HALF)
    assert 0 < near.local_forward_vel < 1.0
    assert fence(0.0, 0.0, 0.0, _cmd(1.0), LEFT_HALF).local_forward_vel == 0.0


def test_only_the_speed_towards_the_line_is_capped_whatever_way_the_robot_faces():
    # Facing +y, its left is -x: a robot on the line driving left-and-forward keeps the +y part.
    out = fence(0.0, 0.0, math.pi / 2, _cmd(0.5, -1.0, 1.0), LEFT_HALF)
    assert out.local_forward_vel == pytest.approx(0.5)
    assert out.local_left_vel == pytest.approx(0.0)
    assert out.angular_vel == 1.0


def test_a_robot_outside_may_drive_back_in_but_not_further_out():
    assert fence(0.4, 0.0, 0.0, _cmd(-1.0), LEFT_HALF).local_forward_vel == -1.0
    assert fence(0.4, 0.0, 0.0, _cmd(1.0), LEFT_HALF).local_forward_vel == 0.0
    assert fence(-1.0, -1.7, 0.0, _cmd(left=-1.0), LEFT_HALF).local_left_vel == 0.0  # past the touch line


def test_the_controller_fences_the_robots_it_sees_and_passes_the_rest():
    sent = {}
    inner = SimpleNamespace(add_robot_commands=sent.update)
    game = SimpleNamespace(friendly_robots={0: SimpleNamespace(p=Vector2D(0.0, 0.0), orientation=0.0)})
    controller = FencedRobotController(inner, LEFT_HALF, lambda: game)
    controller.add_robot_commands({0: _cmd(1.0), 1: _cmd(1.0)})
    assert sent[0].local_forward_vel == 0.0
    assert sent[1].local_forward_vel == 1.0


FULL = STANDARD_FIELD_DIMS.full_field_bounds  # 9 m x 6 m: x in [-4.5, 4.5], y in [-3, 3]


def test_only_the_edges_inside_the_field_are_fenced():
    left_half = FieldBounds(top_left=(-4.5, 3.0), bottom_right=(0.0, -3.0))
    assert inner_edges(left_half, FULL) == FieldBounds(top_left=(-math.inf, math.inf), bottom_right=(0.0, -math.inf))
    assert inner_edges(FULL, FULL) is None


def test_a_robot_may_cross_the_touch_line_to_fetch_a_ball_out_of_play():
    # The ball 0.2 m past the touch line on a free kick: a fence at the line froze the restart.
    left_half = inner_edges(FieldBounds(top_left=(-4.5, 3.0), bottom_right=(0.0, -3.0)), FULL)
    assert fence(-2.0, 3.0, 0.0, _cmd(left=1.0), left_half).local_left_vel == 1.0
    assert fence(-4.5, 0.0, 0.0, _cmd(-1.0), left_half).local_forward_vel == -1.0  # into its goal mouth
    assert fence(0.0, 0.0, 0.0, _cmd(1.0), left_half).local_forward_vel == 0.0  # the half-way line holds


def _runner(mode, field_bounds) -> StrategyRunner:
    runner = StrategyRunner.__new__(StrategyRunner)
    runner.mode, runner.field_bounds, runner.full_field_dims = mode, field_bounds, STANDARD_FIELD_DIMS
    return runner


def _robot_on_the_half_way_line():
    return SimpleNamespace(friendly_robots={0: SimpleNamespace(p=Vector2D(0.0, 0.0), orientation=0.0)})


def test_a_kicker_less_robots_push_is_fenced_too():
    # The profile turns a kick into a 1 m/s push; fenced before it, the push drove 0.3 m past the line.
    sent = {}
    left_half = FieldBounds(top_left=(-4.5, 3.0), bottom_right=(0.0, -3.0))
    controller = _runner(Mode.GRSIM, left_half)._wrapped(
        SimpleNamespace(add_robot_commands=sent.update), {0: RobotProfile(kicker=False)}, _robot_on_the_half_way_line
    )
    controller.add_robot_commands({0: RobotCommand(0.0, 0.0, 0.0, kick=True, chip=False, dribble=False)})
    assert sent[0].local_forward_vel == 0.0


@pytest.mark.parametrize(
    "mode, bounds", [(Mode.RSIM, FieldBounds(top_left=(-4.5, 3.0), bottom_right=(0.0, -3.0))), (Mode.GRSIM, FULL)]
)
def test_no_fence_in_rsim_or_when_the_cameras_see_the_whole_field(mode, bounds):
    inner = SimpleNamespace(add_robot_commands=lambda commands: None)
    assert _runner(mode, bounds)._fenced(inner, _robot_on_the_half_way_line) is inner
