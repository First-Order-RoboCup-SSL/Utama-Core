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
    MARGIN,
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


def test_towards_the_half_way_line_the_robot_slows_and_stops_its_body_inside_it():
    near = fence(-MARGIN - 0.05, 0.0, 0.0, _cmd(1.0), LEFT_HALF)
    assert 0 < near.local_forward_vel < 1.0
    assert fence(-MARGIN, 0.0, 0.0, _cmd(1.0), LEFT_HALF).local_forward_vel == 0.0


def test_a_robot_already_driving_at_the_line_is_slowed_from_where_it_will_be():
    # The cap acts DELAY late: a robot at 1 m/s covers 0.1 m first. Without its measured velocity
    # the fence allowed it the speed to stop from where vision last saw it.
    x = -MARGIN - 0.3
    still = fence(x, 0.0, 0.0, _cmd(1.0), LEFT_HALF).local_forward_vel
    moving = fence(x, 0.0, 0.0, _cmd(1.0), LEFT_HALF, velocity=(1.0, 0.0)).local_forward_vel
    assert still == pytest.approx(_max_speed_towards(0.3))
    assert moving == pytest.approx(_max_speed_towards(0.3 - 1.0 * DELAY))
    # Driving away, or along the line, it is not held back.
    assert fence(x, 0.0, 0.0, _cmd(1.0), LEFT_HALF, velocity=(-1.0, 0.5)).local_forward_vel == pytest.approx(still)


def test_a_robot_too_fast_to_stop_in_time_is_sent_no_speed_towards_the_line():
    assert fence(-MARGIN - 0.1, 0.0, 0.0, _cmd(1.0), LEFT_HALF, velocity=(1.5, 0.0)).local_forward_vel == 0.0


def test_the_fence_brakes_no_harder_than_a_robot_driving_sideways_can():
    assert BRAKE < 1.6  # the weakest sideways acceleration measured, 2026-10-27


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
    robot = SimpleNamespace(p=Vector2D(-MARGIN - 0.1, 0.0), v=Vector2D(1.5, 0.0), orientation=0.0)
    game = SimpleNamespace(friendly_robots={0: robot})
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
    robot = SimpleNamespace(p=Vector2D(0.0, 0.0), v=Vector2D(0.0, 0.0), orientation=0.0)
    return SimpleNamespace(friendly_robots={0: robot})


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


# The lab field's bottom half of its bottom-right quarter (examples/shooting_drill.py, KEEP_OUT).
ZONE = FieldBounds(top_left=(0.0, -0.5625), bottom_right=(1.5, -1.125))


def test_a_robot_driving_down_at_a_keep_out_zone_stops_its_body_outside_it():
    y = -0.5625 + MARGIN
    assert fence(0.7, y, 0.0, _cmd(left=-1.0), None, keep_out=(ZONE,)).local_left_vel == 0.0
    near = fence(0.7, y + 0.2, 0.0, _cmd(left=-1.0), None, keep_out=(ZONE,)).local_left_vel
    assert near == pytest.approx(-_max_speed_towards(0.2))


def test_a_robot_driving_across_into_a_keep_out_zone_is_stopped_at_its_side():
    # From the left half, along the bottom touch line, towards the half-way line.
    assert fence(-MARGIN, -0.9, 0.0, _cmd(1.0), None, keep_out=(ZONE,)).local_forward_vel == 0.0


def test_a_robot_beside_a_keep_out_zone_drives_past_it_freely():
    command = _cmd(1.0)
    assert fence(0.2, 0.3, 0.0, command, None, keep_out=(ZONE,)) is command  # above it, driving along
    assert fence(0.7, 0.3, 0.0, _cmd(left=1.0), None, keep_out=(ZONE,)).local_left_vel == 1.0  # driving away


def test_a_robot_cutting_the_corner_of_a_keep_out_zone_is_slowed_before_it_lines_up():
    # Diagonally at the zone's top-left corner, 0.2 m off each side: not yet beside either side,
    # but within the distance it travels before stopping, so both sides fence it.
    out = fence(-0.2, -0.3625, 0.0, _cmd(0.7, -0.7), None, keep_out=(ZONE,))
    assert out.local_forward_vel < 0.7 and out.local_left_vel > -0.7


def test_a_robot_inside_a_keep_out_zone_may_drive_out():
    command = _cmd(1.0, 1.0)
    assert fence(0.7, -0.9, 0.0, command, None, keep_out=(ZONE,)) is command


def test_the_keep_out_zone_is_fenced_in_rsim_too():
    runner = _runner(Mode.RSIM, FieldBounds(top_left=(-4.5, 3.0), bottom_right=(0.0, -3.0)))
    runner.keep_out = (ZONE,)
    sent = {}
    robot = SimpleNamespace(p=Vector2D(0.7, -0.5625 + MARGIN), v=Vector2D(0.0, 0.0), orientation=0.0)
    controller = runner._fenced(
        SimpleNamespace(add_robot_commands=sent.update), lambda: SimpleNamespace(friendly_robots={0: robot})
    )
    controller.add_robot_commands({0: _cmd(left=-1.0)})
    assert sent[0].local_left_vel == 0.0


def test_calibration_fences_the_field_lines_too():
    # 2026-10-29: calibration drove a robot 0.14 m past the goal line; on the full field nothing fenced it.
    full = STANDARD_FIELD_DIMS.full_field_bounds
    runner = _runner(Mode.REAL, full)
    sent = {}
    robot = SimpleNamespace(p=Vector2D(-4.5 + MARGIN, 0.0), v=Vector2D(0.0, 0.0), orientation=0.0)
    game = lambda: SimpleNamespace(friendly_robots={0: robot})  # noqa: E731
    assert runner._fenced(SimpleNamespace(add_robot_commands=sent.update), game).add_robot_commands is not None
    runner.fence_field_lines = True
    runner._fenced(SimpleNamespace(add_robot_commands=sent.update), game).add_robot_commands({0: _cmd(-1.0)})
    assert sent[0].local_forward_vel == 0.0
