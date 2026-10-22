"""Robots on camera runs are kept inside the area in play: a tactic chasing a ball out of the
camera's half drove its robot out of sight (a grSim test, 0.9 m past the half-way line)."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from utama_core.entities.data.command import RobotCommand
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game.field import FieldBounds
from utama_core.run.field_fence import (
    BRAKE,
    DELAY,
    FencedRobotController,
    _max_speed_towards,
    fence,
)

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
