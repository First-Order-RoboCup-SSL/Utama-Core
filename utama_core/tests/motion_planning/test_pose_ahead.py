"""`PoseAhead`: our robots where they will be when a command sent now takes effect, for `move()`."""

import math

import pytest

from utama_core.entities.data.vector import Vector2D
from utama_core.motion_planning.src.common.motion_controller import MotionController
from utama_core.motion_planning.src.common.pose_ahead import (
    PoseAhead,
    control_latency_s,
)
from utama_core.skills.src.utils.move_utils import move
from utama_core.tests.fixtures.game_builder import build_game, make_ball, make_robot

LATENCY = 0.08


def _game(ts=0.0, orientation=0.0, vx=0.0, vy=0.0):
    return build_game(
        {0: make_robot(0, is_friendly=True, x=1.0, y=-0.5, vx=vx, vy=vy, orientation=orientation)},
        {0: make_robot(0, is_friendly=False, x=-1.0, vx=1.0)},
        make_ball(),
        ts=ts,
    )


def test_only_the_real_robots_have_a_latency():
    assert control_latency_s("rsim") == 0.0
    assert control_latency_s("grsim") == 0.0
    assert control_latency_s("real") > 0.0


def test_without_latency_the_game_is_unchanged():
    game = _game(vx=1.0)
    assert PoseAhead(0.0).game(game) is game


def test_our_robots_move_on_at_their_velocity_and_nothing_else_changes():
    game = _game(vx=1.0, vy=-0.5)
    ahead = PoseAhead(LATENCY).game(game)
    assert ahead.friendly_robots[0].p.x == pytest.approx(1.0 + LATENCY)
    assert ahead.friendly_robots[0].p.y == pytest.approx(-0.5 - 0.5 * LATENCY)
    assert game.friendly_robots[0].p == Vector2D(1.0, -0.5)  # the seen game keeps what vision saw
    assert ahead.enemy_robots[0].p == game.enemy_robots[0].p
    assert ahead.ts == game.ts


@pytest.mark.parametrize(
    "first, second, gap, turned_by",
    [
        (0.0, 0.1, 1 / 60, 0.1),
        (math.pi - 0.05, -math.pi + 0.05, 1 / 60, 0.1),  # across +-pi: a small turn, not a whole one
        (0.0, 0.1, 0.15, 0.0),  # frames too far apart to measure a turn rate
    ],
)
def test_the_heading_moves_on_at_the_turn_rate_between_frames(first, second, gap, turned_by):
    pose_ahead = PoseAhead(LATENCY)
    pose_ahead.game(_game(ts=0.0, orientation=first))
    ahead = pose_ahead.game(_game(ts=gap, orientation=second))
    expected = second + turned_by / gap * LATENCY if turned_by else second
    expected = (expected + math.pi) % (2 * math.pi) - math.pi
    assert ahead.friendly_robots[0].orientation == pytest.approx(expected)


def test_several_robots_in_one_tick_share_one_prediction():
    pose_ahead = PoseAhead(LATENCY)
    pose_ahead.game(_game(ts=0.0, orientation=0.0))
    game = _game(ts=1 / 60, orientation=0.1)
    first = pose_ahead.game(game)
    assert pose_ahead.game(game) is first  # a second call this tick does not zero the turn rate


class _Recording(MotionController):
    """Asks for a fixed world velocity and records the game it planned from."""

    def __init__(self, latency_s):
        super().__init__(mode="rsim")
        self.pose_ahead.latency_s = latency_s
        self.planned_from = None

    def calculate(self, game, robot_id, target_pos, target_oren):
        self.planned_from = game
        return Vector2D(1.0, 0.0), 0.0


def test_move_plans_from_the_predicted_pose_and_commands_in_the_predicted_frame():
    """A robot turning at 1 rad/s faces 0.08 rad further round by the time the command lands, so
    a world +x velocity needs a rightward (negative left) part in the robot's frame."""
    mc = _Recording(LATENCY)
    move(_game(ts=0.0, orientation=0.0), mc, 0, Vector2D(2.0, 0.0), 0.0)
    game = _game(ts=0.1, orientation=0.1, vx=1.0)  # turned 0.1 rad in 0.1 s
    command = move(game, mc, 0, Vector2D(2.0, 0.0), 0.0)
    heading = 0.1 + 1.0 * LATENCY
    assert mc.planned_from.friendly_robots[0].p.x == pytest.approx(1.0 + LATENCY)
    assert command.local_forward_vel == pytest.approx(math.cos(heading))
    assert command.local_left_vel == pytest.approx(-math.sin(heading))
