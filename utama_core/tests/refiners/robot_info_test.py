import pytest

from utama_core.data_processing.refiners.robot_info import (
    _BALL_CAPTURE_DIST,
    RobotInfoRefiner,
)
from utama_core.entities.data.command import RobotResponse
from utama_core.entities.data.vector import Vector2D, Vector3D
from utama_core.entities.game.ball import Ball
from utama_core.entities.game.game_frame import GameFrame
from utama_core.entities.game.robot import Robot


def _make_robot(robot_id: int, x: float, y: float, has_ball: bool = False) -> Robot:
    return Robot(
        id=robot_id,
        is_friendly=True,
        has_ball=has_ball,
        p=Vector2D(x, y),
        v=None,
        a=None,
        orientation=0,
    )


def _make_frame(robots: dict, ball_pos=None) -> GameFrame:
    ball = Ball(Vector3D(*ball_pos, 0), Vector3D(0, 0, 0), None) if ball_pos else None
    return GameFrame(
        ts=0.0,
        my_team_is_yellow=True,
        my_team_is_right=True,
        friendly_robots=robots,
        enemy_robots={},
        ball=ball,
    )


# ---------------------------------------------------------------------------
# Default behaviour (trusted_ir_robots=None — trust all)
# ---------------------------------------------------------------------------


def test_default_uses_ir_sensor_true():
    refiner = RobotInfoRefiner()
    frame = _make_frame({0: _make_robot(0, 0.0, 0.0, has_ball=False)}, ball_pos=(1.0, 0.0))
    result = refiner.refine(frame, [RobotResponse(id=0, has_ball=True)])
    assert result.friendly_robots[0].has_ball is True


def test_default_uses_ir_sensor_false():
    refiner = RobotInfoRefiner()
    # Robot is right on top of the ball but IR says False — should respect IR.
    frame = _make_frame({0: _make_robot(0, 0.0, 0.0, has_ball=True)}, ball_pos=(0.0, 0.0))
    result = refiner.refine(frame, [RobotResponse(id=0, has_ball=False)])
    assert result.friendly_robots[0].has_ball is False


def test_empty_responses_returns_frame_unchanged():
    refiner = RobotInfoRefiner()
    frame = _make_frame({0: _make_robot(0, 0.0, 0.0, has_ball=True)})
    assert refiner.refine(frame, []) is frame
    assert refiner.refine(frame, None) is frame


def test_unknown_robot_id_warns(recwarn):
    refiner = RobotInfoRefiner()
    frame = _make_frame({0: _make_robot(0, 0.0, 0.0)})
    refiner.refine(frame, [RobotResponse(id=99, has_ball=True)])
    assert any("99" in str(w.message) for w in recwarn.list)


# ---------------------------------------------------------------------------
# Trusted-IR allowlist: trusted robot uses IR, untrusted uses vision proximity
# ---------------------------------------------------------------------------


def test_trusted_robot_uses_ir_even_when_far_from_ball():
    refiner = RobotInfoRefiner(trusted_ir_robots=frozenset({0}))
    # Robot is 2 m from ball — no proximity, but IR says True.
    frame = _make_frame({0: _make_robot(0, 0.0, 0.0)}, ball_pos=(2.0, 0.0))
    result = refiner.refine(frame, [RobotResponse(id=0, has_ball=True)])
    assert result.friendly_robots[0].has_ball is True


def test_trusted_robot_respects_ir_false():
    refiner = RobotInfoRefiner(trusted_ir_robots=frozenset({0}))
    # Robot is touching the ball but IR says False — should respect IR.
    frame = _make_frame({0: _make_robot(0, 0.0, 0.0)}, ball_pos=(0.0, 0.0))
    result = refiner.refine(frame, [RobotResponse(id=0, has_ball=False)])
    assert result.friendly_robots[0].has_ball is False


def test_untrusted_robot_infers_true_when_close():
    refiner = RobotInfoRefiner(trusted_ir_robots=frozenset())  # no robots trusted
    # Place robot within capture distance; IR reports False (broken sensor).
    frame = _make_frame({0: _make_robot(0, 0.0, 0.0)}, ball_pos=(0.05, 0.0))
    result = refiner.refine(frame, [RobotResponse(id=0, has_ball=False)])
    assert result.friendly_robots[0].has_ball is True


def test_untrusted_robot_infers_false_when_far():
    refiner = RobotInfoRefiner(trusted_ir_robots=frozenset())
    # Robot is far from ball; IR reports True (broken sensor firing randomly).
    frame = _make_frame({0: _make_robot(0, 0.0, 0.0)}, ball_pos=(1.0, 0.0))
    result = refiner.refine(frame, [RobotResponse(id=0, has_ball=True)])
    assert result.friendly_robots[0].has_ball is False


def test_untrusted_robot_at_exact_capture_boundary():
    refiner = RobotInfoRefiner(trusted_ir_robots=frozenset())
    just_inside = _BALL_CAPTURE_DIST - 0.001
    just_outside = _BALL_CAPTURE_DIST + 0.001

    frame_in = _make_frame({0: _make_robot(0, 0.0, 0.0)}, ball_pos=(just_inside, 0.0))
    assert refiner.refine(frame_in, [RobotResponse(id=0, has_ball=False)]).friendly_robots[0].has_ball is True

    frame_out = _make_frame({0: _make_robot(0, 0.0, 0.0)}, ball_pos=(just_outside, 0.0))
    assert refiner.refine(frame_out, [RobotResponse(id=0, has_ball=False)]).friendly_robots[0].has_ball is False


def test_untrusted_robot_no_ball_in_frame_returns_false():
    refiner = RobotInfoRefiner(trusted_ir_robots=frozenset())
    frame = _make_frame({0: _make_robot(0, 0.0, 0.0)}, ball_pos=None)
    result = refiner.refine(frame, [RobotResponse(id=0, has_ball=True)])
    assert result.friendly_robots[0].has_ball is False


# ---------------------------------------------------------------------------
# Mixed team: one robot trusted, one not
# ---------------------------------------------------------------------------


def test_mixed_team_trusted_uses_ir_untrusted_uses_vision():
    refiner = RobotInfoRefiner(trusted_ir_robots=frozenset({0}))
    robots = {
        0: _make_robot(0, 0.0, 0.0),  # trusted — far from ball
        1: _make_robot(1, 0.05, 0.0),  # untrusted — close to ball
    }
    frame = _make_frame(robots, ball_pos=(0.0, 0.0))
    responses = [
        RobotResponse(id=0, has_ball=True),  # IR says yes → trust it even though far
        RobotResponse(id=1, has_ball=False),  # IR says no → ignore, use proximity
    ]
    result = refiner.refine(frame, responses)
    assert result.friendly_robots[0].has_ball is True  # from IR
    assert result.friendly_robots[1].has_ball is True  # from vision (close enough)
