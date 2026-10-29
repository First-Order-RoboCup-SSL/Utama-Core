"""The browser vision stream only shows the match: on the 2026-10-27 field test it raised drawing
a robot whose filtered position had run off to 10^7 m (pygame: "invalid rect assignment"), and
the exception stopped the runner and so the robots."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from utama_core.config.field_params import STANDARD_FIELD_DIMS
from utama_core.entities.data.vector import Vector2D, Vector3D
from utama_core.entities.game import Ball, GameFrame
from utama_core.entities.game.robot import Robot
from utama_core.run.vision_stream import GameFrameRenderer
from utama_core.run.vision_stream_status import VisionStreamStatusMixin


def _frame(x: float) -> GameFrame:
    robot = Robot(
        id=0, is_friendly=True, has_ball=False, p=Vector2D(x, 0.0), v=Vector2D(0, 0), a=Vector2D(0, 0), orientation=0.0
    )
    ball = Ball(p=Vector3D(0, 0, 0), v=Vector3D(0, 0, 0), a=Vector3D(0, 0, 0))
    return GameFrame(0.0, False, False, {0: robot}, {}, ball)


class _Stream:
    def __init__(self):
        self.frames = 0

    def is_due(self):
        return True

    def publish_status(self, status):
        pass

    def publish_rgb_frame(self, frame):
        self.frames += 1


def _runner(frame: GameFrame) -> SimpleNamespace:
    runner = SimpleNamespace(
        vision_stream=_Stream(),
        _vision_stream_renderer=GameFrameRenderer(STANDARD_FIELD_DIMS, scale=300.0),
        my=SimpleNamespace(current_game_frame=frame),
        _vision_stream_status=lambda: {},
        logger=logging.getLogger("test"),
    )
    runner.publish = lambda: VisionStreamStatusMixin._publish_vision_stream_frame(runner)
    return runner


def test_the_renderer_itself_cannot_draw_a_robot_at_ten_million_metres():
    with pytest.raises(TypeError):
        GameFrameRenderer(STANDARD_FIELD_DIMS, scale=300.0).render(_frame(1.15e7))


def test_a_frame_the_stream_cannot_draw_is_skipped_and_the_next_is_sent():
    runner = _runner(_frame(1.15e7))
    runner.publish()  # must not raise
    assert runner.vision_stream.frames == 0
    runner.my.current_game_frame = _frame(0.5)
    runner.publish()
    assert runner.vision_stream.frames == 1
