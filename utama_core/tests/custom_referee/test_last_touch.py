"""Unit tests for the colour-blind last-touch inference.

These pin the symmetry contract of `infer_last_touch_team`: the two
teams are judged by identical rules, ties are broken geometrically
(never by colour), and attribution persists across frames with no new
evidence so a kicker's touch recorded at contact is not corrupted while
the ball rolls dead.
"""

from __future__ import annotations

from utama_core.custom_referee.rules.last_touch import infer_last_touch_team
from utama_core.entities.data.vector import Vector2D, Vector3D
from utama_core.entities.game.ball import Ball
from utama_core.entities.game.game_frame import GameFrame
from utama_core.entities.game.robot import Robot


def _ball(x: float, y: float) -> Ball:
    return Ball(p=Vector3D(x, y, 0), v=Vector3D(0, 0, 0), a=Vector3D(0, 0, 0))


def _robot(robot_id: int, x: float, y: float, is_friendly: bool, has_ball: bool = False) -> Robot:
    return Robot(
        id=robot_id,
        is_friendly=is_friendly,
        has_ball=has_ball,
        p=Vector2D(x, y),
        v=Vector2D(0, 0),
        a=Vector2D(0, 0),
        orientation=0.0,
    )


def _frame(ball: Ball, friendly: dict | None = None, enemy: dict | None = None) -> GameFrame:
    return GameFrame(
        ts=1.0,
        my_team_is_yellow=True,
        my_team_is_right=False,
        friendly_robots=friendly or {},
        enemy_robots=enemy or {},
        ball=ball,
        referee=None,
    )


class TestColourBlindInference:
    def test_friendly_contact_only(self):
        frame = _frame(
            _ball(0, 0),
            friendly={0: _robot(0, 0.1, 0.0, True, has_ball=True)},
            enemy={0: _robot(0, 0.8, 0.0, False)},
        )
        assert infer_last_touch_team(frame) is True

    def test_enemy_contact_only_detected_even_when_far(self):
        # A confirmed enemy touch must be attributed to the enemy even when
        # the robot is well beyond proximity range — the old tracker only
        # read the friendly side's contact flag.
        frame = _frame(
            _ball(0, 0),
            friendly={0: _robot(0, 0.4, 0.0, True)},
            enemy={0: _robot(0, 0.5, 0.0, False, has_ball=True)},
        )
        assert infer_last_touch_team(frame) is False

    def test_scrum_tie_broken_by_distance_enemy_closer(self):
        # Both teams in contact: the closer robot owns the touch — colour
        # must not decide (the old tracker always gave this to friendly).
        frame = _frame(
            _ball(0, 0),
            friendly={0: _robot(0, 0.14, 0.0, True, has_ball=True)},
            enemy={0: _robot(0, 0.05, 0.0, False, has_ball=True)},
        )
        assert infer_last_touch_team(frame) is False

    def test_scrum_tie_broken_by_distance_friendly_closer(self):
        frame = _frame(
            _ball(0, 0),
            friendly={0: _robot(0, 0.05, 0.0, True, has_ball=True)},
            enemy={0: _robot(0, 0.14, 0.0, False, has_ball=True)},
        )
        assert infer_last_touch_team(frame) is True

    def test_proximity_fallback_within_touch_dist(self):
        # No contact flags: closest robot within touch distance decides.
        frame = _frame(
            _ball(0, 0),
            friendly={0: _robot(0, 0.4, 0.0, True)},
            enemy={0: _robot(0, 0.1, 0.0, False)},
        )
        assert infer_last_touch_team(frame) is False

    def test_persists_previous_attribution_without_new_evidence(self):
        # Kicker long gone, nobody near the dead ball: attribution must
        # survive rather than flip to an unrelated nearby robot.
        frame = _frame(
            _ball(3.0, 3.0),
            friendly={0: _robot(1, 1.0, 1.0, True)},
            enemy={0: _robot(0, -2.0, 0.5, False)},
        )
        assert infer_last_touch_team(frame, previous=False) is False

    def test_any_distance_inference_when_never_attributed(self):
        # No prior attribution at all (ball placed then kicked out): the
        # closest robot at any distance is inferred, like the real
        # GameController — never a hardcoded colour.
        frame = _frame(
            _ball(4.9, 1.0),
            friendly={0: _robot(0, 2.4, 0.8, True)},
            enemy={0: _robot(0, -4.1, 0.6, False)},
        )
        assert infer_last_touch_team(frame) is True

    def test_none_when_frame_has_no_robots(self):
        assert infer_last_touch_team(_frame(_ball(0, 0))) is None

    def test_none_ball_missing_keeps_previous(self):
        frame = GameFrame(
            ts=1.0,
            my_team_is_yellow=True,
            my_team_is_right=False,
            friendly_robots={},
            enemy_robots={},
            ball=None,
            referee=None,
        )
        assert infer_last_touch_team(frame, previous=False) is False
