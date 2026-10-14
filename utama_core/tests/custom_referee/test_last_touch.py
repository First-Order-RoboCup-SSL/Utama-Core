"""Unit tests for the colour-blind last-touch inference.

These pin the symmetry contract of `infer_last_touch_team`: the two
teams are judged by identical rules, ties are broken geometrically
(never by colour), and attribution persists across frames with no new
evidence so a kicker's touch recorded at contact is not corrupted while
the ball rolls dead.
"""

from __future__ import annotations

from utama_core.config.field_params import STANDARD_FIELD_DIMS
from utama_core.custom_referee.geometry import RefereeGeometry
from utama_core.custom_referee.rules.last_touch import infer_last_touch_team
from utama_core.custom_referee.rules.out_of_bounds_rule import OutOfBoundsRule
from utama_core.entities.data.vector import Vector3D
from utama_core.entities.game.ball import Ball
from utama_core.entities.game.game_frame import GameFrame
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.tests.custom_referee.helpers import robot as _robot


def _ball(x: float, y: float) -> Ball:
    return Ball(p=Vector3D(x, y, 0), v=Vector3D(0, 0, 0), a=Vector3D(0, 0, 0))


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


def _moving_ball(x: float, y: float, vx: float, vy: float) -> Ball:
    return Ball(p=Vector3D(x, y, 0), v=Vector3D(vx, vy, 0), a=Vector3D(0, 0, 0))


class TestProximityNeedsABallVelocityChange:
    """A robot near the ball with no contact flag is a toucher only if the ball's
    velocity changed. A robot standing beside a rolling ball is not."""

    def test_nearby_robot_beside_a_freely_rolling_ball_is_not_a_toucher(self):
        frame = _frame(
            _moving_ball(0, 0, 1.0, 0.0),
            friendly={0: _robot(0, 2.0, 2.0, True)},
            enemy={0: _robot(0, 0.1, 0.0, False)},
        )
        assert infer_last_touch_team(frame, previous=True, previous_ball_v=(1.0, 0.0)) is True

    def test_nearby_robot_is_the_toucher_when_the_ball_velocity_changed(self):
        frame = _frame(
            _moving_ball(0, 0, 1.0, 0.0),
            friendly={0: _robot(0, 2.0, 2.0, True)},
            enemy={0: _robot(0, 0.1, 0.0, False)},
        )
        assert infer_last_touch_team(frame, previous=True, previous_ball_v=(0.0, 0.0)) is False

    def test_velocity_change_threshold_boundary(self):
        enemy = {0: _robot(0, 0.1, 0.0, False)}
        far_friendly = {0: _robot(0, 2.0, 2.0, True)}
        just_under = _frame(_moving_ball(0, 0, 0.049, 0.0), friendly=far_friendly, enemy=enemy)
        just_over = _frame(_moving_ball(0, 0, 0.051, 0.0), friendly=far_friendly, enemy=enemy)
        assert infer_last_touch_team(just_under, previous=True, previous_ball_v=(0.0, 0.0)) is True
        assert infer_last_touch_team(just_over, previous=True, previous_ball_v=(0.0, 0.0)) is False

    def test_unknown_previous_velocity_means_proximity_is_no_evidence(self):
        frame = _frame(
            _moving_ball(0, 0, 1.0, 0.0),
            friendly={0: _robot(0, 2.0, 2.0, True)},
            enemy={0: _robot(0, 0.1, 0.0, False)},
        )
        assert infer_last_touch_team(frame, previous=True, previous_ball_v=None) is True

    def test_contact_flag_stays_authoritative_without_a_velocity_change(self):
        frame = _frame(
            _moving_ball(0, 0, 1.0, 0.0),
            friendly={0: _robot(0, 0.1, 0.0, True, has_ball=True)},
            enemy={0: _robot(0, 0.5, 0.0, False)},
        )
        assert infer_last_touch_team(frame, previous=False, previous_ball_v=(1.0, 0.0)) is True

    def test_never_attributed_ball_still_falls_back_to_the_closest_robot(self):
        frame = _frame(
            _moving_ball(0, 0, 1.0, 0.0),
            friendly={0: _robot(0, 2.0, 2.0, True)},
            enemy={0: _robot(0, 0.1, 0.0, False)},
        )
        assert infer_last_touch_team(frame, previous=None, previous_ball_v=(1.0, 0.0)) is False

    def test_out_of_bounds_free_kick_is_not_blamed_on_a_bystander(self):
        """Friendly kicks the ball (contact flag), it rolls out past an enemy robot
        that never touches it. The enemy ends within 0.15 m of the ball at the exit
        tick; the free kick must still go to the enemy, as the friendly side touched
        last."""
        geo = RefereeGeometry.from_field_dims(STANDARD_FIELD_DIMS)
        rule = OutOfBoundsRule()
        kicker = {0: _robot(0, 0.0, 1.9, True, has_ball=True)}
        rule.check(
            _frame(_moving_ball(0.0, 2.0, 0.0, 2.0), friendly=kicker, enemy={}),
            geo,
            RefereeCommand.NORMAL_START,
        )
        rule.check(
            _frame(
                _moving_ball(0.0, 2.9, 0.0, 2.0),
                friendly={0: _robot(0, 0.0, 1.9, True)},
                enemy={0: _robot(0, 0.5, 2.9, False)},
            ),
            geo,
            RefereeCommand.NORMAL_START,
        )
        violation = rule.check(
            _frame(
                _moving_ball(0.0, 3.5, 0.0, 2.0),
                friendly={0: _robot(0, 0.0, 1.9, True)},
                enemy={0: _robot(0, 0.05, 3.4, False)},  # 0.11 m from the ball, never touched it
            ),
            geo,
            RefereeCommand.NORMAL_START,
        )
        assert violation is not None
        assert violation.next_command == RefereeCommand.DIRECT_FREE_BLUE
