"""Tests for AbstractSimController bounds-checking behaviour."""

from unittest.mock import MagicMock, call, patch

import pytest

from utama_core.entities.game.field import Field, FieldBounds
from utama_core.team_controller.src.controllers.common.sim_controller_abstract import (
    AbstractSimController,
)

# ---------------------------------------------------------------------------
# Concrete stub – fulfils abstract interface without any real I/O
# ---------------------------------------------------------------------------


class StubSimController(AbstractSimController):
    """Minimal concrete subclass for unit-testing AbstractSimController."""

    def __init__(self, field_bounds: FieldBounds, exp_ball: bool = True):
        super().__init__(field_bounds, exp_ball)
        self.teleport_ball_calls: list[tuple] = []
        self.teleport_robot_calls: list[tuple] = []

    def _do_teleport_ball_unrestricted(self, x, y, vx, vy):
        self.teleport_ball_calls.append((x, y, vx, vy))

    def _do_teleport_robot_unrestricted(self, is_team_yellow, robot_id, x, y, theta=None):
        self.teleport_robot_calls.append((is_team_yellow, robot_id, x, y, theta))

    def set_robot_presence(self, robot_id, is_team_yellow, should_robot_be_present):
        pass  # not under test here


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# Standard SSL full-field bounds: x ∈ [-4.5, 4.5], y ∈ [-3.0, 3.0]
FULL_BOUNDS = Field.FULL_FIELD_BOUNDS

# A small custom bounds used for many tests: x ∈ [0, 2], y ∈ [-1, 1]
CUSTOM_BOUNDS = FieldBounds(top_left=(0.0, 1.0), bottom_right=(2.0, -1.0))


@pytest.fixture
def ctrl():
    """Controller using custom bounds with ball expected."""
    return StubSimController(CUSTOM_BOUNDS, exp_ball=True)


@pytest.fixture
def ctrl_no_ball():
    """Controller configured without a ball."""
    return StubSimController(CUSTOM_BOUNDS, exp_ball=False)


@pytest.fixture
def ctrl_full():
    """Controller using full-field bounds."""
    return StubSimController(FULL_BOUNDS, exp_ball=True)


# ===========================================================================
# teleport_ball
# ===========================================================================


class TestTeleportBall:
    # --- happy-path ---

    def test_inside_bounds_calls_unrestricted(self, ctrl):
        ctrl.teleport_ball(1.0, 0.0)
        assert ctrl.teleport_ball_calls == [(1.0, 0.0, 0, 0)]

    def test_velocity_forwarded(self, ctrl):
        ctrl.teleport_ball(1.0, 0.0, vx=0.5, vy=-0.3)
        assert ctrl.teleport_ball_calls == [(1.0, 0.0, 0.5, -0.3)]

    def test_center_of_bounds_accepted(self, ctrl):
        ctrl.teleport_ball(1.0, 0.0)  # center of CUSTOM_BOUNDS
        assert len(ctrl.teleport_ball_calls) == 1

    @pytest.mark.parametrize(
        "x, y",
        [
            (0.0, 0.0),  # min-x, center-y corner
            (2.0, 0.0),  # max-x, center-y corner
            (1.0, 1.0),  # center-x, max-y boundary
            (1.0, -1.0),  # center-x, min-y boundary
            (0.0, 1.0),  # top-left exact corner
            (2.0, -1.0),  # bottom-right exact corner
        ],
    )
    def test_boundary_positions_accepted(self, ctrl, x, y):
        ctrl.teleport_ball(x, y)
        assert len(ctrl.teleport_ball_calls) == 1

    # --- out-of-bounds ---

    @pytest.mark.parametrize(
        "x, y",
        [
            (3.0, 0.0),  # x too large
            (-1.0, 0.0),  # x too small
            (1.0, 2.0),  # y too large
            (1.0, -2.0),  # y too small
            (2.1, -1.1),  # both axes out
            (0.0, 1.1),  # y just over the top
            (2.0, -1.01),  # y just under the bottom
        ],
    )
    def test_out_of_bounds_raises(self, ctrl, x, y):
        with pytest.raises(ValueError, match=r"outside of the field boundaries"):
            ctrl.teleport_ball(x, y)

    def test_out_of_bounds_does_not_call_unrestricted(self, ctrl):
        with pytest.raises(ValueError):
            ctrl.teleport_ball(99.0, 99.0)
        assert ctrl.teleport_ball_calls == []

    # --- exp_ball=False ---

    def test_raises_when_ball_not_expected(self, ctrl_no_ball):
        with pytest.raises(ValueError, match=r"not expect a ball"):
            ctrl_no_ball.teleport_ball(1.0, 0.0)

    def test_ball_not_expected_does_not_check_bounds(self, ctrl_no_ball):
        """exp_ball check must fire before bounds check."""
        with pytest.raises(ValueError, match=r"not expect a ball"):
            ctrl_no_ball.teleport_ball(99.0, 99.0)

    def test_ball_not_expected_unrestricted_not_called(self, ctrl_no_ball):
        with pytest.raises(ValueError):
            ctrl_no_ball.teleport_ball(1.0, 0.0)
        assert ctrl_no_ball.teleport_ball_calls == []

    # --- full-field bounds ---

    @pytest.mark.parametrize(
        "x, y",
        [
            (0.0, 0.0),
            (-4.5, 3.0),  # top-left
            (4.5, -3.0),  # bottom-right
            (4.5, 3.0),  # top-right
            (-4.5, -3.0),  # bottom-left
        ],
    )
    def test_full_field_boundary_accepted(self, ctrl_full, x, y):
        ctrl_full.teleport_ball(x, y)
        assert len(ctrl_full.teleport_ball_calls) == 1

    @pytest.mark.parametrize(
        "x, y",
        [
            (4.51, 0.0),
            (-4.51, 0.0),
            (0.0, 3.01),
            (0.0, -3.01),
        ],
    )
    def test_just_outside_full_field_raises(self, ctrl_full, x, y):
        with pytest.raises(ValueError):
            ctrl_full.teleport_ball(x, y)


# ===========================================================================
# teleport_robot
# ===========================================================================


class TestTeleportRobot:
    # --- happy-path ---

    def test_inside_bounds_calls_unrestricted(self, ctrl):
        ctrl.teleport_robot(True, 0, 1.0, 0.0, 0.0)
        assert ctrl.teleport_robot_calls == [(True, 0, 1.0, 0.0, 0.0)]

    def test_theta_forwarded(self, ctrl):
        import math

        ctrl.teleport_robot(False, 2, 1.0, 0.5, math.pi / 2)
        assert ctrl.teleport_robot_calls == [(False, 2, 1.0, 0.5, math.pi / 2)]

    def test_theta_none_forwarded(self, ctrl):
        ctrl.teleport_robot(True, 1, 1.0, 0.0)
        assert ctrl.teleport_robot_calls == [(True, 1, 1.0, 0.0, None)]

    @pytest.mark.parametrize(
        "x, y",
        [
            (0.0, 0.0),
            (2.0, 0.0),
            (1.0, 1.0),
            (1.0, -1.0),
            (0.0, 1.0),
            (2.0, -1.0),
        ],
    )
    def test_boundary_positions_accepted(self, ctrl, x, y):
        ctrl.teleport_robot(True, 0, x, y)
        assert len(ctrl.teleport_robot_calls) == 1

    # --- out-of-bounds ---

    @pytest.mark.parametrize(
        "x, y",
        [
            (3.0, 0.0),
            (-0.1, 0.0),
            (1.0, 1.1),
            (1.0, -1.1),
            (-1.0, -2.0),
        ],
    )
    def test_out_of_bounds_raises(self, ctrl, x, y):
        with pytest.raises(ValueError, match=r"outside of the field boundaries"):
            ctrl.teleport_robot(True, 0, x, y)

    def test_out_of_bounds_does_not_call_unrestricted(self, ctrl):
        with pytest.raises(ValueError):
            ctrl.teleport_robot(True, 0, 99.0, 99.0)
        assert ctrl.teleport_robot_calls == []

    # --- exp_ball flag should NOT affect robot teleport ---

    def test_robot_teleport_allowed_when_no_ball(self, ctrl_no_ball):
        ctrl_no_ball.teleport_robot(True, 0, 1.0, 0.0)
        assert ctrl_no_ball.teleport_robot_calls == [(True, 0, 1.0, 0.0, None)]

    def test_robot_out_of_bounds_still_raises_when_no_ball(self, ctrl_no_ball):
        with pytest.raises(ValueError, match=r"outside of the field boundaries"):
            ctrl_no_ball.teleport_robot(True, 0, 99.0, 0.0)

    # --- team flag ---

    @pytest.mark.parametrize("is_yellow", [True, False])
    def test_team_flag_forwarded(self, ctrl, is_yellow):
        ctrl.teleport_robot(is_yellow, 3, 1.0, 0.0, 0.0)
        assert ctrl.teleport_robot_calls[0][0] == is_yellow

    @pytest.mark.parametrize("rid", [0, 1, 5])
    def test_robot_id_forwarded(self, ctrl, rid):
        ctrl.teleport_robot(True, rid, 1.0, 0.0, 0.0)
        assert ctrl.teleport_robot_calls[0][1] == rid


# ===========================================================================
# remove_ball
# ===========================================================================


class TestRemoveBall:
    def test_remove_ball_calls_unrestricted(self, ctrl):
        ctrl.remove_ball()
        assert len(ctrl.teleport_ball_calls) == 1

    def test_remove_ball_places_outside_field(self, ctrl):
        ctrl.remove_ball()
        x, y, _, _ = ctrl.teleport_ball_calls[0]
        # The removed position must be outside the controller's own field bounds
        from utama_core.global_utils.math_utils import in_field_bounds

        assert not in_field_bounds((x, y), ctrl.field_bounds)

    def test_remove_ball_works_when_ball_not_expected(self, ctrl_no_ball):
        """remove_ball bypasses exp_ball guard (it calls the unrestricted method directly)."""
        ctrl_no_ball.remove_ball()
        assert len(ctrl_no_ball.teleport_ball_calls) == 1
