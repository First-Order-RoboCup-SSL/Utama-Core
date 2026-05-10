from types import SimpleNamespace

import numpy as np
import pytest

from utama_core.skills.src.utils.defense_utils import (
    DefenseGeometry,
    _defense_geometry,
    calculate_defense_area,
    clamp_to_goal_width,
)


# ---------------------------------------------------------------------------
# Field stubs
# ---------------------------------------------------------------------------
def _make_field(goal_x, goal_half_width, defense_front_x, defense_half_width, my_team_is_right):
    return SimpleNamespace(
        my_goal_line=np.array(
            [
                (goal_x, goal_half_width),
                (goal_x, -goal_half_width),
            ]
        ),
        my_defense_area=np.array(
            [
                (goal_x, defense_half_width),
                (defense_front_x, defense_half_width),
                (defense_front_x, -defense_half_width),
                (goal_x, -defense_half_width),
                (goal_x, defense_half_width),
            ]
        ),
    )


def _make_game(my_team_is_right, field):
    return SimpleNamespace(my_team_is_right=my_team_is_right, field=field)


# ---------------------------------------------------------------------------
# _defense_geometry – standard fields
# ---------------------------------------------------------------------------


def test_defense_geometry_left_goal():
    field = _make_field(-4.5, 0.5, -3.5, 1.0, False)
    game = _make_game(False, field)
    geo = _defense_geometry(game)

    assert geo.goal_x == pytest.approx(-4.5)
    assert geo.goal_half_width == pytest.approx(0.5)
    assert geo.defense_front_x == pytest.approx(-3.5)
    assert geo.defense_depth == pytest.approx(1.0)
    assert geo.defense_half_width == pytest.approx(1.0)


def test_defense_geometry_right_goal():
    field = _make_field(4.5, 0.5, 3.5, 1.0, True)
    game = _make_game(True, field)
    geo = _defense_geometry(game)

    assert geo.goal_x == pytest.approx(4.5)
    assert geo.goal_half_width == pytest.approx(0.5)
    assert geo.defense_front_x == pytest.approx(3.5)
    assert geo.defense_depth == pytest.approx(1.0)
    assert geo.defense_half_width == pytest.approx(1.0)


def test_defense_geometry_custom():
    field = _make_field(-6.0, 0.8, -4.5, 1.5, False)
    game = _make_game(False, field)
    geo = _defense_geometry(game)

    assert geo.goal_x == pytest.approx(-6.0)
    assert geo.goal_half_width == pytest.approx(0.8)
    assert geo.defense_front_x == pytest.approx(-4.5)
    assert geo.defense_depth == pytest.approx(1.5)
    assert geo.defense_half_width == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# clamp_to_goal_width
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("y", "half_width", "expected"),
    [
        (0.3, 0.5, 0.3),
        (0.7, 0.5, 0.5),
        (-0.9, 0.5, -0.5),
        (0.3, 0.8, 0.3),
        (0.9, 0.8, 0.8),
        (-1.0, 0.8, -0.8),
    ],
)
def test_clamp_to_goal_width(y, half_width, expected):
    assert clamp_to_goal_width(y, half_width) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# calculate_defense_area – standard field
# ---------------------------------------------------------------------------


def test_calculate_defense_area_standard_field_left_frontmost():
    """At t=pi (front of curve), the x should be goal_x - (defense_depth + 0.1)."""
    field = _make_field(-4.5, 0.5, -3.5, 1.0, False)
    game = _make_game(False, field)
    pos = calculate_defense_area(game, np.pi)

    # Front extent = 1.0 + 0.1 = 1.1, r=2.1
    # At t=pi: cos=-1, sin=0 → x_raw = 1.1 * ((1-2.1)*1*(-1) + 2.1*(-1)) = 1.1 * (1.1 - 2.1) = 1.1 * (-1) = -1.1
    # Left goal: goal_x - x_raw = -4.5 - (-1.1) = -3.4
    assert pos.x == pytest.approx(-3.4)
    assert pos.y == pytest.approx(0.0)


def test_calculate_defense_area_standard_field_right_frontmost():
    """At t=pi (front of curve), for right goal."""
    field = _make_field(4.5, 0.5, 3.5, 1.0, True)
    game = _make_game(True, field)
    pos = calculate_defense_area(game, np.pi)

    # Right goal: goal_x + x_raw = 4.5 + (-1.1) = 3.4
    assert pos.x == pytest.approx(3.4)
    assert pos.y == pytest.approx(0.0)


def test_calculate_defense_area_standard_field_top_bottom():
    """At t=pi/2 and t=3pi/2, the y extents should be ±(defense_half_width + 0.1)."""
    field = _make_field(-4.5, 0.5, -3.5, 1.0, False)
    game = _make_game(False, field)

    top = calculate_defense_area(game, np.pi / 2)
    bottom = calculate_defense_area(game, 3 * np.pi / 2)

    # At t=pi/2: cos=0, sin=1 → y_raw = 1.1 * ((1-2.1)*1*1 + 2.1*1) = 1.1 * 1 = 1.1
    assert top.y == pytest.approx(1.1)
    assert bottom.y == pytest.approx(-1.1)
    # x at top/bottom should be at goal line
    assert top.x == pytest.approx(-4.5)
    assert bottom.x == pytest.approx(-4.5)


# ---------------------------------------------------------------------------
# calculate_defense_area – custom geometry
# ---------------------------------------------------------------------------


def test_calculate_defense_area_custom_geometry_front_extent():
    """Custom defense_depth=2.0 should produce front_extent=2.1."""
    field = _make_field(-6.0, 0.8, -4.0, 1.5, False)
    game = _make_game(False, field)
    pos = calculate_defense_area(game, np.pi)

    # defense_depth = 2.0, front_extent = 2.1
    # x_raw at t=pi: 2.1 * (-1) = -2.1
    # left goal: -6.0 - (-2.1) = -3.9
    assert pos.x == pytest.approx(-3.9)
    assert pos.y == pytest.approx(0.0)


def test_calculate_defense_area_custom_geometry_vertical_extent():
    """Custom defense_half_width=1.5 should produce vertical_extent=1.6."""
    field = _make_field(-6.0, 0.8, -4.0, 1.5, False)
    game = _make_game(False, field)

    top = calculate_defense_area(game, np.pi / 2)

    # vertical_extent = 1.5 + 0.1 = 1.6
    assert top.y == pytest.approx(1.6)
    assert top.x == pytest.approx(-6.0)
