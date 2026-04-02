import numpy as np
import pytest
from numpy.testing import assert_array_equal

from utama_core.config.field_params import (
    GREAT_EXHIBITION_FIELD_DIMS,
    STANDARD_FIELD_DIMS,
    FieldBounds,
    FieldDimensions,
)
from utama_core.entities.game.field import Field


@pytest.mark.parametrize(
    "dims",
    [
        STANDARD_FIELD_DIMS,
        GREAT_EXHIBITION_FIELD_DIMS,
        FieldDimensions(
            full_field_half_length=6.0,
            full_field_half_width=4.0,
            half_defense_area_depth=0.5,
            half_defense_area_width=1.0,
            half_goal_width=0.5,
        ),
    ],
)
def test_full_field_bounds_follow_resized_dimensions(dims: FieldDimensions):
    bounds = dims.full_field_bounds
    assert bounds.top_left == (-dims.full_field_half_length, dims.full_field_half_width)
    assert bounds.bottom_right == (
        dims.full_field_half_length,
        -dims.full_field_half_width,
    )
    assert bounds.center == (0.0, 0.0)


@pytest.mark.parametrize(
    "dims",
    [
        STANDARD_FIELD_DIMS,
        GREAT_EXHIBITION_FIELD_DIMS,
        FieldDimensions(5.2, 3.6, 0.7, 1.4, 0.6),
    ],
)
def test_full_field_polygon_matches_dimensions(dims: FieldDimensions):
    length = dims.full_field_half_length
    width = dims.full_field_half_width
    expected = np.array([(length, width), (length, -width), (-length, -width), (-length, width)])
    assert_array_equal(dims.full_field, expected)


def test_goal_lines_shift_when_full_field_is_resized():
    small = FieldDimensions(3.0, 2.0, 0.5, 1.0, 0.5)
    large = FieldDimensions(6.0, 2.0, 0.5, 1.0, 0.5)

    assert_array_equal(small.right_goal_line, np.array([(3.0, 0.5), (3.0, -0.5)]))
    assert_array_equal(large.right_goal_line, np.array([(6.0, 0.5), (6.0, -0.5)]))
    assert_array_equal(small.left_goal_line, np.array([(-3.0, 0.5), (-3.0, -0.5)]))
    assert_array_equal(large.left_goal_line, np.array([(-6.0, 0.5), (-6.0, -0.5)]))


def test_defense_areas_track_resized_full_length():
    dims = FieldDimensions(5.0, 3.0, 0.75, 1.25, 0.5)

    expected_right = np.array([(5.0, 1.25), (3.5, 1.25), (3.5, -1.25), (5.0, -1.25)])
    expected_left = np.array([(-5.0, 1.25), (-3.5, 1.25), (-3.5, -1.25), (-5.0, -1.25)])

    assert_array_equal(dims.right_defense_area, expected_right)
    assert_array_equal(dims.left_defense_area, expected_left)


@pytest.mark.parametrize("team_is_right", [True, False])
def test_field_goal_lines_match_resized_dimensions(team_is_right: bool):
    dims = FieldDimensions(6.0, 4.0, 0.5, 1.0, 0.5)
    field = Field(
        my_team_is_right=team_is_right,
        field_dims=dims,
        field_bounds=dims.full_field_bounds,
    )

    if team_is_right:
        assert_array_equal(field.my_goal_line, dims.right_goal_line)
        assert_array_equal(field.enemy_goal_line, dims.left_goal_line)
    else:
        assert_array_equal(field.my_goal_line, dims.left_goal_line)
        assert_array_equal(field.enemy_goal_line, dims.right_goal_line)


@pytest.mark.parametrize("team_is_right", [True, False])
def test_field_reports_goal_lines_present_on_full_resized_bounds(team_is_right: bool):
    dims = FieldDimensions(6.0, 4.0, 0.5, 1.0, 0.5)
    field = Field(
        my_team_is_right=team_is_right,
        field_dims=dims,
        field_bounds=dims.full_field_bounds,
    )

    assert field.includes_my_goal_line
    assert field.includes_opp_goal_line


@pytest.mark.parametrize("team_is_right", [True, False])
def test_field_reports_goal_lines_absent_when_bounds_crop_goal_width(
    team_is_right: bool,
):
    dims = FieldDimensions(6.0, 4.0, 0.5, 1.0, 0.5)
    cropped_bounds = FieldBounds(top_left=(-6.0, 0.4), bottom_right=(6.0, -0.4))
    field = Field(
        my_team_is_right=team_is_right,
        field_dims=dims,
        field_bounds=cropped_bounds,
    )

    assert not field.includes_my_goal_line
    assert not field.includes_opp_goal_line


@pytest.mark.parametrize(
    ("kwargs", "error_pattern"),
    [
        (
            {
                "full_field_half_length": 0.0,
                "full_field_half_width": 3.0,
                "half_defense_area_depth": 0.5,
                "half_defense_area_width": 1.0,
                "half_goal_width": 0.5,
            },
            "Field length/width must be positive",
        ),
        (
            {
                "full_field_half_length": 4.5,
                "full_field_half_width": 3.0,
                "half_defense_area_depth": 0.0,
                "half_defense_area_width": 1.0,
                "half_goal_width": 0.5,
            },
            "Goal/defense measurements must be positive",
        ),
        (
            {
                "full_field_half_length": 1.0,
                "full_field_half_width": 3.0,
                "half_defense_area_depth": 0.6,
                "half_defense_area_width": 1.0,
                "half_goal_width": 0.5,
            },
            "exceeds field length",
        ),
        (
            {
                "full_field_half_length": 4.5,
                "full_field_half_width": 1.0,
                "half_defense_area_depth": 0.5,
                "half_defense_area_width": 1.1,
                "half_goal_width": 0.5,
            },
            "Defense width .* exceeds field width",
        ),
        (
            {
                "full_field_half_length": 4.5,
                "full_field_half_width": 1.0,
                "half_defense_area_depth": 0.5,
                "half_defense_area_width": 1.0,
                "half_goal_width": 1.1,
            },
            "Goal width .* exceeds field width",
        ),
        (
            {
                "full_field_half_length": 4.5,
                "full_field_half_width": 2.0,
                "half_defense_area_depth": 0.5,
                "half_defense_area_width": 0.6,
                "half_goal_width": 0.8,
            },
            "should not exceed defense width",
        ),
    ],
)
def test_invalid_field_dimensions_raise_value_errors(kwargs, error_pattern: str):
    with pytest.raises(ValueError, match=error_pattern):
        FieldDimensions(**kwargs)


def test_cached_geometry_properties_return_same_objects():
    dims = FieldDimensions(4.5, 3.0, 0.5, 1.0, 0.5)

    assert dims.full_field is dims.full_field
    assert dims.full_field_bounds is dims.full_field_bounds
    assert dims.left_goal_line is dims.left_goal_line
    assert dims.right_goal_line is dims.right_goal_line
    assert dims.left_defense_area is dims.left_defense_area
    assert dims.right_defense_area is dims.right_defense_area
