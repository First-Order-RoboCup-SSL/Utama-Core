"""Striker evaluation starts stay inside the selected physical pitch."""

import random

import pytest

from tools import striker_duel as duel
from utama_core.config.field_params import (
    GREAT_EXHIBITION_FIELD_DIMS,
    STANDARD_FIELD_DIMS,
)
from utama_core.config.physical_constants import ROBOT_RADIUS


@pytest.mark.parametrize("dims", [STANDARD_FIELD_DIMS, GREAT_EXHIBITION_FIELD_DIMS], ids=["standard", "gerf"])
def test_starts_fit_the_pitch_and_keep_the_attacker_out_of_the_defense_area(dims):
    front = dims.full_field_half_length - 2 * dims.half_defense_area_depth
    for seed in range(200):
        start = duel.sample_start(random.Random(seed), dims)
        assert abs(start.ball_x) < dims.full_field_half_length
        assert abs(start.ball_y) < dims.full_field_half_width
        for robot in (*start.friendly_robots, *start.enemy_robots):
            assert abs(robot.x) + ROBOT_RADIUS < dims.full_field_half_length
            assert abs(robot.y) + ROBOT_RADIUS < dims.full_field_half_width
        attacker, keeper = start.friendly_robots[0], start.enemy_robots[0]
        assert not (attacker.x + ROBOT_RADIUS > front and abs(attacker.y) < dims.half_defense_area_width + ROBOT_RADIUS)
        assert front <= keeper.x <= dims.full_field_half_length
        assert abs(keeper.y) <= dims.half_goal_width
