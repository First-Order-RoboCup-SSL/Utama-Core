import pytest

from utama_core.global_utils.mapping_utils import (
    map_friendly_enemy_to_colors,
    map_left_right_to_colors,
)


@pytest.mark.parametrize(
    "team_is_yellow, friendly_item, enemy_item, expected",
    [
        (
            True,
            "ally",
            "enemy",
            ("ally", "enemy"),
        ),  # yellow team → yellow=friendly, blue=enemy
        (
            False,
            "ally",
            "enemy",
            ("enemy", "ally"),
        ),  # blue team → blue=friendly, yellow=enemy
        (True, [1, 2], [3, 4], ([1, 2], [3, 4])),  # non-scalar items
        (False, {"f": 1}, {"e": 2}, ({"e": 2}, {"f": 1})),  # dicts
    ],
)
def test_map_friendly_enemy_to_colors(team_is_yellow, friendly_item, enemy_item, expected):
    yellow_item, blue_item = map_friendly_enemy_to_colors(team_is_yellow, friendly_item, enemy_item)
    assert (yellow_item, blue_item) == expected


@pytest.mark.parametrize(
    "side_team_is_yellow, team_is_right, right_item, left_item, expected",
    [
        # Case 1: Yellow team, yellow is right → yellow gets right item
        (True, True, "R", "L", ("R", "L")),
        # Case 2: Yellow team, yellow is left → yellow gets left item
        (True, False, "R", "L", ("L", "R")),
        # Case 3: Blue team, blue is right → yellow is left
        (False, True, "R", "L", ("L", "R")),
        # Case 4: Blue team, blue is left → yellow is right
        (False, False, "R", "L", ("R", "L")),
        # Non-scalar tests
        (True, True, [10], [20], ([10], [20])),
        (False, False, {"right": 1}, {"left": 2}, ({"right": 1}, {"left": 2})),
    ],
)
def test_map_left_right_to_colors(side_team_is_yellow, team_is_right, right_item, left_item, expected):
    yellow_item, blue_item = map_left_right_to_colors(side_team_is_yellow, team_is_right, right_item, left_item)
    assert (yellow_item, blue_item) == expected
