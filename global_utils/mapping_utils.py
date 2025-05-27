from typing import Tuple, TypeVar

T = TypeVar("T")  # generic type variable


def map_friendly_enemy_to_colors(
    my_team_is_yellow: bool, friendly_item: T, enemy_item: T
) -> Tuple[T, T]:
    """
    Map friendly and enemy items to their respective colors based on my team color.

    Args:
        my_team_is_yellow (bool): True if the team is yellow, False if blue.
        friendly_item (T): Any item from the friendly team (int, list, etc.)
        enemy_item (T): Any item from the enemy team (int, list, etc.)

    Returns:
        Tuple[T, T]: A tuple of (yellow_item, blue_item).
    """
    if my_team_is_yellow:
        yellow_item = friendly_item
        blue_item = enemy_item
    else:
        yellow_item = enemy_item
        blue_item = friendly_item
    return yellow_item, blue_item


def map_left_right_to_colors(
    my_team_is_yellow: bool, my_team_is_right: bool, right_item: T, left_item: T
):
    """
    Map left and right items to their respective colors based on my team color and position.

    Args:
        my_team_is_yellow (bool): True if the team is yellow, False if blue.
        my_team_is_right (bool): True if the team is on the right side, False if left.
        right_item (T): Any item from the right side (int, list, etc.)
        left_item (T): Any item from the left side (int, list, etc.)

    Returns:
        Tuple[T, T]: A tuple of (yellow_item, blue_item).
    """
    yellow_is_right = my_team_is_yellow == my_team_is_right
    if yellow_is_right:
        yellow_item = right_item
        blue_item = left_item
    else:
        yellow_item = left_item
        blue_item = right_item
    return yellow_item, blue_item
