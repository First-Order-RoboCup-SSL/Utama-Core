from types import SimpleNamespace

import pytest

from utama_core.entities.data.object import TeamType
from utama_core.strategy.split_press import _split_press_picker

ALL_TACTICS = frozenset({"attack", "defense", "press"})


def _game(n, friendly_dist, enemy_dist):
    """n free friendly robots, none holding the ball, ball moving so no still-ball kicker."""
    distances = {TeamType.FRIENDLY: (1, friendly_dist), TeamType.ENEMY: (1, enemy_dist)}
    robot = SimpleNamespace(has_ball=False, p=SimpleNamespace(x=0.0, y=0.0))
    return SimpleNamespace(
        proximity_lookup=SimpleNamespace(closest_to_ball=lambda team_type_filter: distances[team_type_filter]),
        ball=SimpleNamespace(p=SimpleNamespace(x=0.0, y=0.0), v=SimpleNamespace(x=1.0, y=0.0)),
        friendly_robots={rid: robot for rid in range(1, n + 1)},
        enemy_robots={1: robot},
    )


def _picker(n, available, friendly_dist=1.0, enemy_dist=0.5):
    return _split_press_picker(
        _game(n, friendly_dist, enemy_dist), frozenset(range(1, n + 1)), None, frozenset(available)
    )


def _assert_covers(partition, n):
    assert set().union(*partition.values()) == set(range(1, n + 1))
    assert all(robots for robots in partition.values()), "no empty frozenset may be emitted"


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_press_available_without_ball_edge_presses_with_two_and_defends_the_rest(n):
    partition = _picker(n, ALL_TACTICS)
    assert partition["press"] == frozenset({1, 2})
    assert partition.get("defense", frozenset()) == frozenset(range(3, n + 1))
    assert "attack" not in partition
    _assert_covers(partition, n)


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_press_not_applicable_falls_back_to_split_shape_split(n):
    # Without an enemy in press range the split is split_shape's: attack n//2-1, defense rest.
    partition = _picker(n, {"attack", "defense"})
    attack_count = max(0, n // 2 - 1)
    assert partition.get("attack", frozenset()) == frozenset(range(1, attack_count + 1))
    assert partition["defense"] == frozenset(range(attack_count + 1, n + 1))
    assert "press" not in partition
    _assert_covers(partition, n)


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_ball_edge_keeps_split_shape_split_even_with_press_available(n):
    partition = _picker(n, ALL_TACTICS, friendly_dist=0.2, enemy_dist=0.9)
    attack_count = (n + 1) // 2 + 1
    assert partition["attack"] == frozenset(range(1, attack_count + 1))
    assert "press" not in partition
    _assert_covers(partition, n)


def test_press_boundary_is_exactly_the_press_availability():
    # Same geometry, only availability differs: the press slot decides the shape.
    with_press = _picker(4, ALL_TACTICS)
    without_press = _picker(4, {"attack", "defense"})
    assert "press" in with_press and "press" not in without_press


def test_tie_at_the_ball_is_not_a_friendly_ball_edge():
    # Ties go to the press (friendly_dist < enemy_dist is strict, as in split_shape).
    partition = _picker(4, ALL_TACTICS, friendly_dist=0.5, enemy_dist=0.5)
    assert partition["press"] == frozenset({1, 2})


@pytest.mark.parametrize("n", [1])
def test_a_single_robot_presses_alone(n):
    partition = _picker(n, ALL_TACTICS)
    assert partition == {"press": frozenset({1})}


def test_two_robots_press_and_neither_is_left_for_defense():
    assert _picker(2, ALL_TACTICS) == {"press": frozenset({1, 2})}


def test_press_gets_all_when_nothing_left_for_defense_or_attack():
    partition = _picker(3, {"press"})
    assert partition == {"press": frozenset({1, 2})}


def test_no_robots_gives_no_partition():
    game = _game(1, 1.0, 0.5)
    assert _split_press_picker(game, frozenset(), None, ALL_TACTICS) == {}
