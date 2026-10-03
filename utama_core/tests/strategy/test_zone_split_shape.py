from types import SimpleNamespace

import pytest

from utama_core.entities.data.object import TeamType
from utama_core.strategy.kernel_strategy import _zone_split_picker
from utama_core.strategy.zone_split_shape import _attack_count

FREE = frozenset({1, 2, 3, 4, 5})
BOTH = frozenset({"attack", "defense"})


@pytest.mark.parametrize(
    "edge, zone, expected",
    [
        (True, "own", 3),
        (True, "mid", 4),
        (True, "final", 4),
        (False, "own", 1),
        (False, "mid", 1),
        (False, "final", 2),
    ],
)
def test_five_robot_split_by_possession_and_third(edge, zone, expected):
    assert _attack_count(5, edge, zone, prev_attack=None) == expected


def test_a_near_tie_keeps_the_previous_count_whatever_the_third():
    assert _attack_count(5, None, "own", prev_attack=4) == 4
    assert _attack_count(5, None, "final", prev_attack=1) == 1


def test_a_near_tie_with_no_history_is_the_defensive_split():
    assert _attack_count(5, None, "mid", prev_attack=None) == 1


def test_counts_stay_within_the_robots_available():
    assert _attack_count(1, True, "mid", None) == 1
    assert _attack_count(2, False, "final", None) == 1
    assert _attack_count(0, True, "mid", None) == 0


def _game(ball_x, friendly_dist, enemy_dist, carrier=None):
    distances = {TeamType.FRIENDLY: (3, friendly_dist), TeamType.ENEMY: (1, enemy_dist)}
    return SimpleNamespace(
        proximity_lookup=SimpleNamespace(closest_to_ball=lambda team_type_filter: distances[team_type_filter]),
        my_team_is_right=True,
        field=SimpleNamespace(half_length=4.5),
        ball=SimpleNamespace(
            p=SimpleNamespace(to_2d=lambda: SimpleNamespace(x=ball_x)), v=SimpleNamespace(x=1.0, y=0.0)
        ),
        friendly_robots={rid: SimpleNamespace(has_ball=rid == carrier) for rid in range(1, 6)},
        enemy_robots={1: SimpleNamespace(has_ball=False)},
    )


def test_building_out_from_our_own_third_keeps_a_second_defender_back():
    # my_team_is_right: our goal is at +x, so x = +3.2 is our own third.
    partition = _zone_split_picker(_game(3.2, 0.2, 0.8, carrier=3), FREE, None, BOTH)
    assert len(partition["attack"]) == 3 and len(partition["defense"]) == 2
    assert 3 in partition["attack"]


def test_losing_the_ball_in_their_third_keeps_a_robot_up():
    partition = _zone_split_picker(_game(-3.2, 0.8, 0.2), FREE, None, BOTH)
    assert len(partition["attack"]) == 2 and len(partition["defense"]) == 3


def test_a_pinned_slot_gets_every_free_robot():
    only_defense = _zone_split_picker(_game(0.0, 0.2, 0.8), FREE, None, frozenset({"defense"}))
    assert only_defense == {"defense": FREE}
    only_attack = _zone_split_picker(_game(0.0, 0.2, 0.8), FREE, None, frozenset({"attack"}))
    assert only_attack == {"attack": FREE}
    assert _zone_split_picker(_game(0.0, 0.2, 0.8), FREE, None, frozenset()) == {}
