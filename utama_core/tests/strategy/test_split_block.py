from types import SimpleNamespace

from utama_core.entities.data.object import TeamType
from utama_core.strategy.split_block import _possession_split_picker

_FIVE = frozenset({1, 2, 3, 4, 5})
_BOTH = frozenset({"attack", "defense"})


class _Proximity:
    def __init__(self, friendly_dist, enemy_dist):
        self._dists = {TeamType.FRIENDLY: friendly_dist, TeamType.ENEMY: enemy_dist}

    def closest_to_ball(self, team_type_filter=None):
        return (None, self._dists[team_type_filter])


def _game(friendly_dist, enemy_dist):
    # Ball at the origin and at rest; no robot holds it, so `carrier_first` is plain id order.
    far = SimpleNamespace(has_ball=False, p=SimpleNamespace(x=-9.0, y=0.0))
    return SimpleNamespace(
        proximity_lookup=_Proximity(friendly_dist, enemy_dist),
        ball=SimpleNamespace(p=SimpleNamespace(x=0.0, y=0.0), v=SimpleNamespace(x=0.0, y=0.0)),
        friendly_robots={rid: far for rid in range(1, 6)},
        enemy_robots={},
    )


def test_we_have_the_ball_edge_gives_attack_four_and_defense_one():
    partition = _possession_split_picker(_game(0.2, 1.5), _FIVE, None, _BOTH)
    assert partition == {"attack": frozenset({1, 2, 3, 4}), "defense": frozenset({5})}


def test_they_have_the_ball_edge_gives_attack_one_and_defense_four():
    partition = _possession_split_picker(_game(1.5, 0.2), _FIVE, None, _BOTH)
    assert partition == {"attack": frozenset({1}), "defense": frozenset({2, 3, 4, 5})}


def test_an_exact_tie_is_not_our_edge_so_it_takes_the_defence_heavy_split():
    # Strict `<` in the picker: equal distances must not count as our ball.
    partition = _possession_split_picker(_game(0.5, 0.5), _FIVE, None, _BOTH)
    assert partition == {"attack": frozenset({1}), "defense": frozenset({2, 3, 4, 5})}


def test_a_pinned_attack_leaves_every_robot_to_defense():
    partition = _possession_split_picker(_game(0.2, 1.5), _FIVE, None, frozenset({"defense"}))
    assert partition == {"defense": _FIVE}


def test_a_pinned_defense_gives_every_robot_to_attack():
    partition = _possession_split_picker(_game(1.5, 0.2), _FIVE, None, frozenset({"attack"}))
    assert partition == {"attack": _FIVE}
