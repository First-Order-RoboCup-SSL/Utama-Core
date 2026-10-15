from types import SimpleNamespace

from utama_core.entities.data.object import TeamType
from utama_core.strategy.split_sticky import _sticky_split_picker

ALL = frozenset({1, 2, 3, 4, 5})
ALL_TACTICS = frozenset({"attack", "defense"})


def _game(friendly_dist, enemy_dist):
    """Fake Game: ball gap only, no carrier. `friendly_closer_to_ball` reads the proximity lookup."""
    distances = {TeamType.FRIENDLY: (1, friendly_dist), TeamType.ENEMY: (2, enemy_dist)}
    return SimpleNamespace(
        proximity_lookup=SimpleNamespace(closest_to_ball=lambda team_type_filter: distances[team_type_filter]),
        friendly_robots={rid: SimpleNamespace(has_ball=False) for rid in ALL},
        enemy_robots={2: SimpleNamespace(has_ball=False)},
    )


def _pick(game, prev_attack_n):
    """Previous partition with `prev_attack_n` robots on attack (rest on defense), or no history."""
    prev = None
    if prev_attack_n is not None:
        ids = sorted(ALL)
        prev = {"attack": frozenset(ids[:prev_attack_n]), "defense": frozenset(ids[prev_attack_n:])}
    return _sticky_split_picker(game, ALL, prev, ALL_TACTICS)


def _attack_size(partition):
    return len(partition.get("attack", ()))


# With 5 free robots: attacking posture is (5+1)//2+1 = 4 attackers, defence posture is 5//2-1 = 1.
ATTACKING, DEFENDING = 4, 1
NEAR_TIE = _game(0.30, 0.30)
WE_WIN = _game(0.10, 0.30)
THEY_WIN = _game(0.30, 0.10)


def test_near_tie_keeps_attacking_posture_once_held():
    # Held attack last tick (4 vs 1): a None edge must not hand the ball back.
    assert _attack_size(_pick(NEAR_TIE, ATTACKING)) == ATTACKING


def test_near_tie_keeps_defending_posture_when_not_held():
    # Last tick was defence-heavy (1 vs 4): a None edge must not flip to attack.
    assert _attack_size(_pick(NEAR_TIE, DEFENDING)) == DEFENDING


def test_no_history_near_tie_defends():
    assert _attack_size(_pick(NEAR_TIE, None)) == DEFENDING


def test_clear_loss_gives_up_attacking_posture():
    assert _attack_size(_pick(THEY_WIN, ATTACKING)) == DEFENDING


def test_clear_gain_is_needed_to_take_attacking_posture():
    assert _attack_size(_pick(WE_WIN, DEFENDING)) == ATTACKING
    assert _attack_size(_pick(WE_WIN, None)) == ATTACKING


def test_clear_loss_does_not_flip_defending_posture_further():
    assert _attack_size(_pick(THEY_WIN, DEFENDING)) == DEFENDING


def test_split_sizes_match_split_shape():
    # Same split as split_shape: attack (n+1)//2+1 with the edge, n//2-1 without.
    for n in range(1, 6):
        ids = frozenset(range(1, n + 1))
        game = _game(0.10, 0.30)
        part = _sticky_split_picker(game, ids, None, ALL_TACTICS)
        assert _attack_size(part) == min(n, (n + 1) // 2 + 1)
        assert frozenset().union(*part.values()) == ids
