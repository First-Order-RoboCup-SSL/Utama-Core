"""`split_shape` keeps a second robot in attack while defending, from four outfield robots up, so a
won ball has someone upfield to go to."""

import pytest

from utama_core.strategy import split_shape

_TACTICS = frozenset({"attack", "defense"})


@pytest.mark.parametrize("robots, attackers", [(5, 2), (4, 2), (3, 0)])
def test_defending_keeps_a_second_attacker_from_four_robots(monkeypatch, robots, attackers):
    monkeypatch.setattr(split_shape, "carrier_first", lambda game, free: sorted(free))
    monkeypatch.setattr(split_shape, "friendly_closer_to_ball", lambda game: False)
    free = frozenset(range(1, robots + 1))

    partition = split_shape._possession_split_picker(None, free, None, _TACTICS)

    assert len(partition.get("attack", ())) == attackers
    assert partition.get("attack", frozenset()) | partition["defense"] == free
