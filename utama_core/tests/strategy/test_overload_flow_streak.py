"""`overload_flow` adds a 4th give-and-go attacker once its team has held the
possession edge for `_POSSESSION_STREAK_TICKS` ticks. The streak was one
module-level counter, so two `overload_flow` teams in one process (a mirror match,
a bench A/B against itself) counted each other's possession.
"""

from unittest.mock import MagicMock

from utama_core.strategy import overload_flow
from utama_core.strategy.overload_flow import (
    _POSSESSION_STREAK_TICKS,
    build_overload_flow_kernel_strategy,
)

_ROBOTS = frozenset({1, 2, 3, 4, 5})
_TACTICS = frozenset({"givego", "overload", "defense"})


def _picker():
    return build_overload_flow_kernel_strategy((1, 2, 3, 4, 5))(MagicMock())._partitioner


def test_each_strategy_counts_its_own_possession_streak(monkeypatch):
    monkeypatch.setattr(overload_flow, "carrier_first", lambda game, free: sorted(free))
    monkeypatch.setattr(overload_flow, "friendly_closer_to_ball", lambda game: True)
    monkeypatch.setattr(overload_flow, "ball_zone", lambda game: "middle")
    a, b = _picker(), _picker()

    for _ in range(_POSSESSION_STREAK_TICKS - 1):
        assert len(a(None, _ROBOTS, None, _TACTICS)["givego"]) == 3
    assert len(a(None, _ROBOTS, None, _TACTICS)["givego"]) == 4
    # b has had the ball for one tick, whatever a's streak.
    assert len(b(None, _ROBOTS, None, _TACTICS)["givego"]) == 3
