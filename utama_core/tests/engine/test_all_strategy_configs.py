"""Comprehensive, table-driven tests over every strategy
in `utama_core/strategy/`.

Replaces the earlier one-file-per-config pattern (`test_pendulum_kernel_strategy.py`,
`test_terrier_kernel_strategy.py`, `test_wolfpack_kernel_strategy.py`,
`test_ricochet_kernel_strategy.py`), each of which re-asserted the same
handful of generic properties ("builds a Strategy", "runs N ticks without
error", "partition covers every outfield robot", "goalkeeper stays pinned")
around a near-identical `StrategyRunner`/rsim fixture. A single parametrized
suite over `_CONFIGS` gives the same coverage without the duplication.

Does not replace `test_kernel_strategy.py` (tests the `AbstractStrategy`
adapter/blackboard wiring itself, via `default` as one
convenient example config — not "does this config's partition behave
correctly") or `test_kernel_strategy_debug_status.py` (tests `debug_status()`
mechanics specifically). Picker-level unit tests that don't need rsim at all
(e.g. `fixed_ratio_picker`'s exact split ratios) also stay separate, below, since they're cheap, precise, and
don't belong in a `StrategyRunner`-driven table.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from utama_core.engine.abstract_strategy import AbstractStrategy
from utama_core.engine.strategy import Strategy
from utama_core.entities.data.object import TeamType
from utama_core.strategy import catalog
from utama_core.strategy.breakwater import _breakwater_picker
from utama_core.strategy.catalog import builder
from utama_core.strategy.firebreak import _firebreak_picker
from utama_core.strategy.pickers import fixed_ratio_picker
from utama_core.strategy.riptide import _riptide_picker

_OUTFIELD_IDS = (1, 2, 3, 4, 5)

# Every strategy, as round-robins discover them, so a new one is covered without editing this
# file (a strategy branch can't: see tools/check_strategy_branch.py).
_CONFIGS = [pytest.param(name, id=name) for name in catalog.names() if name != "default"]


@pytest.fixture
def make_runner():
    from utama_core.run.strategy_runner import StrategyRunner

    made = []

    def _make(name):
        strategy = AbstractStrategy(build=catalog.builder(name, _OUTFIELD_IDS))
        r = StrategyRunner(
            strategy=strategy,
            my_team_is_yellow=True,
            my_team_is_right=True,
            mode="rsim",
            exp_friendly=6,
            exp_enemy=3,
            exp_ball=True,
        )
        made.append(r)
        return r

    yield _make
    for r in made:
        r.close()


@pytest.mark.parametrize("name", _CONFIGS)
def test_partitions_every_outfield_robot_and_never_the_keeper_for_30_ticks(name, make_runner):
    """30 ticks, not 5-10: enough for PressAndContainTactic's applicable()
    to plausibly flip in configs that use it, exercising the exact hazard
    `_terrier_split_picker`/`_three_way_picker` exist to avoid (a
    Partitioner proposing robots for a currently-inapplicable tactic)."""
    runner = make_runner(name)
    strategy = runner.my.strategy
    assert isinstance(strategy._kernel_strategy, Strategy)
    assert strategy._goalkeeper_id == 0
    for tick in range(30):
        runner.step_once()
        partition = strategy._kernel_strategy.active_partition
        all_assigned = frozenset().union(*partition.values()) if partition else frozenset()
        assert all_assigned == frozenset(_OUTFIELD_IDS), f"tick {tick}: {partition}"


# --- picker-level unit tests (no rsim needed) ---


# The pickers only read `friendly_robots` (to put a ball carrier first) here.
_NOBODY_ON_BALL = SimpleNamespace(friendly_robots={})


def test_fixed_ratio_picker_wolfpack_split():
    picker = fixed_ratio_picker("attack", "defense", attack_fraction=0.8)
    partition = picker(
        game=_NOBODY_ON_BALL,
        free_robots=frozenset({1, 2, 3, 4, 5}),
        prev_partition=None,
        available_tactic_ids=frozenset({"attack", "defense"}),
    )
    assert len(partition["attack"]) == 4
    assert len(partition["defense"]) == 1


def test_fixed_ratio_picker_defense_heavy_split_respects_min_attack_floor():
    """attack_fraction=0.2 on 5 robots rounds to 1, but PassAndShootTactic
    hard-requires >=2 (it unconditionally reads robot_ids[1]) — min_attack=2
    floors the split rather than letting a defense-heavy
    split crash the tactic it wires (the exact regression this test guards)."""
    picker = fixed_ratio_picker("attack", "defense", attack_fraction=0.2, min_attack=2)
    partition = picker(
        game=_NOBODY_ON_BALL,
        free_robots=frozenset({1, 2, 3, 4, 5}),
        prev_partition=None,
        available_tactic_ids=frozenset({"attack", "defense"}),
    )
    assert len(partition["attack"]) == 2
    assert len(partition["defense"]) == 3


def test_ricochet_always_assigns_everyone_to_attack(make_runner):
    runner = make_runner("ricochet")
    runner.step_once()
    partition = runner.my.strategy._kernel_strategy.active_partition
    assert partition == {"attack": frozenset(_OUTFIELD_IDS)}


# --- arena-picker unit tests (stub game: the pickers read live state) ---


class _StubProximityLookup:
    """Stand-in for `Game.proximity_lookup`: fixed per-team closest distances."""

    def __init__(self, friendly_dist, enemy_dist):
        self._dists = {TeamType.FRIENDLY: friendly_dist, TeamType.ENEMY: enemy_dist}

    def closest_to_ball(self, team_type_filter=None):
        return (None, self._dists.get(team_type_filter))


def _stub_game(friendly_dist, enemy_dist, ball_x: float, my_team_is_right: bool = True):
    """Minimal fake `Game` exposing what `_tiki_taka_picker` and friends read:
    proximity lookup, side sign, field half-length (standard 4.5 m), ball x, and
    no robot holding the ball.

    With my_team_is_right (attacking -x), ball_x < -1.5 is the final third,
    -1.5..1.5 the middle, > 1.5 our own third (thirds = 2*4.5/3 = 3 m each).
    """
    from types import SimpleNamespace

    return SimpleNamespace(
        proximity_lookup=_StubProximityLookup(friendly_dist, enemy_dist),
        my_team_is_right=my_team_is_right,
        field=SimpleNamespace(half_length=4.5),
        ball=SimpleNamespace(
            p=SimpleNamespace(x=ball_x, y=0.0, to_2d=lambda: SimpleNamespace(x=ball_x)), v=SimpleNamespace(x=0.0, y=0.0)
        ),
        friendly_robots={},
        enemy_robots={},
    )


_FIVE = frozenset({1, 2, 3, 4, 5})


_CLEAR_ALL = frozenset({"attack", "press", "block", "clear"})


def test_firebreak_valve_takes_one_robot_when_applicable():
    """Danger read fires (kernel reports 'clear' applicable): 1 clearer + block
    screen with the rest, regardless of the possession edge."""
    partition = _firebreak_picker(
        _stub_game(friendly_dist=2.0, enemy_dist=0.3, ball_x=3.0),  # losing, deep
        _FIVE,
        None,
        _CLEAR_ALL,
    )
    assert len(partition["clear"]) == 1
    assert len(partition["block"]) == 4
    assert frozenset().union(*partition.values()) == _FIVE


def test_firebreak_valve_folds_everyone_in_when_block_unavailable():
    partition = _firebreak_picker(
        _stub_game(friendly_dist=2.0, enemy_dist=0.3, ball_x=3.0),
        _FIVE,
        None,
        frozenset({"clear"}),  # block pinned/inapplicable
    )
    assert partition == {"clear": _FIVE}


def test_firebreak_uses_riptide_press_posture_when_lost_no_danger():
    partition = _firebreak_picker(
        _stub_game(friendly_dist=1.5, enemy_dist=0.2, ball_x=0.0),
        _FIVE,
        None,
        _CLEAR_ALL - {"clear"},  # no danger: valve out of the candidate pool
    )
    assert len(partition["press"]) == 3
    assert len(partition["block"]) == 2


def test_firebreak_attacks_with_three_when_won_no_danger():
    partition = _firebreak_picker(
        _stub_game(friendly_dist=0.3, enemy_dist=1.5, ball_x=0.0),
        _FIVE,
        None,
        _CLEAR_ALL - {"clear"},
    )
    assert len(partition["attack"]) == 3
    assert len(partition["block"]) == 2


def test_firebreak_attack_stays_on_unknown_possession_edge():
    """Sticky possession edge (same fix riptide/high_line_zone needed):
    while we held attack last tick, an unreadable edge must not reset the
    give-and-go trio — only a clear loss (enemy strictly closer) does."""
    prev = {"attack": frozenset({1, 2, 3}), "block": frozenset({4, 5})}
    partition = _firebreak_picker(
        _stub_game(friendly_dist=None, enemy_dist=None, ball_x=0.0),
        _FIVE,
        prev,
        _CLEAR_ALL - {"clear"},
    )
    assert len(partition["attack"]) == 3


def test_firebreak_valve_overrides_standing_attack():
    """Danger fires while we were attacking last tick: the valve still takes its
    robot — danger is depth+pressure, not just possession."""
    prev = {"attack": frozenset({1, 2, 3}), "block": frozenset({4, 5})}
    partition = _firebreak_picker(
        _stub_game(friendly_dist=0.5, enemy_dist=1.5, ball_x=3.0),  # we're closer BUT deep+contested
        _FIVE,
        prev,
        _CLEAR_ALL,
    )
    assert len(partition["clear"]) == 1


# `ClearBallTactic` never commits (`is_committed()` is always False, by design: a
# clearance takes about a second and the clearer is simply re-picked each tick), so
# a "clear" robot is back in `free_robots` on the next tick. The pickers used to
# keep a "clearer still pinned" branch that held everyone on the block screen when
# last tick's "clear" robots were all absent from `free_robots`; no state the
# scheduler can produce for this tactic reaches it, and a robot that merely left the
# free pool (e.g. one off the field) would have set it off. These tests pin that the
# picker now decides from the possession edge alone.


def test_firebreak_ignores_a_prior_clear_robot_missing_from_the_free_pool():
    free = frozenset({2, 3, 4, 5})  # robot 1 held "clear" last tick, is not free now
    prev = {"clear": frozenset({1})}
    partition = _firebreak_picker(
        _stub_game(friendly_dist=0.2, enemy_dist=1.5, ball_x=0.0),  # we're closer: attack
        free,
        prev,
        _CLEAR_ALL - {"clear"},
    )
    assert len(partition["attack"]) == 3
    assert len(partition["block"]) == 1


def test_breakwater_ignores_a_prior_clear_robot_missing_from_the_free_pool():
    free = frozenset({2, 3, 4, 5})
    prev = {"clear": frozenset({1})}
    partition = _breakwater_picker(
        _stub_game(friendly_dist=0.2, enemy_dist=1.5, ball_x=0.0),
        free,
        prev,
        (_CLEAR_ALL | {"overload"}) - {"clear"},
    )
    assert len(partition["attack"]) == 3
    assert len(partition["block"]) == 1


def _clear_game(positions: dict[int, tuple[float, float]], carrier: int | None = None, ball=(3.0, 0.0)):
    """Danger-zone game stub with robots at `positions` and the ball at `ball`."""
    game = _stub_game(friendly_dist=2.0, enemy_dist=0.3, ball_x=ball[0])
    game.ball.p.y = ball[1]
    game.friendly_robots = {
        rid: SimpleNamespace(p=SimpleNamespace(x=x, y=y), has_ball=(rid == carrier))
        for rid, (x, y) in positions.items()
    }
    return game


# Robot 1 is the lowest id but the farthest from the ball at (3, 0); robot 4 is nearest.
_SPREAD = {1: (-2.0, 1.0), 2: (0.0, -1.0), 3: (1.0, 1.0), 4: (2.7, 0.2), 5: (-1.0, 0.0)}


def test_firebreak_valve_goes_to_the_robot_nearest_the_ball():
    partition = _firebreak_picker(_clear_game(_SPREAD), _FIVE, None, _CLEAR_ALL)
    assert partition["clear"] == frozenset({4})


def test_breakwater_valve_goes_to_the_robot_nearest_the_ball():
    partition = _breakwater_picker(_clear_game(_SPREAD), _FIVE, None, _CLEAR_ALL | {"overload"})
    assert partition["clear"] == frozenset({4})


def test_clear_valve_keeps_the_carrier_even_when_another_robot_is_nearer():
    # Robot 2 holds the ball (so it is first from `carrier_first`); robot 4 happens to
    # be nearer the ball's reported position. The carrier keeps the clearance.
    game = _clear_game(_SPREAD, carrier=2)
    assert _firebreak_picker(game, _FIVE, None, _CLEAR_ALL)["clear"] == frozenset({2})
    assert _breakwater_picker(game, _FIVE, None, _CLEAR_ALL | {"overload"})["clear"] == frozenset({2})


def test_clear_valve_breaks_a_distance_tie_by_lowest_id():
    tied = {1: (2.0, 1.0), 2: (2.0, -1.0), 3: (-2.0, 0.0), 4: (-2.0, 1.0), 5: (-2.0, -1.0)}
    partition = _firebreak_picker(_clear_game(tied), _FIVE, None, _CLEAR_ALL)
    assert partition["clear"] == frozenset({1})


# --- sticky possession edge: a near-tie must keep the previous posture ---

_COUNTER_FLOW_ALL = frozenset({"attack", "press", "block"})


def test_riptide_keeps_attack_through_a_near_tie_for_the_ball():
    """Friendly 3 cm behind the nearest enemy is inside `CLOSER_TO_BALL_MARGIN`:
    undecided, so a side that held attack last tick keeps it (the documented
    hysteresis). `friendly_closer_to_ball` used to return False here, which
    dropped attack on noise-level gaps."""
    prev = {"attack": frozenset({1, 2, 3}), "block": frozenset({4, 5})}
    partition = _riptide_picker(
        _stub_game(friendly_dist=0.33, enemy_dist=0.30, ball_x=0.0), _FIVE, prev, _COUNTER_FLOW_ALL
    )
    assert len(partition["attack"]) == 3


def test_riptide_without_prior_attack_stays_conservative_on_a_near_tie():
    partition = _riptide_picker(
        _stub_game(friendly_dist=0.33, enemy_dist=0.30, ball_x=0.0), _FIVE, {}, _COUNTER_FLOW_ALL
    )
    assert "attack" not in partition


def test_riptide_drops_attack_on_a_clear_loss():
    prev = {"attack": frozenset({1, 2, 3}), "block": frozenset({4, 5})}
    partition = _riptide_picker(
        _stub_game(friendly_dist=0.5, enemy_dist=0.30, ball_x=0.0), _FIVE, prev, _COUNTER_FLOW_ALL
    )
    assert "attack" not in partition
