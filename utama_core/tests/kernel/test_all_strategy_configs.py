"""Comprehensive, table-driven tests over every `build_*_kernel_strategy` factory
in `kernel_strategy.py`.

Replaces the earlier one-file-per-config pattern (`test_split_shape_kernel_strategy.py`,
`test_press_and_pass_kernel_strategy.py`, `test_high_press_kernel_strategy.py`,
`test_low_block_kernel_strategy.py`, `test_three_slot_kernel_strategy.py`,
`test_give_and_go_solo_kernel_strategy.py`), each of which re-asserted the same
handful of generic properties ("builds a Strategy", "runs N ticks without
error", "partition covers every outfield robot", "goalkeeper stays pinned")
around a near-identical `StrategyRunner`/rsim fixture. A single parametrized
suite over `_CONFIGS` gives the same coverage without the duplication.

Does not replace `test_kernel_strategy.py` (tests the `AbstractStrategy`
adapter/blackboard wiring itself, via `build_default_kernel_strategy` as one
convenient example config — not "does this config's partition behave
correctly") or `test_kernel_strategy_debug_status.py` (tests `debug_status()`
mechanics specifically). Picker-level unit tests that don't need rsim at all
(e.g. `_fixed_ratio_picker`'s exact split ratios, `_three_way_picker`'s slot
allocation) also stay separate, below, since they're cheap, precise, and
don't belong in a `StrategyRunner`-driven table.
"""

from __future__ import annotations

import pytest

from utama_core.entities.data.object import TeamType
from utama_core.kernel.kernel_strategy import (
    _counter_press_picker,
    _fixed_ratio_picker,
    _three_way_picker,
    _tiki_taka_picker,
    _zone_flow_picker,
    build_counter_press_kernel_strategy,
    build_decoy_and_overload_kernel_strategy,
    build_give_and_go_solo_kernel_strategy,
    build_high_press_kernel_strategy,
    build_low_block_kernel_strategy,
    build_press_and_pass_kernel_strategy,
    build_split_shape_kernel_strategy,
    build_three_slot_kernel_strategy,
    build_tiki_taka_kernel_strategy,
    build_zone_fluid_kernel_strategy,
)
from utama_core.kernel.strategy import Strategy
from utama_core.strategy.common.abstract_strategy import AbstractStrategy

_OUTFIELD_IDS = (1, 2, 3, 4, 5)

_CONFIGS = [
    pytest.param(build_split_shape_kernel_strategy, id="split_shape"),
    pytest.param(build_press_and_pass_kernel_strategy, id="press_and_pass"),
    pytest.param(build_high_press_kernel_strategy, id="high_press"),
    pytest.param(build_low_block_kernel_strategy, id="low_block"),
    pytest.param(build_three_slot_kernel_strategy, id="three_slot"),
    pytest.param(build_give_and_go_solo_kernel_strategy, id="give_and_go_solo"),
    pytest.param(build_decoy_and_overload_kernel_strategy, id="decoy_and_overload"),
    pytest.param(build_tiki_taka_kernel_strategy, id="tiki_taka"),
    pytest.param(build_counter_press_kernel_strategy, id="counter_press"),
    pytest.param(build_zone_fluid_kernel_strategy, id="zone_fluid"),
]


@pytest.fixture
def make_runner():
    from utama_core.run.strategy_runner import StrategyRunner

    made = []

    def _make(build_kernel_strategy_factory):
        strategy = AbstractStrategy(build_kernel_strategy=build_kernel_strategy_factory(_OUTFIELD_IDS))
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


@pytest.mark.parametrize("build_factory", _CONFIGS)
def test_builds_a_kernel_strategy(build_factory, make_runner):
    runner = make_runner(build_factory)
    assert isinstance(runner.my.strategy._kernel_strategy, Strategy)


@pytest.mark.parametrize("build_factory", _CONFIGS)
def test_steps_without_error_for_many_ticks(build_factory, make_runner):
    """30 ticks, not 5-10: enough for PressAndContainTactic's applicable()
    to plausibly flip in configs that use it, exercising the exact hazard
    `_press_and_pass_split_picker`/`_three_way_picker` exist to avoid (a
    Partitioner proposing robots for a currently-inapplicable tactic)."""
    runner = make_runner(build_factory)
    for _ in range(30):
        runner.step_once()


@pytest.mark.parametrize("build_factory", _CONFIGS)
def test_partition_covers_all_outfield_robots_every_tick(build_factory, make_runner):
    runner = make_runner(build_factory)
    runner.step_once()
    partition = runner.my.strategy._kernel_strategy.active_partition
    all_assigned = frozenset().union(*partition.values()) if partition else frozenset()
    assert all_assigned == frozenset(_OUTFIELD_IDS)


@pytest.mark.parametrize("build_factory", _CONFIGS)
def test_goalkeeper_never_appears_in_the_partition(build_factory, make_runner):
    runner = make_runner(build_factory)
    strategy = runner.my.strategy
    assert strategy._goalkeeper_id == 0
    runner.step_once()
    partition = strategy._kernel_strategy.active_partition
    for robots in partition.values():
        assert 0 not in robots


# --- picker-level unit tests (no rsim needed) ---


def test_fixed_ratio_picker_high_press_split():
    picker = _fixed_ratio_picker("attack", "defense", attack_fraction=0.8)
    partition = picker(
        game=None,
        free_robots=frozenset({1, 2, 3, 4, 5}),
        prev_partition=None,
        applicable_tactic_ids=frozenset({"attack", "defense"}),
    )
    assert len(partition["attack"]) == 4
    assert len(partition["defense"]) == 1


def test_fixed_ratio_picker_low_block_split_respects_min_attack_floor():
    """attack_fraction=0.2 on 5 robots rounds to 1, but PassAndShootTactic
    hard-requires >=2 (it unconditionally reads robot_ids[1]) — min_attack=2
    floors the split rather than letting `build_low_block_kernel_strategy`
    crash the tactic it wires (the exact regression this test guards)."""
    picker = _fixed_ratio_picker("attack", "defense", attack_fraction=0.2, min_attack=2)
    partition = picker(
        game=None,
        free_robots=frozenset({1, 2, 3, 4, 5}),
        prev_partition=None,
        applicable_tactic_ids=frozenset({"attack", "defense"}),
    )
    assert len(partition["attack"]) == 2
    assert len(partition["defense"]) == 3


def test_three_way_picker_splits_all_three_slots_when_all_applicable():
    partition = _three_way_picker(
        game=None,
        free_robots=frozenset({1, 2, 3, 4, 5}),
        prev_partition=None,
        applicable_tactic_ids=frozenset({"press", "mark", "attack"}),
    )
    assert len(partition["press"]) == 1
    assert len(partition["mark"]) == 2
    assert len(partition["attack"]) == 2
    assert frozenset().union(*partition.values()) == frozenset({1, 2, 3, 4, 5})


def test_three_way_picker_falls_back_to_attack_when_press_and_mark_inapplicable():
    partition = _three_way_picker(
        game=None,
        free_robots=frozenset({1, 2, 3}),
        prev_partition=None,
        applicable_tactic_ids=frozenset({"attack"}),
    )
    assert partition == {"attack": frozenset({1, 2, 3})}


def test_give_and_go_solo_always_assigns_everyone_to_attack(make_runner):
    runner = make_runner(build_give_and_go_solo_kernel_strategy)
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
    proximity lookup, side sign, field half-length (standard 4.5 m), ball x.

    With my_team_is_right (attacking -x), ball_x < -1.5 is the final third,
    -1.5..1.5 the middle, > 1.5 our own third (thirds = 2*4.5/3 = 3 m each).
    """
    from types import SimpleNamespace

    return SimpleNamespace(
        proximity_lookup=_StubProximityLookup(friendly_dist, enemy_dist),
        my_team_is_right=my_team_is_right,
        field=SimpleNamespace(half_length=4.5),
        ball=SimpleNamespace(p=SimpleNamespace(to_2d=lambda: SimpleNamespace(x=ball_x))),
    )


_FIVE = frozenset({1, 2, 3, 4, 5})
_ALL = frozenset({"attack", "defense", "press", "givego", "overload", "block"})


def test_tiki_taka_attacks_with_three_when_ball_is_ours():
    partition = _tiki_taka_picker(
        _stub_game(friendly_dist=0.2, enemy_dist=1.5, ball_x=0.0),
        _FIVE,
        None,
        _ALL,
    )
    assert len(partition["attack"]) == 3
    assert len(partition["defense"]) == 2
    assert frozenset().union(*partition.values()) == _FIVE


def test_tiki_taka_presses_with_three_when_ball_is_lost():
    partition = _tiki_taka_picker(
        _stub_game(friendly_dist=1.5, enemy_dist=0.2, ball_x=0.0),
        _FIVE,
        None,
        _ALL,
    )
    assert len(partition["press"]) == 3
    assert len(partition["defense"]) == 2


def test_tiki_taka_unknown_edge_falls_back_to_conservative_press():
    # Unknown possession edge (both None) must not send everyone forward.
    partition = _tiki_taka_picker(
        _stub_game(friendly_dist=None, enemy_dist=None, ball_x=0.0),
        _FIVE,
        None,
        _ALL,
    )
    assert len(partition["press"]) == 3
    assert len(partition["defense"]) == 2


def test_tiki_taka_press_inapplicable_covers_with_defense():
    partition = _tiki_taka_picker(
        _stub_game(friendly_dist=1.5, enemy_dist=0.2, ball_x=0.0),
        _FIVE,
        None,
        frozenset({"attack", "defense"}),  # press out of pressing range
    )
    assert partition == {"defense": _FIVE}


def test_tiki_taka_attack_pinned_folds_share_into_defense():
    # Give-and-go committed mid-hop: its robots are pinned (not in the free
    # pool) and "attack" never appears in applicable_tactic_ids — everyone
    # free must still land in a legal slot.
    partition = _tiki_taka_picker(
        _stub_game(friendly_dist=0.2, enemy_dist=1.5, ball_x=-3.0),
        _FIVE,
        None,
        frozenset({"defense", "press"}),
    )
    assert partition == {"defense": _FIVE}


def test_counter_press_everyone_presses_when_lost():
    partition = _counter_press_picker(
        _stub_game(friendly_dist=2.0, enemy_dist=0.3, ball_x=0.0),
        _FIVE,
        None,
        _ALL,
    )
    assert partition == {"press": _FIVE}


def test_counter_press_drops_into_low_block_when_nothing_to_press():
    partition = _counter_press_picker(
        _stub_game(friendly_dist=2.0, enemy_dist=0.3, ball_x=0.0),
        _FIVE,
        None,
        frozenset({"attack", "block"}),  # opponent shielded from the press
    )
    assert partition == {"block": _FIVE}


def test_counter_press_attacks_with_four_and_one_screen_when_won():
    partition = _counter_press_picker(
        _stub_game(friendly_dist=0.3, enemy_dist=2.0, ball_x=-3.0),
        _FIVE,
        None,
        _ALL,
    )
    assert len(partition["attack"]) == 4
    assert len(partition["block"]) == 1


def test_counter_press_won_ball_without_block_sends_all_five_forward():
    partition = _counter_press_picker(
        _stub_game(friendly_dist=0.3, enemy_dist=2.0, ball_x=-3.0),
        _FIVE,
        None,
        frozenset({"attack", "press"}),
    )
    assert partition == {"attack": _FIVE}


def test_zone_flow_everyone_takes_shape_when_ball_lost():
    partition = _zone_flow_picker(
        _stub_game(friendly_dist=2.0, enemy_dist=0.3, ball_x=0.0),
        _FIVE,
        None,
        _ALL,
    )
    assert partition == {"defense": _FIVE}


def test_zone_flow_givego_trio_builds_up_in_middle_third():
    partition = _zone_flow_picker(
        _stub_game(friendly_dist=0.3, enemy_dist=2.0, ball_x=0.0),  # mid third
        _FIVE,
        None,
        _ALL,
    )
    assert len(partition["givego"]) == 3
    assert len(partition["defense"]) == 2


def test_zone_flow_overload_pair_swaps_in_for_final_third():
    partition = _zone_flow_picker(
        _stub_game(friendly_dist=0.3, enemy_dist=2.0, ball_x=-3.0),  # final third
        _FIVE,
        None,
        _ALL,
    )
    assert partition["overload"] == frozenset({1, 2})
    assert len(partition["defense"]) == 3


def test_zone_flow_overload_pinned_builds_with_givego_instead():
    partition = _zone_flow_picker(
        _stub_game(friendly_dist=0.3, enemy_dist=2.0, ball_x=-3.0),  # final third
        _FIVE,
        None,
        frozenset({"givego", "defense"}),  # decoy/overload duet elsewhere
    )
    assert len(partition["givego"]) == 3
    assert len(partition["defense"]) == 2
