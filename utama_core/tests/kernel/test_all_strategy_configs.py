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

from utama_core.kernel.kernel_strategy import (
    _fixed_ratio_picker,
    _three_way_picker,
    build_decoy_and_overload_kernel_strategy,
    build_give_and_go_solo_kernel_strategy,
    build_high_press_kernel_strategy,
    build_low_block_kernel_strategy,
    build_press_and_pass_kernel_strategy,
    build_split_shape_kernel_strategy,
    build_three_slot_kernel_strategy,
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
