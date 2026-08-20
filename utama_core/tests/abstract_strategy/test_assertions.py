"""Tests for `AbstractStrategy.assert_field_requirements`.

Covers `get_min_bounding_req`'s two accepted shapes (`FieldBounds`,
`SpaceRequirements`) and the validation `assert_field_requirements` performs
against the actual field bounds. This logic is unchanged from the pre-kernel
`AbstractStrategy` — only the strategy stand-in construction changed (plain
`game`/`motion_controller` attributes now, no blackboard).

The other half of this file's original coverage — `_ResetStrategyOnRefereeStoppage`
/ `_REFEREE_STOPPAGE_COMMANDS` reset-guard tests — tested BT-only machinery
that no longer exists. Its functional equivalent (barrier-reset behavior) is
already covered by `utama_core/tests/engine/test_strategy.py`'s
`test_barrier_reset_clears_mem_and_overrides_commitment` and
`test_barrier_reset_clears_all_tactics_and_unpins_commitments`.
"""

import pytest

from utama_core.config.field_params import STANDARD_FIELD_DIMS
from utama_core.engine.abstract_strategy import AbstractStrategy, SpaceRequirements
from utama_core.entities.game.field import Field, FieldBounds
from utama_core.strategy.kernel_strategy import build_default_kernel_strategy


def make_dummy_game(field_bounds):
    class _Game:
        field = Field(my_team_is_right=True, field_dims=STANDARD_FIELD_DIMS, field_bounds=field_bounds)

    return _Game()


def _strategy(min_bb) -> AbstractStrategy:
    strategy = AbstractStrategy(build_kernel_strategy=build_default_kernel_strategy(()))
    strategy.get_min_bounding_req = lambda: min_bb
    return strategy


def test_normal_case():
    actual_field = FieldBounds(top_left=(-4.5, 3.0), bottom_right=(4.5, -3.0))
    min_bb = FieldBounds(top_left=(-4.0, 2.5), bottom_right=(4.0, -2.5))
    strategy = _strategy(min_bb)
    game = make_dummy_game(actual_field)
    strategy.assert_field_requirements(game)  # should pass


def test_min_bb_none():
    actual_field = FieldBounds(top_left=(-4.5, 3.0), bottom_right=(4.5, -3.0))
    strategy = _strategy(None)
    game = make_dummy_game(actual_field)
    strategy.assert_field_requirements(game)  # should pass


def test_min_bb_outside_field():
    actual_field = FieldBounds(top_left=(-4.5, 3.0), bottom_right=(4.5, -3.0))
    min_bb = FieldBounds(top_left=(-5.0, 3.5), bottom_right=(4.0, -2.5))
    strategy = _strategy(min_bb)
    game = make_dummy_game(actual_field)
    with pytest.raises(ValueError):
        strategy.assert_field_requirements(game)


def test_crossed_bounding_box():
    actual_field = FieldBounds(top_left=(-4.5, 3.0), bottom_right=(4.5, -3.0))
    min_bb = FieldBounds(top_left=(1.0, -1.0), bottom_right=(-1.0, 1.0))  # crossed
    strategy = _strategy(min_bb)
    game = make_dummy_game(actual_field)
    with pytest.raises(ValueError):
        strategy.assert_field_requirements(game)


def test_min_bb_exceeds_full_field():
    actual_field = FieldBounds(top_left=(-4.5, 3.0), bottom_right=(4.5, -3.0))
    min_bb = FieldBounds(top_left=(-5.0, 4.0), bottom_right=(5.0, -4.0))
    strategy = _strategy(min_bb)
    game = make_dummy_game(actual_field)
    with pytest.raises(ValueError):
        strategy.assert_field_requirements(game)


def test_min_bb_not_contained_in_actual_field():
    actual_field = FieldBounds(top_left=(-2.0, 2.0), bottom_right=(2.0, -2.0))
    min_bb = FieldBounds(top_left=(-3.0, 1.5), bottom_right=(1.0, -1.5))
    strategy = _strategy(min_bb)
    game = make_dummy_game(actual_field)
    with pytest.raises(ValueError, match="does not contain"):
        strategy.assert_field_requirements(game)


def test_space_requirements_rejected_when_field_too_small():
    actual_field = FieldBounds(top_left=(-2.0, 1.0), bottom_right=(2.0, -1.0))
    strategy = _strategy(SpaceRequirements(min_length=4.5, min_width=2.5))
    game = make_dummy_game(actual_field)
    with pytest.raises(ValueError, match="too small for strategy"):
        strategy.assert_field_requirements(game)
