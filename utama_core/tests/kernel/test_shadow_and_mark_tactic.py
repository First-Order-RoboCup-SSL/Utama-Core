"""Integration test for `ShadowAndMarkTactic` — drives it via a real
`StrategyRunner`/rsim `Game`, following `test_defense_tactic.py`'s pattern.
"""

from __future__ import annotations

import pytest

from utama_core.kernel.context import KernelContext
from utama_core.tactics.shadow_and_mark import ShadowAndMarkTactic


@pytest.fixture
def runner():
    from utama_core.run.strategy_runner import StrategyRunner
    from utama_core.tests.strategy_runner.strat_runner_test_utils import DummyStrategy

    r = StrategyRunner(
        strategy=DummyStrategy(),
        my_team_is_yellow=True,
        my_team_is_right=True,
        mode="rsim",
        exp_friendly=5,
        exp_enemy=3,
        exp_ball=True,
    )
    yield r
    r.close()


@pytest.fixture
def game(runner):
    return runner.my.game


def _ctx(runner) -> KernelContext:
    motion_controller = runner.my.motion_controller(runner.mode, runner.rsim_env)
    return KernelContext(motion_controller=motion_controller, rsim_env=runner.rsim_env)


def test_two_shadow_defenders_only(game, runner):
    tactic = ShadowAndMarkTactic()
    mem = tactic.initial_mem()
    commands, mem = tactic.tick(game, _ctx(runner), (1, 2), mem)
    assert set(commands.keys()) == {1, 2}


def test_extra_defenders_get_marking_commands(game, runner):
    tactic = ShadowAndMarkTactic()
    mem = tactic.initial_mem()
    commands, mem = tactic.tick(game, _ctx(runner), (1, 2, 3, 4), mem)
    assert set(commands.keys()) == {1, 2, 3, 4}


@pytest.fixture
def outnumbering_runner():
    """A fixture with more outfield markers (4, ids 1-4) than enemies (1)."""
    from utama_core.run.strategy_runner import StrategyRunner
    from utama_core.tests.strategy_runner.strat_runner_test_utils import DummyStrategy

    r = StrategyRunner(
        strategy=DummyStrategy(),
        my_team_is_yellow=True,
        my_team_is_right=True,
        mode="rsim",
        exp_friendly=5,
        exp_enemy=1,
        exp_ball=True,
    )
    yield r
    r.close()


def test_more_markers_than_enemies_still_produces_commands_for_all(outnumbering_runner):
    game = outnumbering_runner.my.game
    tactic = ShadowAndMarkTactic()
    mem = tactic.initial_mem()
    commands, mem = tactic.tick(game, _ctx(outnumbering_runner), (1, 2, 3, 4), mem)
    assert set(commands.keys()) == {1, 2, 3, 4}


def test_unmatched_markers_hold_clear_of_own_defense_area(outnumbering_runner):
    """Regression test: an unmatched marker's fallback used to call
    `defend_parameter` again, converging on the same post as a real shadow
    defender and violating the "max 1 non-goalkeeper defender in own area"
    rule (found via a live grsim run — see `shadow_and_mark.py`'s docstring).
    With 1 enemy and 4 friendly outfield robots, robots 3 and 4 both fall
    back — their targets must land outside our own defense area."""
    game = outnumbering_runner.my.game

    from utama_core.tactics.shadow_and_mark import _fallback_hold_target

    fallback_0 = _fallback_hold_target(game, 0)
    fallback_1 = _fallback_hold_target(game, 1)

    corners = game.field.my_defense_area
    min_x = min(c[0] for c in corners)
    max_x = max(c[0] for c in corners)
    min_y = min(c[1] for c in corners)
    max_y = max(c[1] for c in corners)

    for target in (fallback_0, fallback_1):
        inside = min_x <= target.x <= max_x and min_y <= target.y <= max_y
        assert not inside, f"fallback target {target} lands inside our own defense area"

    # Also must not coincide with each other (would still cluster two robots).
    assert fallback_0.distance_to(fallback_1) > 0.1


def test_never_commits(game):
    tactic = ShadowAndMarkTactic()
    mem = tactic.initial_mem()
    assert tactic.is_committed(game, mem) is False
