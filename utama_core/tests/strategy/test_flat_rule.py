"""`flat_rule`: the OpenJev flat engine's scripted rule as a catalog strategy. Round-robins play it
with no server, so it must build the rule, not the model."""

from __future__ import annotations

from types import SimpleNamespace

from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.strategy.flat_rule import build_flat_rule_kernel_strategy
from utama_core.tactics.flat import FlatJevTactic


def test_flat_rule_plays_the_scripted_rule_with_no_server_and_its_own_free_kicks():
    strategy = build_flat_rule_kernel_strategy((1, 2, 3, 4, 5))(SimpleNamespace())
    (tactic,) = strategy._tactics.values()
    assert isinstance(tactic, FlatJevTactic)
    assert tactic.policy == "scripted" and tactic.client is None and tactic.net is None
    assert tactic.period == 1.0  # 1 Hz plus the situation triggers: the rule's best rate
    overrides = strategy._referee_override.overrides
    assert {RefereeCommand.DIRECT_FREE_YELLOW, RefereeCommand.DIRECT_FREE_BLUE} <= set(overrides)
