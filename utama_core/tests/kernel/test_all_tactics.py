"""Comprehensive, table-driven tests over every registered `Tactic`.

Replaces the earlier one-file-per-Tactic pattern (`test_defense_tactic.py`,
`test_lead_and_support_tactic.py`, `test_shadow_and_mark_tactic.py`,
`test_press_and_contain_tactic.py`, `test_give_and_go_tactic.py`), which had
each file re-asserting the same handful of generic properties ("produces a
command for every assigned robot", "is_committed() returns a bool") with
near-identical boilerplate around a real `StrategyRunner`/rsim fixture. A
single parametrized suite over `_TACTIC_CASES` below gives the same coverage
without the duplication, and — concretely — would have caught
`TwoRobotAttackTactic`'s `robot_ids[1]` crash (see `build_low_block_kernel_strategy`
in `kernel_strategy.py`) automatically instead of requiring a bespoke test
to notice a Tactic has an undeclared minimum robot count.

Behavior specific to one Tactic (e.g. `LeadAndSupportTactic`'s leader
picked by ball proximity, `ShadowAndMarkTactic`'s fallback-hold-target
regression) stays in its own test — those don't generalize across the
table and shouldn't be forced into it. `TwoRobotAttackTactic`'s pure-logic
`assign_passer_receiver`/`is_committed()` tests also stay separate
(`test_two_robot_attack_tactic.py`) since they need no rsim fixture at all.
"""

from __future__ import annotations

import pytest

from utama_core.kernel.context import KernelContext
from utama_core.kernel.tactic import TacticTag
from utama_core.tactics.defense import DefenseTactic
from utama_core.tactics.give_and_go import GiveAndGoTactic
from utama_core.tactics.lead_and_support import LeadAndSupportTactic
from utama_core.tactics.press_and_contain import PressAndContainTactic
from utama_core.tactics.shadow_and_mark import ShadowAndMarkTactic
from utama_core.tactics.two_robot_attack import TwoRobotAttackTactic

# (tactic_factory, robot_ids, exp_friendly, exp_enemy) — robot_ids and
# exp_friendly/exp_enemy are chosen per-Tactic to respect each one's real
# constraints (e.g. TwoRobotAttackTactic hard-requires >=2 robots; no Tactic
# declares this anywhere today — see design doc §15's "explicitly deferred"
# per-Tactic robot-count-bound item — so it's encoded here instead).
_TACTIC_CASES = [
    pytest.param(DefenseTactic, (1, 2), 3, 0, id="defense"),
    pytest.param(TwoRobotAttackTactic, (1, 2), 3, 0, id="two_robot_attack"),
    pytest.param(LeadAndSupportTactic, (1, 2, 3, 4), 5, 2, id="lead_and_support"),
    pytest.param(ShadowAndMarkTactic, (1, 2, 3, 4), 5, 3, id="shadow_and_mark"),
    pytest.param(PressAndContainTactic, (1, 2, 3), 4, 3, id="press_and_contain"),
    pytest.param(GiveAndGoTactic, (1, 2, 3), 5, 2, id="give_and_go"),
]


@pytest.fixture
def runner_factory():
    from utama_core.run.strategy_runner import StrategyRunner
    from utama_core.tests.strategy_runner.strat_runner_test_utils import DummyStrategy

    made = []

    def _make(exp_friendly: int, exp_enemy: int):
        r = StrategyRunner(
            strategy=DummyStrategy(),
            my_team_is_yellow=True,
            my_team_is_right=True,
            mode="rsim",
            exp_friendly=exp_friendly,
            exp_enemy=exp_enemy,
            exp_ball=True,
        )
        made.append(r)
        return r

    yield _make
    for r in made:
        r.close()


def _ctx(runner) -> KernelContext:
    motion_controller = runner.my.motion_controller(runner.mode, runner.rsim_env)
    return KernelContext(motion_controller=motion_controller)


@pytest.mark.parametrize("tactic_cls, robot_ids, exp_friendly, exp_enemy", _TACTIC_CASES)
def test_declares_a_tag(tactic_cls, robot_ids, exp_friendly, exp_enemy):
    assert isinstance(tactic_cls.tag, TacticTag)


@pytest.mark.parametrize("tactic_cls, robot_ids, exp_friendly, exp_enemy", _TACTIC_CASES)
def test_is_committed_returns_a_bool_before_any_tick(tactic_cls, robot_ids, exp_friendly, exp_enemy):
    tactic = tactic_cls()
    mem = tactic.initial_mem()
    assert isinstance(tactic.is_committed(game=None, mem=mem), bool)


@pytest.mark.parametrize("tactic_cls, robot_ids, exp_friendly, exp_enemy", _TACTIC_CASES)
def test_produces_a_command_for_every_assigned_robot(tactic_cls, robot_ids, exp_friendly, exp_enemy, runner_factory):
    runner = runner_factory(exp_friendly, exp_enemy)
    game = runner.my.game
    tactic = tactic_cls()
    if not tactic.applicable(game):
        pytest.skip(f"{tactic_cls.__name__} reported inapplicable() for this fixture's default game state")
    mem = tactic.initial_mem()
    commands, _mem = tactic.tick(game, _ctx(runner), robot_ids, mem)
    assert set(commands.keys()) == set(robot_ids)


@pytest.mark.parametrize("tactic_cls, robot_ids, exp_friendly, exp_enemy", _TACTIC_CASES)
def test_ticking_twice_does_not_raise(tactic_cls, robot_ids, exp_friendly, exp_enemy, runner_factory):
    """A weak but genuinely useful smoke test: most cross-tick state bugs
    (mem misuse, role-assignment flip-flopping into an inconsistent state)
    only surface on the second call, not the first."""
    runner = runner_factory(exp_friendly, exp_enemy)
    game = runner.my.game
    tactic = tactic_cls()
    if not tactic.applicable(game):
        pytest.skip(f"{tactic_cls.__name__} reported inapplicable() for this fixture's default game state")
    mem = tactic.initial_mem()
    _commands, mem = tactic.tick(game, _ctx(runner), robot_ids, mem)
    _commands, _mem = tactic.tick(game, _ctx(runner), robot_ids, mem)


# --- Tactic-specific behavior that doesn't generalize across the table ---


def test_lead_and_support_leader_is_closest_robot_to_ball(runner_factory):
    runner = runner_factory(exp_friendly=5, exp_enemy=2)
    game = runner.my.game
    tactic = LeadAndSupportTactic()
    mem = tactic.initial_mem()
    _commands, mem = tactic.tick(game, _ctx(runner), (1, 2, 3, 4), mem)
    assert mem.leader_id in (1, 2, 3, 4)


def test_lead_and_support_single_robot_has_no_supports(runner_factory):
    runner = runner_factory(exp_friendly=5, exp_enemy=2)
    game = runner.my.game
    tactic = LeadAndSupportTactic()
    mem = tactic.initial_mem()
    commands, mem = tactic.tick(game, _ctx(runner), (1,), mem)
    assert set(commands.keys()) == {1}
    assert mem.leader_id == 1


def test_shadow_and_mark_unmatched_markers_hold_clear_of_own_defense_area(runner_factory):
    """Regression test: an unmatched marker's fallback used to call
    `defend_parameter` again, converging on the same post as a real shadow
    defender and violating the "max 1 non-goalkeeper defender in own area"
    rule (found via a live grsim run — see `shadow_and_mark.py`'s docstring).
    With 1 enemy and 4 friendly outfield robots, robots 3 and 4 both fall
    back — their targets must land outside our own defense area."""
    from utama_core.tactics.shadow_and_mark import _fallback_hold_target

    runner = runner_factory(exp_friendly=5, exp_enemy=1)
    game = runner.my.game

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
    assert fallback_0.distance_to(fallback_1) > 0.1


def test_press_and_contain_inapplicable_when_no_enemy_is_near_the_ball():
    """A pressing tactic should refuse to start when nothing dangerous is
    happening — pure-logic test, no rsim needed."""
    from utama_core.tactics.press_and_contain import _PRESS_RANGE

    class _FakeVec:
        def __init__(self, x, y):
            self.x, self.y = x, y

        def distance_to(self, other):
            return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

        def to_2d(self):
            return self

    class _FakeEnemy:
        def __init__(self, x, y):
            self.p = _FakeVec(x, y)

    class _FakeGame:
        def __init__(self, enemy_robots):
            self.ball = type("B", (), {"p": _FakeVec(0.0, 0.0)})()
            self.enemy_robots = enemy_robots

    far_game = _FakeGame(enemy_robots={1: _FakeEnemy(_PRESS_RANGE + 1.0, 0.0)})
    near_game = _FakeGame(enemy_robots={1: _FakeEnemy(_PRESS_RANGE - 0.5, 0.0)})
    empty_game = _FakeGame(enemy_robots={})

    tactic = PressAndContainTactic()
    assert tactic.applicable(far_game) is False
    assert tactic.applicable(near_game) is True
    assert tactic.applicable(empty_game) is False


def test_give_and_go_reassigned_carrier_resets_role_state(runner_factory):
    """If the kernel hands this tactic a robot set that no longer contains
    the previous carrier (e.g. after a scheduler reassignment), roles must
    reset rather than keep pointing at a robot this tactic no longer owns."""
    from utama_core.tactics.give_and_go import GiveAndGoMem

    runner = runner_factory(exp_friendly=5, exp_enemy=2)
    game = runner.my.game
    tactic = GiveAndGoTactic()
    mem = GiveAndGoMem(carrier_id=99, receiver_id=98, hop_count=3)
    commands, mem = tactic.tick(game, _ctx(runner), (1, 2, 3), mem)
    assert mem.carrier_id == 1
    assert mem.receiver_id is None
    assert mem.hop_count == 0
    assert set(commands.keys()) == {1, 2, 3}
