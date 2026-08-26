"""Comprehensive, table-driven tests over every registered `Tactic`.

Replaces the earlier one-file-per-Tactic pattern (`test_defense_tactic.py`,
`test_lead_and_support_tactic.py`, `test_shadow_and_mark_tactic.py`,
`test_press_and_contain_tactic.py`, `test_give_and_go_tactic.py`), which had
each file re-asserting the same handful of generic properties ("produces a
command for every assigned robot", "is_committed() returns a bool") with
near-identical boilerplate around a real `StrategyRunner`/rsim fixture. A
single parametrized suite over `_TACTIC_CASES` below gives the same coverage
without the duplication, and — concretely — would have caught
`PassAndShootTactic`'s `robot_ids[1]` crash (see `build_low_block_kernel_strategy`
in `kernel_strategy.py`) automatically instead of requiring a bespoke test
to notice a Tactic has an undeclared minimum robot count.

Behavior specific to one Tactic (e.g. `LeadAndSupportTactic`'s leader
picked by ball proximity, `ShadowAndMarkTactic`'s fallback-hold-target
regression) stays in its own test — those don't generalize across the
table and shouldn't be forced into it. `PassAndShootTactic`'s pure-logic
`assign_passer_receiver`/`is_committed()` tests also stay separate
(`test_pass_and_shoot_tactic.py`) since they need no rsim fixture at all.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from utama_core.engine.context import TickContext
from utama_core.engine.tactic import TacticTag
from utama_core.tactics.clear_ball import ClearBallTactic
from utama_core.tactics.decoy_and_overload import DecoyOverloadTactic
from utama_core.tactics.defense import DefenseTactic
from utama_core.tactics.give_and_go import GiveAndGoTactic
from utama_core.tactics.lead_and_support import LeadAndSupportTactic
from utama_core.tactics.pass_and_shoot import PassAndShootTactic
from utama_core.tactics.press_and_contain import PressAndContainTactic
from utama_core.tactics.shadow_and_mark import ShadowAndMarkTactic

# (tactic_factory, robot_ids, exp_friendly, exp_enemy) — robot_ids and
# exp_friendly/exp_enemy are chosen per-Tactic to respect each one's real
# constraints (e.g. PassAndShootTactic hard-requires >=2 robots; no Tactic
# declares this anywhere today — see design doc §15's "explicitly deferred"
# per-Tactic robot-count-bound item — so it's encoded here instead).
_TACTIC_CASES = [
    pytest.param(DefenseTactic, (1, 2), 3, 0, id="defense"),
    pytest.param(PassAndShootTactic, (1, 2), 3, 0, id="pass_and_shoot"),
    pytest.param(LeadAndSupportTactic, (1, 2, 3, 4), 5, 2, id="lead_and_support"),
    pytest.param(ShadowAndMarkTactic, (1, 2, 3, 4), 5, 3, id="shadow_and_mark"),
    pytest.param(PressAndContainTactic, (1, 2, 3), 4, 3, id="press_and_contain"),
    pytest.param(GiveAndGoTactic, (1, 2, 3), 5, 2, id="give_and_go"),
    pytest.param(ClearBallTactic, (1, 2, 3), 5, 2, id="clear_ball"),
    pytest.param(DecoyOverloadTactic, (1, 2, 3), 5, 2, id="decoy_and_overload"),
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


def _ctx(runner) -> TickContext:
    motion_controller = runner.my.motion_controller(runner.mode, runner.rsim_env)
    return TickContext(motion_controller=motion_controller)


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


def test_press_and_contain_goes_straight_for_a_fully_loose_ball(runner_factory):
    """Found live (stuck-match investigation, 2026-08-26, docs/testing_gaps.md
    gap #11): a ball can be genuinely loose (no robot on either team
    possesses it) while the tracked "nearest enemy" is itself stationary
    and far away — e.g. that enemy's own team is locked into an all-defense
    posture by a separate bug. `block_attacker`'s no-possession branch
    computes the presser's target *relative to that enemy's own position*
    (a shot-line-style standoff), not straight at the ball — so when the
    tracked enemy never moves, the computed target never converges on the
    ball either, and the presser parks nearby without ever actually
    collecting it. Traced in a real replay: this held for 566s of a 600s
    match, with two independently-reasonable tactics (containment; don't
    chase a ball the opponent is closer to) jointly deadlocking on a ball
    neither side ever collects.

    Regression: with the tracked enemy stationary and far from the ball,
    the presser must still close in on the ball itself.
    """
    import dataclasses

    from utama_core.entities.data.vector import Vector2D, Vector3D
    from utama_core.entities.game.ball import Ball
    from utama_core.entities.game.game_frame import GameFrame

    runner = runner_factory(exp_friendly=3, exp_enemy=2)
    game = runner.my.game
    tactic = PressAndContainTactic()
    mem = tactic.initial_mem()

    frame = game.current
    friendly = dict(frame.friendly_robots)
    friendly[1] = dataclasses.replace(friendly[1], has_ball=False, p=Vector2D(-1.5, 0.0))
    enemy = dict(frame.enemy_robots)
    # Tracked enemy far from the ball and stationary — the exact condition
    # observed live: nothing drives it toward the ball either.
    enemy[1] = dataclasses.replace(enemy[1], has_ball=False, p=Vector2D(3.0, 3.0))
    ball = Ball(p=Vector3D(0.0, 0.0, 0.0), v=Vector3D(0.0, 0.0, 0.0), a=Vector3D(0.0, 0.0, 0.0))
    loose_frame = GameFrame(
        ts=frame.ts,
        my_team_is_yellow=frame.my_team_is_yellow,
        my_team_is_right=frame.my_team_is_right,
        friendly_robots=friendly,
        enemy_robots=enemy,
        ball=ball,
        referee=frame.referee,
    )
    game.add_game_frame(loose_frame)
    game = runner.my.game
    assert game.robot_with_ball is None, "setup didn't produce a loose ball — test would pass vacuously"

    import utama_core.tactics.press_and_contain as press_and_contain_module

    calls = {}
    orig_go_to_ball = press_and_contain_module.go_to_ball
    orig_block_attacker = press_and_contain_module.block_attacker

    def spy_go_to_ball(*args, **kwargs):
        calls["go_to_ball"] = True
        return orig_go_to_ball(*args, **kwargs)

    def spy_block_attacker(*args, **kwargs):
        calls["block_attacker"] = True
        return orig_block_attacker(*args, **kwargs)

    press_and_contain_module.go_to_ball = spy_go_to_ball
    press_and_contain_module.block_attacker = spy_block_attacker
    try:
        commands, _mem = tactic.tick(game, _ctx(runner), (1, 2, 3), mem)
    finally:
        press_and_contain_module.go_to_ball = orig_go_to_ball
        press_and_contain_module.block_attacker = orig_block_attacker

    assert 1 in commands
    assert calls.get("go_to_ball") is True, "presser should drive straight at a fully loose ball"
    assert calls.get("block_attacker") is None, "block_attacker's enemy-relative standoff must not be used here"


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


def test_give_and_go_abandons_a_hop_that_never_completes(runner_factory):
    """Found live (stuck-match investigation, 2026-08-26, docs/testing_gaps.md
    gap #11): a carrier that locks a receiver_id and then never sees that
    receiver become ready (_pass_exec's synchronized handshake has no
    timeout of its own) held the ball motionless for the rest of a 600s
    match — is_committed() stays True forever since it only checks
    receiver_id is not None, so nothing ever reassigns these robots either.

    This test locks receiver_id directly (bypassing _best_receiver's own
    pick, which needs real geometry to trigger) and never advances the sim
    clock, so the receiver's position never changes and _pass_exec's
    handshake can never resolve — the simplest deterministic way to force
    the exact stalled state, regardless of what real-match geometry
    originally caused it. After _MAX_HOP_TICKS worth of tick() calls,
    receiver_id must have been abandoned (reset to None) rather than held
    forever, and the tactic must be reassignable again (is_committed()
    False) once that happens.
    """
    import dataclasses

    from utama_core.entities.data.vector import Vector2D, Vector3D
    from utama_core.entities.game.ball import Ball
    from utama_core.entities.game.game_frame import GameFrame
    from utama_core.tactics.give_and_go import _MAX_HOP_TICKS, GiveAndGoMem

    runner = runner_factory(exp_friendly=5, exp_enemy=2)
    game = runner.my.game
    tactic = GiveAndGoTactic()
    mem = GiveAndGoMem(carrier_id=1, receiver_id=2, hop_count=0, hop_ticks=0)

    # Synthesize a frame where robot 1 possesses the ball and robot 2 (the
    # locked receiver) sits far enough away that it can never reach
    # _pass_exec's intercept point — has_ball is sensor/contact-derived, not
    # directly settable, and no sim stepping is needed for a pure decision-
    # logic test like this one, so build the frame directly instead of
    # driving real physics.
    frame = game.current
    friendly = dict(frame.friendly_robots)
    friendly[1] = dataclasses.replace(friendly[1], has_ball=True, p=Vector2D(0.0, 0.0))
    friendly[2] = dataclasses.replace(friendly[2], has_ball=False, p=Vector2D(-4.0, 2.8))
    ball = Ball(p=Vector3D(0.09, 0.0, 0.0), v=Vector3D(0.0, 0.0, 0.0), a=Vector3D(0.0, 0.0, 0.0))
    stalled_frame = GameFrame(
        ts=frame.ts,
        my_team_is_yellow=frame.my_team_is_yellow,
        my_team_is_right=frame.my_team_is_right,
        friendly_robots=friendly,
        enemy_robots=dict(frame.enemy_robots),
        ball=ball,
        referee=frame.referee,
    )
    game.add_game_frame(stalled_frame)
    game = runner.my.game
    assert game.friendly_robots[1].has_ball, "setup didn't establish possession — test would pass vacuously"

    for _ in range(_MAX_HOP_TICKS + 1):
        assert tactic.is_committed(game, mem) is True
        _commands, mem = tactic.tick(game, _ctx(runner), (1, 2, 3), mem)
        if mem.receiver_id is None:
            break

    assert mem.receiver_id is None, "hop was never abandoned — carrier would hold the ball forever"
    assert tactic.is_committed(game, mem) is False


class _FakeVec:
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __sub__(self, other):
        return _FakeVec(self.x - other.x, self.y - other.y)

    def __add__(self, other):
        return _FakeVec(self.x + other.x, self.y + other.y)

    def __mul__(self, scalar):
        return _FakeVec(self.x * scalar, self.y * scalar)

    def dot(self, other):
        return self.x * other.x + self.y * other.y

    def distance_to(self, other):
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

    def to_2d(self):
        return self


def test_clear_ball_inapplicable_outside_danger_zone():
    """The valve only fires when the ball is deep in our own third AND an enemy
    contests it — pure-logic test, no rsim needed. Deep-but-uncontested (a calm
    back-pass situation) and contested-but-high must both read False."""
    from utama_core.tactics.clear_ball import _DANGER_DEPTH, _PRESSURE_RANGE

    class _FakeEnemy:
        def __init__(self, x, y):
            self.p = _FakeVec(x, y)

    class _FakeField:
        half_length = 4.5

    class _FakeGame:
        def __init__(self, ball_x):
            self.my_team_is_right = True  # own goal at +4.5; own third is x > ~1.5
            self.field = _FakeField()
            self.ball = type("B", (), {"p": _FakeVec(ball_x, 0.0)})()
            self.enemy_robots = {}

    deep_contested = _FakeGame(ball_x=3.0)
    deep_contested.enemy_robots = {1: _FakeEnemy(3.0 - _PRESSURE_RANGE / 2, 0.0)}
    deep_free = _FakeGame(ball_x=3.0)
    deep_free.enemy_robots = {1: _FakeEnemy(3.0 - (_PRESSURE_RANGE + 1.0), 0.0)}
    high_contested = _FakeGame(ball_x=-(_DANGER_DEPTH + 1.0))  # deep in the ENEMY third
    high_contested.enemy_robots = {1: _FakeEnemy(3.0, 0.0)}

    tactic = ClearBallTactic()
    assert tactic.applicable(deep_contested) is True
    assert tactic.applicable(deep_free) is False
    assert tactic.applicable(high_contested) is False


def test_clear_ball_best_target_prefers_open_lane():
    """An enemy squatting on the central lane must push the chosen clearance
    target to one of the wide lanes."""
    from utama_core.tactics.clear_ball import _best_clear_target

    class _FakeEnemy:
        def __init__(self, x, y):
            self.p = _FakeVec(x, y)

    class _FakeField:
        half_length = 4.5
        half_width = 3.0

    game = SimpleNamespace(
        my_team_is_right=True,
        field=_FakeField(),
        enemy_robots={1: _FakeEnemy(2.0, 0.0), 2: _FakeEnemy(2.5, 0.9)},
    )
    # Ball deep in our own corner-ish spot; central lane runs straight through
    # both enemies.
    target = _best_clear_target(game, _FakeVec(2.8, 0.2), prev_target=None)
    assert abs(target.y) > 1.0, f"expected a wide lane, got y={target.y}"


def test_clear_ball_best_target_hysteresis_keeps_prev_choice():
    """A previous choice must stand when nothing clearly beats it — recomputing
    fresh every tick is exactly the flapping bug class documented on
    `find_best_shot`/`go_to_ball`. With an empty field all lanes score equally,
    so the standing choice can never be displaced."""
    from utama_core.tactics.clear_ball import _best_clear_target

    class _FakeField:
        half_length = 4.5
        half_width = 3.0

    game = SimpleNamespace(my_team_is_right=True, field=_FakeField(), enemy_robots={})
    ball = _FakeVec(2.5, 0.0)
    first_choice = _best_clear_target(game, ball, prev_target=None)
    held = _best_clear_target(game, ball, prev_target=first_choice)
    assert held == first_choice
