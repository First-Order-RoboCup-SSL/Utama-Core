"""`KernelStrategy` — adapts a `kernel.Strategy` to the `AbstractStrategy`/`StrategyRunner` contract.

`StrategyRunner` only knows how to drive an `AbstractStrategy`: it calls
`setup_strategy_blackboard`, `load_game`, `load_robot_controller`,
`load_motion_controller`, and once per tick, `step()`. Rather than build a
second runner for the Tactic model, this class satisfies that same contract
so `StrategyRunner` can drive it unmodified — `step()` is overridden to tick
the kernel `Strategy` (and the pinned goalkeeper) directly, bypassing
py_trees/blackboard entirely for command computation. `create_behaviour_tree`
still needs to return *something* because `AbstractStrategy.__init__` always
builds one (used only for the unused referee-override subtree machinery);
an empty Selector is enough since `step()` never ticks it.

Robot 0 is always the goalkeeper, ticked directly and never handed to the
kernel `Strategy` — see `tactics/goalkeeper.py` and the "Tactics as
Processes" design note, section 1.
"""

from __future__ import annotations

from typing import Optional

import py_trees

from utama_core.config.enums import Role
from utama_core.entities.data.command import RobotCommand
from utama_core.entities.data.object import TeamType
from utama_core.entities.game import Game
from utama_core.entities.game.field import FieldBounds
from utama_core.kernel.context import KernelContext
from utama_core.kernel.referee_override import is_override_command
from utama_core.kernel.referee_reset import is_paused
from utama_core.kernel.strategy import Strategy as KernelSchedulerStrategy
from utama_core.kernel.tactic import RobotId
from utama_core.motion_planning.src.common.motion_controller import MotionController
from utama_core.strategy.common.abstract_strategy import (
    AbstractStrategy,
    SpaceRequirements,
)
from utama_core.tactics.defense import DefenseTactic
from utama_core.tactics.give_and_go import GiveAndGoTactic
from utama_core.tactics.goalkeeper import GoalkeeperTactic
from utama_core.tactics.lead_and_support import LeadAndSupportTactic
from utama_core.tactics.press_and_contain import PressAndContainTactic
from utama_core.tactics.shadow_and_mark import ShadowAndMarkTactic
from utama_core.tactics.two_robot_attack import TwoRobotAttackTactic


class KernelStrategy(AbstractStrategy):
    """`AbstractStrategy` subclass that delegates ticking to a `kernel.Strategy`.

    Args:
        build_kernel_strategy: called once, after `load_game`, as
            `build_kernel_strategy(game, motion_controller) -> kernel.Strategy`.
            Deferred to a factory (rather than passed pre-built) because the
            motion controller is only available on the blackboard once
            `StrategyRunner` calls `load_motion_controller` — which happens
            before `load_game` in `StrategyRunner.__init__`, so it's safe to
            read here, but not at `KernelStrategy.__init__`.
        goalkeeper_id: robot ID pinned to the goalkeeper tactic, outside the
            kernel scheduler. Defaults to 0 per SSL/team convention.
        exp_ball: forwarded to `AbstractStrategy`.
    """

    def __init__(
        self,
        build_kernel_strategy,
        goalkeeper_id: int = 0,
        exp_ball: bool = True,
    ):
        self.exp_ball = exp_ball
        self._build_kernel_strategy = build_kernel_strategy
        self._kernel_strategy: Optional[KernelSchedulerStrategy] = None
        self._goalkeeper = GoalkeeperTactic(robot_id=goalkeeper_id)
        self._goalkeeper_id = goalkeeper_id
        self._goalkeeper_mem = self._goalkeeper.initial_mem()
        super().__init__()

    def create_behaviour_tree(self) -> py_trees.behaviour.Behaviour:
        # step() is fully overridden and never ticks this; AbstractStrategy's
        # __init__ requires a tree to exist regardless.
        return py_trees.composites.Selector(name="KernelStrategyUnusedRoot", memory=False)

    def assert_exp_robots(self, n_runtime_friendly: int, n_runtime_enemy: int) -> bool:
        return True

    def assert_exp_goals(self, includes_my_goal_line: bool, includes_opp_goal_line: bool) -> bool:
        return True

    def get_min_bounding_req(self) -> Optional[FieldBounds | SpaceRequirements]:
        return None

    def load_game(self, game: Game):
        super().load_game(game)
        if self._kernel_strategy is None:
            motion_controller = self.blackboard.motion_controller
            self._kernel_strategy = self._build_kernel_strategy(game, motion_controller)

    def step(self):
        game = self.blackboard.game

        outfield_commands = self._kernel_strategy.tick(game)

        cmd_map: dict[int, RobotCommand] = {}
        cmd_map.update(outfield_commands)

        # During a referee-restart override, `outfield_commands` already covers
        # every friendly robot including the goalkeeper (the BT-path Step
        # classes this delegates to compute for all of `game.friendly_robots`,
        # not just the outfield pool) — ticking GoalkeeperTactic on top would
        # overwrite that with normal ball-tracking logic mid-restart.
        #
        # During HALT/STOP, the goalkeeper must stop issuing motion commands
        # for the same reason `Strategy.tick()` freezes the outfield pool via
        # `is_paused` — skipping the tick here falls through to
        # `execute_default_action` below, which returns `empty_command(False)`,
        # the correct "stop" command.
        referee = getattr(game, "referee", None)
        current_command = getattr(referee, "referee_command", None) if referee is not None else None
        if (
            not is_override_command(current_command)
            and not is_paused(current_command)
            and self._goalkeeper_id not in cmd_map
        ):
            gk_commands, self._goalkeeper_mem = self._goalkeeper.tick(
                game, self._kernel_strategy._ctx, (self._goalkeeper_id,), self._goalkeeper_mem
            )
            cmd_map.update(gk_commands)

        for robot_id in game.friendly_robots:
            if robot_id in cmd_map:
                self.robot_controller.add_robot_commands(cmd_map[robot_id], robot_id)
            else:
                role = Role.GOALKEEPER if robot_id == self._goalkeeper_id else Role.UNASSIGNED
                self.robot_controller.add_robot_commands(self.execute_default_action(game, role, robot_id), robot_id)

        self.robot_controller.send_robot_commands()

    def debug_status(self) -> dict[int, list[str]]:
        """Per-robot `["<tactic>", "committed"?]` for GUI display.

        The Tactic model's analog of `StrategyRunner._push_bt_nodes_to_referee`'s
        BT-node breadcrumb — there is no behaviour tree to walk here, so this
        reports which tactic slot each robot currently belongs to and whether
        that slot is `is_committed()` (the actual "why won't this reassign"
        signal in this model), rather than any tactic-internal phase detail.
        """
        game = self.blackboard.game
        status: dict[int, list[str]] = {self._goalkeeper_id: ["goalkeeper"]}
        for tactic_id, info in self._kernel_strategy.slot_status(game).items():
            label = tactic_id if not info["committed"] else f"{tactic_id} (committed)"
            for robot_id in info["robots"]:
                status[robot_id] = [label]
        return status


def build_default_kernel_strategy(outfield_robot_ids: tuple[int, ...]):
    """Minimal, single-tactic-pool `Strategy` factory: everyone attacks.

    Only one real multi-robot tactic exists so far (`two_robot_attack`), so
    the picker has nothing to choose between yet — per the design doc's
    "splitting policy" deferral, this deliberately does not invent an
    allocation policy ahead of a second concrete tactic that would need one.
    Callers with more than one outfield tactic should build their own
    `kernel.Strategy` with a real `Picker` instead of using this helper.

    Returns a `build_kernel_strategy(game, motion_controller)`
    callable suitable for `KernelStrategy`'s constructor argument of the
    same name.
    """

    def _build(game: Game, motion_controller: MotionController) -> KernelSchedulerStrategy:
        ctx = KernelContext(motion_controller=motion_controller)
        return KernelSchedulerStrategy(
            tactics={"two_robot_attack": TwoRobotAttackTactic()},
            partitioner=KernelSchedulerStrategy.single_tactic_picker(lambda game, active: "two_robot_attack"),
            outfield_robot_ids=outfield_robot_ids,
            ctx=ctx,
        )

    return _build


def _possession_split_picker(
    game: Game,
    free_robots: frozenset[RobotId],
    prev_partition: Optional[dict[str, frozenset[RobotId]]],
    applicable_tactic_ids: frozenset[str],
) -> dict[str, frozenset[RobotId]]:
    """Whichever side is closer to the ball decides posture; the split is fixed once decided.

    Ignores `applicable_tactic_ids`: neither `LeadAndSupportTactic` nor
    `ShadowAndMarkTactic` overrides `applicable()` (both default to always
    applicable), so there is nothing for this picker to react to. Accepted
    only because every `Partitioner` must match the shared signature — see
    `_press_and_pass_split_picker` for a picker that actually uses it.

    Deliberately the simplest rule that gives the split-shape scheduler
    something real to react to, not a scored/tunable allocator (see the
    design doc's stance against bid/fitness-scoring machinery) — one signal
    (which team is closer to the ball), two fixed splits. Ties and an
    unreadable proximity lookup default to the more conservative
    (defense-heavy) split.

    Only emits a key for a tactic it is actually assigning free robots to —
    never a zero-robot entry, since `Strategy` treats every key in a
    `Partitioner`'s return value as "I am claiming this tactic id right now,"
    and a present-but-empty entry for a tactic committed and pinned
    elsewhere would collide with that pin.

    Never re-proposes a tactic id that is currently pinned by a commitment.
    With exactly two ids and a binary split, this picker cannot tell from
    `free_robots` alone whether "attack" is absent because it's pinned
    elsewhere (holding robots outside the free pool) or because this picker
    itself chose to leave it empty last tick — but `prev_partition` (the full
    previous partition `Strategy` always passes in, including pinned
    entries) does distinguish the two: if "attack" held a non-empty set last
    tick and none of the free pool overlaps that set, "attack" must still be
    pinned with it, and this picker must leave "attack" out entirely rather
    than propose a *second*, conflicting claim on the same tactic id.
    """
    ordered = sorted(free_robots)
    if not ordered:
        return {}

    prev_partition = prev_partition or {}
    pinned_ids = {tid for tid, robots in prev_partition.items() if robots and not (robots & free_robots)}

    _friendly_closest, friendly_dist = game.proximity_lookup.closest_to_ball(team_type_filter=TeamType.FRIENDLY)
    _enemy_closest, enemy_dist = game.proximity_lookup.closest_to_ball(team_type_filter=TeamType.ENEMY)
    friendly_has_ball_edge = friendly_dist < enemy_dist

    if "attack" in pinned_ids:
        return {"defense": frozenset(ordered)}
    if "defense" in pinned_ids:
        return {"attack": frozenset(ordered)}

    attack_count = (len(ordered) + 1) // 2 + 1 if friendly_has_ball_edge else len(ordered) // 2 - 1
    attack_count = max(0, min(len(ordered), attack_count))

    partition = {}
    if attack_count > 0:
        partition["attack"] = frozenset(ordered[:attack_count])
    if attack_count < len(ordered):
        partition["defense"] = frozenset(ordered[attack_count:])
    return partition


def build_split_shape_kernel_strategy(outfield_robot_ids: tuple[int, ...]):
    """5-robot (or fewer) split-shape `Strategy` factory.

    Wires `LeadAndSupportTactic` ("attack") and `ShadowAndMarkTactic`
    ("defense") as two concurrently active slots, split by
    `_possession_split_picker`. This is the concrete forcing case the design
    doc's §7 deferral was waiting on — see §11 for the full rationale.

    Returns a `build_kernel_strategy(game, motion_controller)`
    callable suitable for `KernelStrategy`'s constructor argument of the
    same name.
    """

    def _build(game: Game, motion_controller: MotionController) -> KernelSchedulerStrategy:
        ctx = KernelContext(motion_controller=motion_controller)
        return KernelSchedulerStrategy(
            tactics={"attack": LeadAndSupportTactic(), "defense": ShadowAndMarkTactic()},
            partitioner=_possession_split_picker,
            outfield_robot_ids=outfield_robot_ids,
            ctx=ctx,
        )

    return _build


def _press_and_pass_split_picker(
    game: Game,
    free_robots: frozenset[RobotId],
    prev_partition: Optional[dict[str, frozenset[RobotId]]],
    applicable_tactic_ids: frozenset[str],
) -> dict[str, frozenset[RobotId]]:
    """Same possession-edge split as `_possession_split_picker`, but must also
    respect `PressAndContainTactic.applicable()` — see design doc §15.

    `Strategy` passes `applicable_tactic_ids` precisely so a picker doesn't
    have to reconstruct a tactic just to ask it a question the kernel
    already knows the answer to. When pressing isn't applicable (no enemy
    near the ball), there is nothing to defend against, so every free robot
    goes to "attack" instead of leaving "defense" empty for no
    game-theoretic reason — a sensible default, not just a way to dodge the
    kernel's applicability check.
    """
    ordered = sorted(free_robots)
    if not ordered:
        return {}

    prev_partition = prev_partition or {}
    pinned_ids = {tid for tid, robots in prev_partition.items() if robots and not (robots & free_robots)}

    pressing_applicable = "defense" not in pinned_ids and "defense" in applicable_tactic_ids
    if not pressing_applicable:
        return {} if "attack" in pinned_ids else {"attack": frozenset(ordered)}

    _friendly_closest, friendly_dist = game.proximity_lookup.closest_to_ball(team_type_filter=TeamType.FRIENDLY)
    _enemy_closest, enemy_dist = game.proximity_lookup.closest_to_ball(team_type_filter=TeamType.ENEMY)
    friendly_has_ball_edge = friendly_dist < enemy_dist

    if "attack" in pinned_ids:
        return {"defense": frozenset(ordered)}

    attack_count = (len(ordered) + 1) // 2 + 1 if friendly_has_ball_edge else len(ordered) // 2 - 1
    attack_count = max(0, min(len(ordered), attack_count))

    partition = {}
    if attack_count > 0:
        partition["attack"] = frozenset(ordered[:attack_count])
    if attack_count < len(ordered):
        partition["defense"] = frozenset(ordered[attack_count:])
    return partition


def build_press_and_pass_kernel_strategy(outfield_robot_ids: tuple[int, ...]):
    """5-robot (or fewer) `Strategy` factory exercising the newer tactics.

    Wires `GiveAndGoTactic` ("attack") and `PressAndContainTactic`
    ("defense") as two concurrently active slots, split by
    `_press_and_pass_split_picker`. Sibling to
    `build_split_shape_kernel_strategy`, same shape, different tactic pair —
    lets the give-and-go/press-and-contain tactics be driven end to end via
    `StrategyRunner` instead of only unit-level `tick()` calls.

    Returns a `build_kernel_strategy(game, motion_controller)`
    callable suitable for `KernelStrategy`'s constructor argument of the
    same name.
    """

    def _build(game: Game, motion_controller: MotionController) -> KernelSchedulerStrategy:
        ctx = KernelContext(motion_controller=motion_controller)
        return KernelSchedulerStrategy(
            tactics={"attack": GiveAndGoTactic(), "defense": PressAndContainTactic()},
            partitioner=_press_and_pass_split_picker,
            outfield_robot_ids=outfield_robot_ids,
            ctx=ctx,
        )

    return _build


def _fixed_ratio_picker(attack_id: str, defense_id: str, attack_fraction: float, min_attack: int = 0):
    """Builds a `Partitioner` that ignores game state entirely and splits the
    free pool by a constant fraction — the simplest possible allocation rule,
    useful as a deliberately non-reactive baseline/contrast to the
    possession-edge pickers above. Still respects commitment pinning and
    `applicable_tactic_ids` exactly like every other `Partitioner`; "fixed"
    only describes the *ratio* decision, not an exemption from the kernel's
    invariants.

    `min_attack`: a floor on the attack slot's robot count whenever attack is
    getting any robots at all. Exists because not every Tactic is
    robot-count-agnostic the way `GiveAndGoTactic`/`LeadAndSupportTactic`
    are — `TwoRobotAttackTactic.tick()` unconditionally reads
    `robot_ids[1]`, so a rounded-down fraction that hands it a single robot
    crashes with an `IndexError` rather than degrading gracefully. This
    picker has no way to know that from the Tactic itself (no
    `min_robots`/`max_robots` declaration exists yet — see design doc §15's
    "explicitly deferred" list), so the caller states the floor explicitly.
    """

    def _partitioner(
        game: Game,
        free_robots: frozenset[RobotId],
        prev_partition: Optional[dict[str, frozenset[RobotId]]],
        applicable_tactic_ids: frozenset[str],
    ) -> dict[str, frozenset[RobotId]]:
        del game
        ordered = sorted(free_robots)
        if not ordered:
            return {}

        prev_partition = prev_partition or {}
        pinned_ids = {tid for tid, robots in prev_partition.items() if robots and not (robots & free_robots)}

        attack_ok = attack_id not in pinned_ids and attack_id in applicable_tactic_ids
        defense_ok = defense_id not in pinned_ids and defense_id in applicable_tactic_ids

        if attack_id in pinned_ids:
            return {defense_id: frozenset(ordered)} if defense_ok else {}
        if defense_id in pinned_ids:
            return {attack_id: frozenset(ordered)} if attack_ok else {}
        if not defense_ok:
            return {attack_id: frozenset(ordered)} if attack_ok else {}
        if not attack_ok:
            return {defense_id: frozenset(ordered)}

        attack_count = max(0, min(len(ordered), round(len(ordered) * attack_fraction)))
        if 0 < attack_count < min_attack:
            attack_count = min(len(ordered), min_attack)
        partition = {}
        if attack_count > 0:
            partition[attack_id] = frozenset(ordered[:attack_count])
        if attack_count < len(ordered):
            partition[defense_id] = frozenset(ordered[attack_count:])
        return partition

    return _partitioner


def build_high_press_kernel_strategy(outfield_robot_ids: tuple[int, ...]):
    """5-robot (or fewer) `Strategy` factory: an aggressive, *non-reactive*
    posture — most of the pool commits forward regardless of who has the
    ball, unlike `build_press_and_pass_kernel_strategy`'s possession-edge
    split. Wires the same `GiveAndGoTactic`/`PressAndContainTactic` pair as
    that config, but the two are not interchangeable: the point here is to
    demonstrate that the same Tactic set can be driven by a mechanically
    different `Partitioner` (fixed ratio, via `_fixed_ratio_picker`) — the
    scheduling *policy* is what varies between example strategies, not just
    the Tactic roster.

    Returns a `build_kernel_strategy(game, motion_controller)`
    callable suitable for `KernelStrategy`'s constructor argument of the
    same name.
    """

    def _build(game: Game, motion_controller: MotionController) -> KernelSchedulerStrategy:
        ctx = KernelContext(motion_controller=motion_controller)
        return KernelSchedulerStrategy(
            tactics={"attack": GiveAndGoTactic(), "defense": PressAndContainTactic()},
            partitioner=_fixed_ratio_picker("attack", "defense", attack_fraction=0.8),
            outfield_robot_ids=outfield_robot_ids,
            ctx=ctx,
        )

    return _build


def build_low_block_kernel_strategy(outfield_robot_ids: tuple[int, ...]):
    """5-robot (or fewer) `Strategy` factory: a conservative, defense-heavy
    counterpart to `build_high_press_kernel_strategy` — most of the pool
    stays back regardless of possession (floored at `min_attack=2`, since
    `TwoRobotAttackTactic` hard-requires at least 2 robots). Wires
    `TwoRobotAttackTactic` ("attack") and `DefenseTactic` ("defense"), the
    original two ported Tactics, still useful as the minimal-risk baseline
    they were designed to be (§1/§2's original pairing) rather than the
    newer, more elaborate Tactics used elsewhere in this file.

    Returns a `build_kernel_strategy(game, motion_controller)`
    callable suitable for `KernelStrategy`'s constructor argument of the
    same name.
    """

    def _build(game: Game, motion_controller: MotionController) -> KernelSchedulerStrategy:
        ctx = KernelContext(motion_controller=motion_controller)
        return KernelSchedulerStrategy(
            tactics={"attack": TwoRobotAttackTactic(), "defense": DefenseTactic()},
            partitioner=_fixed_ratio_picker("attack", "defense", attack_fraction=0.2, min_attack=2),
            outfield_robot_ids=outfield_robot_ids,
            ctx=ctx,
        )

    return _build


def _three_way_picker(
    game: Game,
    free_robots: frozenset[RobotId],
    prev_partition: Optional[dict[str, frozenset[RobotId]]],
    applicable_tactic_ids: frozenset[str],
) -> dict[str, frozenset[RobotId]]:
    """Splits the free pool three ways — first exercise of `Strategy` running
    more than two concurrent slots (nothing in `Strategy`/`_validate_partition`
    is hardcoded to two, but no config before this one had actually tried
    three). One presser, one shadow-and-mark defensive pair when there are
    enough robots to spare, everyone else attacks via give-and-go. Falls back
    to redistributing a slot's share to "attack" whenever that slot's Tactic
    is currently inapplicable or pinned elsewhere, same defensive pattern as
    `_press_and_pass_split_picker`.
    """
    ordered = sorted(free_robots)
    if not ordered:
        return {}

    prev_partition = prev_partition or {}
    pinned_ids = {tid for tid, robots in prev_partition.items() if robots and not (robots & free_robots)}

    press_ok = "press" not in pinned_ids and "press" in applicable_tactic_ids
    mark_ok = "mark" not in pinned_ids and "mark" in applicable_tactic_ids
    attack_ok = "attack" not in pinned_ids and "attack" in applicable_tactic_ids

    remaining = list(ordered)
    partition: dict[str, frozenset[RobotId]] = {}

    if press_ok and remaining:
        partition["press"] = frozenset([remaining.pop(0)])
    if mark_ok and len(remaining) >= 2:
        partition["mark"] = frozenset(remaining[:2])
        remaining = remaining[2:]
    if attack_ok:
        if remaining:
            partition["attack"] = frozenset(remaining)
    elif remaining:
        # "attack" unavailable too — nothing left to hand the rest to; leave
        # them off the partition only if every other slot already claimed
        # everyone, otherwise this would violate the exhaustive-cover
        # invariant, so fall back to whichever defensive slot is still open.
        if mark_ok:
            partition["mark"] = frozenset(set(partition.get("mark", frozenset())) | set(remaining))
        elif press_ok:
            partition["press"] = frozenset(set(partition.get("press", frozenset())) | set(remaining))

    return partition


def build_three_slot_kernel_strategy(outfield_robot_ids: tuple[int, ...]):
    """5-robot (or fewer) `Strategy` factory running three concurrent Tactic
    slots at once: `PressAndContainTactic` ("press"), `ShadowAndMarkTactic`
    ("mark"), and `GiveAndGoTactic` ("attack"). Distinct in kind from every
    other factory in this file, which all run exactly two slots — this is
    the first config to actually exercise `Strategy` with N>2, which the
    kernel has always structurally supported (`_validate_partition` iterates
    `partition.items()` generically) but nothing had tested until now.

    Returns a `build_kernel_strategy(game, motion_controller)`
    callable suitable for `KernelStrategy`'s constructor argument of the
    same name.
    """

    def _build(game: Game, motion_controller: MotionController) -> KernelSchedulerStrategy:
        ctx = KernelContext(motion_controller=motion_controller)
        return KernelSchedulerStrategy(
            tactics={
                "press": PressAndContainTactic(),
                "mark": ShadowAndMarkTactic(),
                "attack": GiveAndGoTactic(),
            },
            partitioner=_three_way_picker,
            outfield_robot_ids=outfield_robot_ids,
            ctx=ctx,
        )

    return _build


def build_give_and_go_solo_kernel_strategy(outfield_robot_ids: tuple[int, ...]):
    """Single-tactic `Strategy` factory: the entire outfield pool always runs
    `GiveAndGoTactic`, no defense slot at all. Mirrors
    `build_default_kernel_strategy`'s shape (single Tactic, no real
    allocation decision, via `single_tactic_picker`) but for the newer
    passing tactic — useful as an isolated benchmark/eval config when
    testing `GiveAndGoTactic` in isolation matters more than realistic match
    posture (e.g. tuning `_MAX_HOPS_PER_POSSESSION` or the support-scoring
    weights without a defensive Tactic's behaviour as a confound).

    Returns a `build_kernel_strategy(game, motion_controller)`
    callable suitable for `KernelStrategy`'s constructor argument of the
    same name.
    """

    def _build(game: Game, motion_controller: MotionController) -> KernelSchedulerStrategy:
        ctx = KernelContext(motion_controller=motion_controller)
        return KernelSchedulerStrategy(
            tactics={"attack": GiveAndGoTactic()},
            partitioner=KernelSchedulerStrategy.single_tactic_picker(lambda game, active: "attack"),
            outfield_robot_ids=outfield_robot_ids,
            ctx=ctx,
        )

    return _build
