"""Kernel `Strategy` factories — `build_kernel_strategy(motion_controller) -> kernel.Strategy`
callables suitable for `AbstractStrategy`'s constructor argument of the same name.

See `utama_core.kernel.abstract_strategy.AbstractStrategy` for the class that
consumes these and drives them under `StrategyRunner`.
"""

from __future__ import annotations

from typing import Optional

from utama_core.entities.data.object import TeamType
from utama_core.entities.game import Game
from utama_core.kernel.context import KernelContext
from utama_core.kernel.strategy import Strategy as KernelSchedulerStrategy
from utama_core.kernel.tactic import RobotId
from utama_core.motion_planning.src.common.motion_controller import MotionController
from utama_core.tactics.block_shape import BlockShapeTactic
from utama_core.tactics.decoy_and_overload import DecoyOverloadTactic
from utama_core.tactics.defense import DefenseTactic
from utama_core.tactics.give_and_go import GiveAndGoTactic
from utama_core.tactics.lead_and_support import LeadAndSupportTactic
from utama_core.tactics.pass_and_shoot import PassAndShootTactic
from utama_core.tactics.press_and_contain import PressAndContainTactic
from utama_core.tactics.shadow_and_mark import ShadowAndMarkTactic
from utama_core.tactics.switch_of_play import SwitchOfPlayTactic


def build_default_kernel_strategy(outfield_robot_ids: tuple[int, ...]):
    """Minimal, single-tactic-pool `Strategy` factory: everyone attacks.

    Only one real multi-robot tactic exists so far (`pass_and_shoot`), so
    the picker has nothing to choose between yet — per the design doc's
    "splitting policy" deferral, this deliberately does not invent an
    allocation policy ahead of a second concrete tactic that would need one.
    Callers with more than one outfield tactic should build their own
    `kernel.Strategy` with a real `Picker` instead of using this helper.

    Returns a `build_kernel_strategy(motion_controller)`
    callable suitable for `AbstractStrategy`'s constructor argument of the
    same name.
    """

    def _build(motion_controller: MotionController) -> KernelSchedulerStrategy:
        ctx = KernelContext(motion_controller=motion_controller)
        return KernelSchedulerStrategy(
            tactics={"pass_and_shoot": PassAndShootTactic()},
            partitioner=KernelSchedulerStrategy.single_tactic_picker(lambda game, active: "pass_and_shoot"),
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

    Returns a `build_kernel_strategy(motion_controller)`
    callable suitable for `AbstractStrategy`'s constructor argument of the
    same name.
    """

    def _build(motion_controller: MotionController) -> KernelSchedulerStrategy:
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

    Returns a `build_kernel_strategy(motion_controller)`
    callable suitable for `AbstractStrategy`'s constructor argument of the
    same name.
    """

    def _build(motion_controller: MotionController) -> KernelSchedulerStrategy:
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
    are — `PassAndShootTactic.tick()` unconditionally reads
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

    Returns a `build_kernel_strategy(motion_controller)`
    callable suitable for `AbstractStrategy`'s constructor argument of the
    same name.
    """

    def _build(motion_controller: MotionController) -> KernelSchedulerStrategy:
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
    `PassAndShootTactic` hard-requires at least 2 robots). Wires
    `PassAndShootTactic` ("attack") and `DefenseTactic` ("defense"), the
    original two ported Tactics, still useful as the minimal-risk baseline
    they were designed to be (§1/§2's original pairing) rather than the
    newer, more elaborate Tactics used elsewhere in this file.

    Returns a `build_kernel_strategy(motion_controller)`
    callable suitable for `AbstractStrategy`'s constructor argument of the
    same name.
    """

    def _build(motion_controller: MotionController) -> KernelSchedulerStrategy:
        ctx = KernelContext(motion_controller=motion_controller)
        return KernelSchedulerStrategy(
            tactics={"attack": PassAndShootTactic(), "defense": DefenseTactic()},
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

    Returns a `build_kernel_strategy(motion_controller)`
    callable suitable for `AbstractStrategy`'s constructor argument of the
    same name.
    """

    def _build(motion_controller: MotionController) -> KernelSchedulerStrategy:
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


def build_decoy_and_overload_kernel_strategy(outfield_robot_ids: tuple[int, ...]):
    """5-robot (or fewer) `Strategy` factory exercising `DecoyOverloadTactic`.

    Wires `DecoyOverloadTactic` ("attack") and `ShadowAndMarkTactic`
    ("defense"), split by `_fixed_ratio_picker` with `min_attack=2` —
    `DecoyOverloadTactic` hard-requires at least 2 robots (a decoy and an
    overloader) the same way `PassAndShootTactic` does, so it needs the
    same floor `build_low_block_kernel_strategy` gives that tactic. Attack
    fraction left at a plain 0.5 split (possession-agnostic) since nothing
    about the lure/overload pattern is more or less urgent when the
    opponent has the ball, unlike the press/shadow pairings above.

    Returns a `build_kernel_strategy(motion_controller)`
    callable suitable for `AbstractStrategy`'s constructor argument of the
    same name.
    """

    def _build(motion_controller: MotionController) -> KernelSchedulerStrategy:
        ctx = KernelContext(motion_controller=motion_controller)
        return KernelSchedulerStrategy(
            tactics={"attack": DecoyOverloadTactic(), "defense": ShadowAndMarkTactic()},
            partitioner=_fixed_ratio_picker("attack", "defense", attack_fraction=0.5, min_attack=2),
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

    Returns a `build_kernel_strategy(motion_controller)`
    callable suitable for `AbstractStrategy`'s constructor argument of the
    same name.
    """

    def _build(motion_controller: MotionController) -> KernelSchedulerStrategy:
        ctx = KernelContext(motion_controller=motion_controller)
        return KernelSchedulerStrategy(
            tactics={"attack": GiveAndGoTactic()},
            partitioner=KernelSchedulerStrategy.single_tactic_picker(lambda game, active: "attack"),
            outfield_robot_ids=outfield_robot_ids,
            ctx=ctx,
        )

    return _build


def build_switch_of_play_kernel_strategy(outfield_robot_ids: tuple[int, ...]):
    """5-robot (or fewer) `Strategy` factory exercising `SwitchOfPlayTactic`.

    Wires `SwitchOfPlayTactic` ("attack") and `DefenseTactic` ("defense"),
    split by `_fixed_ratio_picker` with `min_attack=3` — the tactic's full
    carrier/pivot/runner relay needs 3 robots to actually exercise the
    "switch" leg (the pivot outlet) rather than silently degrading to its
    2-robot direct-pass fallback every tick, the same floor rationale
    `build_low_block_kernel_strategy` gives `PassAndShootTactic` and
    `build_decoy_and_overload_kernel_strategy` gives `DecoyOverloadTactic`.
    Attack fraction left at a plain 0.5 split (possession-agnostic), same as
    `build_decoy_and_overload_kernel_strategy` — nothing about reading the
    weak side and relaying the ball across it is more or less urgent when the
    opponent has the ball.

    Returns a `build_kernel_strategy(motion_controller)`
    callable suitable for `AbstractStrategy`'s constructor argument of the
    same name.
    """

    def _build(motion_controller: MotionController) -> KernelSchedulerStrategy:
        ctx = KernelContext(motion_controller=motion_controller)
        return KernelSchedulerStrategy(
            tactics={"attack": SwitchOfPlayTactic(), "defense": DefenseTactic()},
            partitioner=_fixed_ratio_picker("attack", "defense", attack_fraction=0.5, min_attack=3),
            outfield_robot_ids=outfield_robot_ids,
            ctx=ctx,
        )

    return _build


# ---------------------------------------------------------------------------
# Arena strategies — plan-driven multi-tactic teams (2026-08-20 addition)
#
# The barebone factories (default/split_shape/low_block/...) exist to exercise
# the kernel machinery; these three are meant as *playable* teams: every
# posture reads live game state (ball ownership, ball zone), every posture
# has both an attack and a defense answer, and the allocation reacts to the
# game instead of being a fixed constant.
# ---------------------------------------------------------------------------


def _friendly_closer_to_ball(game: Game) -> Optional[bool]:
    """True if a friendly robot is closer to the ball than every enemy.

    None when the proximity lookup cannot read the ball's side (ball missing
    or no robots on one side) — callers should fall back to the conservative
    posture in that case.
    """
    _friendly_closest, friendly_dist = game.proximity_lookup.closest_to_ball(team_type_filter=TeamType.FRIENDLY)
    _enemy_closest, enemy_dist = game.proximity_lookup.closest_to_ball(team_type_filter=TeamType.ENEMY)
    if friendly_dist is None or enemy_dist is None:
        return None
    # Explicit `bool()`: the proximity lookup returns numpy floats, and a raw
    # comparison yields np.bool_ — whose `is True` is False, which a caller
    # checking `edge is True` would read as "unknown/losing" forever.
    return bool(friendly_dist < enemy_dist)


def _ball_zone(game: Game) -> str:
    """Where the ball sits along our attacking axis: 'own', 'mid', or 'final'.

    "Final" means the third of the pitch nearest the enemy goal we attack;
    "own" the third our own goal defends. Uses the same
    `own_goal_sign = 1.0 if my_team_is_right else -1.0` convention as
    `strategy/referee/actions.py`.
    """
    half_length = game.field.half_length
    # progress measured from our own goal line toward the enemy goal.
    own_goal_x = (1.0 if game.my_team_is_right else -1.0) * half_length
    attack_dir = -1.0 if game.my_team_is_right else 1.0
    progress = (game.ball.p.to_2d().x - own_goal_x) * attack_dir
    third = 2.0 * half_length / 3.0
    if progress < third:
        return "own"
    if progress < 2.0 * third:
        return "mid"
    return "final"


def _allocate_ordered(
    ordered: list[RobotId],
    primary: str,
    primary_n: int,
    secondary: Optional[str] = None,
) -> dict[str, frozenset[RobotId]]:
    """Split `ordered` into (primary: primary_n, secondary: rest) with coverage guaranteed.

    Only slots already confirmed available by the caller are named here; the
    caller is responsible for passing a `secondary` that is either None (when
    the primary alone may take everyone) or a real available slot id.
    """
    out: dict[str, frozenset[RobotId]] = {}
    if not ordered:
        return out
    if primary_n >= len(ordered):
        out[primary] = frozenset(ordered)
        return out
    if secondary is None:
        # Primary may take everyone (caller chose no backup slot).
        out[primary] = frozenset(ordered)
        return out
    out[primary] = frozenset(ordered[:primary_n])
    out[secondary] = frozenset(ordered[primary_n:])
    return out


def _tiki_taka_picker(
    game: Game,
    free_robots: frozenset[RobotId],
    prev_partition: Optional[dict[str, frozenset[RobotId]]],
    applicable_tactic_ids: frozenset[str],
) -> dict[str, frozenset[RobotId]]:
    """Tiki-taka posture: possession attack with give-and-go, press on loss,
    shadow-and-mark cover in both postures.

    - We have the ball (friendly closer to it, or unknown): 3 attackers
      (give-and-go trio), 2 defenders (one shadowing pair on the shot line).
    - The opponent has the ball: 3 pressers (1 ball-presser + 2 man-markers)
      and 2 shadowers — the press denies the immediate play while the shadow
      pair keeps the shot line honest behind it.
    - A slot that is unavailable (inapplicable — PressAndContain only, or
      commitment-pinned so it never appears in `applicable_tactic_ids`) has
      its share folded into the other non-pinned attacker/defender slot, so
      every free robot always lands somewhere.
    """
    ordered = sorted(free_robots)
    if not ordered:
        return {}

    friendly_edge = _friendly_closer_to_ball(game)
    losing = friendly_edge is not True  # enemy closer, or unknown -> conservative

    press_ok = "press" in applicable_tactic_ids
    attack_ok = "attack" in applicable_tactic_ids
    defense_ok = "defense" in applicable_tactic_ids

    if losing and press_ok:
        # Press with the ball-side group, shadow with the rest; if there is
        # nothing to shadow behind (no defense slot), press with everyone.
        if defense_ok:
            return _allocate_ordered(ordered, "press", 3, "defense")
        return _allocate_ordered(ordered, "press", len(ordered))
    if losing:
        # No press available: everyone shadows/marks.
        if defense_ok:
            return _allocate_ordered(ordered, "defense", len(ordered))
        if attack_ok:
            return _allocate_ordered(ordered, "attack", len(ordered))
        return {}

    # We have the ball: attack with the forward group, keep a covering pair.
    if attack_ok:
        return _allocate_ordered(ordered, "attack", 3, "defense" if defense_ok else None)
    if defense_ok:
        return _allocate_ordered(ordered, "defense", len(ordered))
    return {}


def _counter_press_picker(
    game: Game,
    free_robots: frozenset[RobotId],
    prev_partition: Optional[dict[str, frozenset[RobotId]]],
    applicable_tactic_ids: frozenset[str],
) -> dict[str, frozenset[RobotId]]:
    """Counter-press posture: all-out pressure when we lose it, low block when
    there is nothing to press, four-up on the switch when we regain it.

    - Opponent has the ball and pressing is possible: everyone presses
      (1 ball-presser + everyone else man-marking) — the ball is smothered
      where it was lost.
    - Opponent has the ball but nothing is pressable: everyone holds the
      `block_shape` zone screen — the compact low block.
    - We have the ball: the switch-of-play needs its three roles
      (carrier/pivot/runner), so 4 robots attack through the weak side while
      1 keeps the screen shape as insurance on the counter.
    """
    ordered = sorted(free_robots)
    if not ordered:
        return {}

    friendly_edge = _friendly_closer_to_ball(game)
    losing = friendly_edge is not True

    press_ok = "press" in applicable_tactic_ids
    attack_ok = "attack" in applicable_tactic_ids
    block_ok = "block" in applicable_tactic_ids

    if losing:
        if press_ok:
            return _allocate_ordered(ordered, "press", len(ordered))
        if block_ok:
            return _allocate_ordered(ordered, "block", len(ordered))
        if attack_ok:
            return _allocate_ordered(ordered, "attack", len(ordered))
        return {}
    if attack_ok:
        return _allocate_ordered(ordered, "attack", 4, "block" if block_ok else None)
    if block_ok:
        return _allocate_ordered(ordered, "block", len(ordered))
    return {}


def _zone_flow_picker(
    game: Game,
    free_robots: frozenset[RobotId],
    prev_partition: Optional[dict[str, frozenset[RobotId]]],
    applicable_tactic_ids: frozenset[str],
) -> dict[str, frozenset[RobotId]]:
    """Zone-flow posture: the attacking *pattern* changes with ball zone, the
    defense stays man-shaped throughout.

    - Opponent has the ball: everyone shadows/marks (the whole team in
      shape).
    - We have the ball in our own or middle third: a give-and-go trio works
      the ball forward while 2 shadow/mark.
    - We have the ball in the final third: the decoy-and-overload pair
      lures the last line out of position (2 robots — that tactic is a
      two-role duet by design) while the other 3 hold the defensive shape.
    """
    ordered = sorted(free_robots)
    if not ordered:
        return {}

    friendly_edge = _friendly_closer_to_ball(game)
    losing = friendly_edge is not True

    defense_ok = "defense" in applicable_tactic_ids
    givego_ok = "givego" in applicable_tactic_ids
    overload_ok = "overload" in applicable_tactic_ids

    if losing:
        if defense_ok:
            return _allocate_ordered(ordered, "defense", len(ordered))
        if givego_ok:
            return _allocate_ordered(ordered, "givego", len(ordered))
        if overload_ok:
            return _allocate_ordered(ordered, "overload", len(ordered))
        return {}

    zone = _ball_zone(game)
    if zone == "final" and overload_ok:
        return _allocate_ordered(ordered, "overload", 2, "defense" if defense_ok else None)
    if givego_ok:
        return _allocate_ordered(ordered, "givego", 3, "defense" if defense_ok else None)
    if overload_ok:
        return _allocate_ordered(ordered, "overload", len(ordered))
    if defense_ok:
        return _allocate_ordered(ordered, "defense", len(ordered))
    return {}


def build_tiki_taka_kernel_strategy(outfield_robot_ids: tuple[int, ...]):
    """Real possession-play team: give-and-go build-up, press on loss,
    shadow-and-mark cover behind every attack.

    Three concurrent slots — `GiveAndGoTactic` ("attack"),
    `PressAndContainTactic` ("press", only when an enemy is within pressing
    range of the ball — its `applicable()`), and
    `ShadowAndMarkTactic` ("defense") — allocated by `_tiki_taka_picker` on
    the possession edge: 3+2 attack/cover when the ball is ours, 3+2
    press/cover when it is lost. The give-and-go loop is the build-up engine:
    short hops under pressure instead of `PassAndShootTactic`'s scripted
    setup-then-shoot, which the match-stuck investigation showed cannot
    complete under contest.

    Returns a `build_kernel_strategy(motion_controller)`
    callable suitable for `AbstractStrategy`'s constructor argument of the
    same name.
    """

    def _build(motion_controller: MotionController) -> KernelSchedulerStrategy:
        ctx = KernelContext(motion_controller=motion_controller)
        return KernelSchedulerStrategy(
            tactics={
                "attack": GiveAndGoTactic(),
                "press": PressAndContainTactic(),
                "defense": ShadowAndMarkTactic(),
            },
            partitioner=_tiki_taka_picker,
            outfield_robot_ids=outfield_robot_ids,
            ctx=ctx,
        )

    return _build


def build_counter_press_kernel_strategy(outfield_robot_ids: tuple[int, ...]):
    """High-intensity transition team: smother the ball where it was lost,
    then switch the play to the weak side at full commitment.

    Three concurrent slots — `SwitchOfPlayTactic` ("attack"),
    `PressAndContainTactic` ("press"), and the new `BlockShapeTactic`
    ("block") — allocated by `_counter_press_picker`: everyone presses while
    the opponent has it (and something is pressable), everyone drops into
    the zone screen when they are shielded from the press, and 4 robots
    attack through the pivot/runner switch the moment the ball is won. The
    three-role switch (carrier/pivot/runner) needs at least 3 robots, hence
    the 4+1 split.

    Returns a `build_kernel_strategy(motion_controller)`
    callable suitable for `AbstractStrategy`'s constructor argument of the
    same name.
    """

    def _build(motion_controller: MotionController) -> KernelSchedulerStrategy:
        ctx = KernelContext(motion_controller=motion_controller)
        return KernelSchedulerStrategy(
            tactics={
                "attack": SwitchOfPlayTactic(),
                "press": PressAndContainTactic(),
                "block": BlockShapeTactic(),
            },
            partitioner=_counter_press_picker,
            outfield_robot_ids=outfield_robot_ids,
            ctx=ctx,
        )

    return _build


def build_zone_fluid_kernel_strategy(outfield_robot_ids: tuple[int, ...]):
    """Zone-adaptive team: the attacking pattern itself changes with ball
    position, the first strategy in the catalog with two concurrent ATTACK-
    tagged slots that the picker chooses between — exactly the use the closed
    `TacticTag` vocabulary exists for.

    Three concurrent slots — `GiveAndGoTactic` ("givego"),
    `DecoyOverloadTactic` ("overload"), and `ShadowAndMarkTactic`
    ("defense") — allocated by `_zone_flow_picker`: the whole team takes
    man-shape when the ball is lost; the give-and-go trio builds up through
    the middle thirds; and in the final third the two-robot decoy/overload
    duet replaces the trio, luring the last line out of position instead of
    passing into it.

    Returns a `build_kernel_strategy(motion_controller)`
    callable suitable for `AbstractStrategy`'s constructor argument of the
    same name.
    """

    def _build(motion_controller: MotionController) -> KernelSchedulerStrategy:
        ctx = KernelContext(motion_controller=motion_controller)
        return KernelSchedulerStrategy(
            tactics={
                "givego": GiveAndGoTactic(),
                "overload": DecoyOverloadTactic(),
                "defense": ShadowAndMarkTactic(),
            },
            partitioner=_zone_flow_picker,
            outfield_robot_ids=outfield_robot_ids,
            ctx=ctx,
        )

    return _build


# ---------------------------------------------------------------------------
# Anti-tiki_taka strategies (2026-08-20 addition)
#
# tiki_taka (`_tiki_taka_picker`) splits 3/2 in both postures: 3 attack + 2
# defense when it has the ball, 3 press + 2 defense when it doesn't. Two
# exploitable properties fall directly out of that split:
#
# 1. Its "defense" slot is `ShadowAndMarkTactic`, whose man-marking only
#    starts at the *3rd* assigned robot (robots 1-2 always shadow the shot
#    line; see the tactic's own docstring/tick — marking is
#    `robot_ids[2:]`). tiki_taka only ever gives that slot 2 robots, in
#    either posture — so its defense is *always* pure shot-line shadowing,
#    never man-marking, no matter how many attackers we send. A numbers
#    overload (more attacking bodies than tiki_taka has cover for) faces
#    zero marking, only a two-robot shadow to beat with width or a switch.
# 2. Its 3-press only forms *after* it reads possession loss — there is no
#    press while it still has the ball, and nothing pre-positioned for a
#    turnover. A fast direct counter (no scripted setup phase, re-picks its
#    leader every uncommitted tick) that strikes in the transition window
#    before the 3-press organizes skips the fight tiki_taka is built to win.
# ---------------------------------------------------------------------------


def _overload_press_picker(
    game: Game,
    free_robots: frozenset[RobotId],
    prev_partition: Optional[dict[str, frozenset[RobotId]]],
    applicable_tactic_ids: frozenset[str],
) -> dict[str, frozenset[RobotId]]:
    """Overload-and-strike posture: outnumber tiki_taka's 2-robot shadow line
    when we have the ball, hit immediately on a turnover before its press
    organizes.

    - We have the ball (or unknown): 4 robots overload (`DecoyOverloadTactic`
      lure + fill, backed by `SwitchOfPlayTactic`'s weak-side read once the
      overload draws cover across) — more attackers than tiki_taka's defense
      slot ever man-marks, since that slot never grows past 2 and only
      shadows. 1 robot holds `BlockShapeTactic` as counter insurance.
    - The opponent has the ball: everyone (or as many as `applicable_tactic_ids`
      allows) goes straight to `LeadAndSupportTactic` — a direct,
      no-setup-phase counter — rather than a organized press, to strike in
      the transition window before tiki_taka's own 3-press forms. Falls back
      to the block screen if the counter is unavailable (e.g. `switch`
      pinned mid-relay elsewhere — not expected with this tactic set, but
      every picker in this file keeps this fallback chain for the same
      reason: every free robot must land somewhere).
    """
    ordered = sorted(free_robots)
    if not ordered:
        return {}

    friendly_edge = _friendly_closer_to_ball(game)
    losing = friendly_edge is not True

    overload_ok = "overload" in applicable_tactic_ids
    switch_ok = "switch" in applicable_tactic_ids
    block_ok = "block" in applicable_tactic_ids
    counter_ok = "counter" in applicable_tactic_ids

    if losing:
        if counter_ok:
            return _allocate_ordered(ordered, "counter", len(ordered))
        if block_ok:
            return _allocate_ordered(ordered, "block", len(ordered))
        if switch_ok:
            return _allocate_ordered(ordered, "switch", len(ordered))
        return {}

    if overload_ok:
        return _allocate_ordered(ordered, "overload", 4, "block" if block_ok else None)
    if switch_ok:
        return _allocate_ordered(ordered, "switch", 4, "block" if block_ok else None)
    if block_ok:
        return _allocate_ordered(ordered, "block", len(ordered))
    return {}


def build_overload_press_kernel_strategy(outfield_robot_ids: tuple[int, ...]):
    """Numbers-overload team built to beat tiki_taka's shot-line-only shadow
    defense, with a direct pre-press counter for the transition window.

    Four concurrent slots — `DecoyOverloadTactic` ("overload"),
    `SwitchOfPlayTactic` ("switch"), `LeadAndSupportTactic` ("counter"), and
    `BlockShapeTactic` ("block") — allocated by `_overload_press_picker`:
    4 robots overload/switch the attack (more bodies than tiki_taka's 2-robot
    defense slot ever marks) with 1 held on the block screen while we have
    the ball; on loss, the whole team goes direct via the counter rather than
    building a press, to hit before tiki_taka's own 3-press organizes.

    Returns a `build_kernel_strategy(motion_controller)`
    callable suitable for `AbstractStrategy`'s constructor argument of the
    same name.
    """

    def _build(motion_controller: MotionController) -> KernelSchedulerStrategy:
        ctx = KernelContext(motion_controller=motion_controller)
        return KernelSchedulerStrategy(
            tactics={
                "overload": DecoyOverloadTactic(),
                "switch": SwitchOfPlayTactic(),
                "counter": LeadAndSupportTactic(),
                "block": BlockShapeTactic(),
            },
            partitioner=_overload_press_picker,
            outfield_robot_ids=outfield_robot_ids,
            ctx=ctx,
        )

    return _build


def _high_line_zone_picker(
    game: Game,
    free_robots: frozenset[RobotId],
    prev_partition: Optional[dict[str, frozenset[RobotId]]],
    applicable_tactic_ids: frozenset[str],
) -> dict[str, frozenset[RobotId]]:
    """High-line zone posture: deny tiki_taka's give-and-go trio the 1v1s it
    wants with a zone screen instead of man-marking, switch the ball to the
    weak side its 2-robot defense can't cover, finish with the overload duet
    once we're through.

    - The opponent has the ball: the whole team holds `BlockShapeTactic`'s
      shifting zone line — tiki_taka's give-and-go build-up is a 3-robot
      short-passing relay that thrives against man-marking cover it can
      dribble/pass past one defender at a time; a zone screen that shifts
      with the ball and never breaks shape denies it the individual
      matchups it's built around, unlike `ShadowAndMarkTactic`'s greedy
      per-robot marking (which is what tiki_taka's own defense runs, and
      exactly the shape `overload_press` targets instead).
    - We have the ball, not yet in the final third: `SwitchOfPlayTactic`
      leads — tiki_taka's defense is only ever 2 robots, so a genuine
      weak-side imbalance is easy to manufacture; 2 hold the block screen as
      insurance against the counter.
    - We have the ball in the final third: switch to the overload duet
      (`DecoyOverloadTactic`) to finish, same final-third handoff
      `_zone_flow_picker` uses — the switch has already done its job of
      breaking the defense's shape open by this point.
    """
    ordered = sorted(free_robots)
    if not ordered:
        return {}

    block_ok = "block" in applicable_tactic_ids
    switch_ok = "switch" in applicable_tactic_ids
    overload_ok = "overload" in applicable_tactic_ids

    # Sticky possession edge: a plain `_friendly_closer_to_ball` re-read every
    # tick flips constantly in a genuinely contested 50/50 (measured: switch
    # assigned and released again within single-digit ticks, over and over,
    # for the first ~10s of a live match against tiki_taka) — `switch`'s
    # carrier/pivot/runner relay needs several seconds to settle and never
    # got the chance, discarded before `is_committed()` ever saw it commit.
    # `prev_partition` is this picker's only persistent state (a `Partitioner`
    # is a plain function, no `mem` of its own — unlike a `Tactic`, which
    # gets one), so use "did we hold switch/overload last tick" as the
    # attacking side's memory and require a full possession loss (proximity
    # edge, not just "not clearly ahead") before dropping it. Mirrors
    # `SwitchOfPlayTactic`'s own internal weak-side hysteresis, one level up.
    prev_partition = prev_partition or {}
    currently_attacking = bool(prev_partition.get("switch")) or bool(prev_partition.get("overload"))
    friendly_edge = _friendly_closer_to_ball(game)
    if currently_attacking:
        losing = friendly_edge is False  # only give up the ball on a clear loss, not just "unknown"
    else:
        losing = friendly_edge is not True  # regaining needs a clear win, same conservative default as elsewhere

    if losing:
        if block_ok:
            return _allocate_ordered(ordered, "block", len(ordered))
        if switch_ok:
            return _allocate_ordered(ordered, "switch", len(ordered))
        if overload_ok:
            return _allocate_ordered(ordered, "overload", len(ordered))
        return {}

    zone = _ball_zone(game)
    if zone == "final" and overload_ok:
        return _allocate_ordered(ordered, "overload", 2, "block" if block_ok else None)
    if switch_ok:
        return _allocate_ordered(ordered, "switch", 3, "block" if block_ok else None)
    if overload_ok:
        return _allocate_ordered(ordered, "overload", len(ordered))
    if block_ok:
        return _allocate_ordered(ordered, "block", len(ordered))
    return {}


def _counter_flow_picker(
    game: Game,
    free_robots: frozenset[RobotId],
    prev_partition: Optional[dict[str, frozenset[RobotId]]],
    applicable_tactic_ids: frozenset[str],
) -> dict[str, frozenset[RobotId]]:
    """Counter-flow posture: fight tiki_taka on its own strongest ground —
    beat its give-and-go attack with our own give-and-go, contest its press
    window with our own press instead of ceding it, and fall back to a
    space-denying screen (not man-shadow) when neither side has an edge.

    Both `overload_press` and `high_line_zone` tried to out-number
    tiki_taka's 2-robot shadow defense and lost before that numbers edge
    ever got to matter — tiki_taka's 3-robot press (1 presser + 2 markers,
    `PressAndContainTactic` only becomes applicable within 1.5 m of the
    ball) smothered both attempts in the transition window, every time. The
    lesson from that failure isn't "the overload theory was wrong" — it's
    that neither prior attempt had a real answer to the press itself:
    `LeadAndSupportTactic` never passes at all (dribble-only, easy to
    press-trap), and the switch-of-play relay needs several seconds to
    settle that the press never gave it. `GiveAndGoTactic` is the one
    attacking tactic in the catalog actually built to survive exactly this
    — cycle a fresh hop the instant the current lane closes, shoot the
    moment one opens — and it is tiki_taka's *own* attack engine, already
    proven undefeated. This posture borrows it rather than assuming our own
    novel attack pattern would fare better than tiki_taka's did.

    - We have the ball (or unknown): 3 attack (give-and-go trio, matching
      tiki_taka's own commitment there so we are not thinner in possession
      than the team we are trying to out-cycle), 2 hold the block screen —
      space denial instead of `ShadowAndMarkTactic`'s greedy man-marking,
      so committing bodies forward never leaves us exposed to the
      clustering/defense-area fouls a shadow line invites under pressure.
    - The opponent has the ball and is pressable: press with 3 (matching
      tiki_taka's own press numbers instead of ceding the transition
      window to it, which is exactly the window both prior attempts lost
      in), block screen with the rest.
    - The opponent has the ball but is not pressable (out of press range):
      the whole team holds the block screen — no reason to commit anyone
      forward with no ball-side trigger.

    Sticky possession edge, same fix `_high_line_zone_picker` needed: a
    plain re-read of `_friendly_closer_to_ball` every tick flips constantly
    in a genuinely contested match (measured on the first real run against
    tiki_taka: attack/press alternated every 1-3 s for the entire first
    7 s, `GiveAndGoTactic`'s hop-cycle reset before a single hop completed,
    6%/94% possession, 4.8 m of ball travel in 60 s). `prev_partition` is
    this picker's only persistent state, so use "did we hold attack last
    tick" as memory and require a clear loss (not just "not clearly ahead")
    before giving it up — mirrors `_high_line_zone_picker`'s identical fix.
    """
    ordered = sorted(free_robots)
    if not ordered:
        return {}

    prev_partition = prev_partition or {}
    currently_attacking = bool(prev_partition.get("attack"))
    friendly_edge = _friendly_closer_to_ball(game)
    if currently_attacking:
        losing = friendly_edge is False
    else:
        losing = friendly_edge is not True

    attack_ok = "attack" in applicable_tactic_ids
    press_ok = "press" in applicable_tactic_ids
    block_ok = "block" in applicable_tactic_ids

    if losing:
        if press_ok:
            return _allocate_ordered(ordered, "press", 3, "block" if block_ok else None)
        if block_ok:
            return _allocate_ordered(ordered, "block", len(ordered))
        if attack_ok:
            return _allocate_ordered(ordered, "attack", len(ordered))
        return {}

    if attack_ok:
        return _allocate_ordered(ordered, "attack", 3, "block" if block_ok else None)
    if block_ok:
        return _allocate_ordered(ordered, "block", len(ordered))
    return {}


def build_counter_flow_kernel_strategy(outfield_robot_ids: tuple[int, ...]):
    """Direct answer to tiki_taka: our own give-and-go attack, our own press
    on the ball, a space-denying screen everywhere else.

    Three concurrent slots — `GiveAndGoTactic` ("attack"),
    `PressAndContainTactic` ("press"), and `BlockShapeTactic` ("block") —
    allocated by `_counter_flow_picker` on a 3/2 split matching tiki_taka's
    own commitment in both postures, rather than the numbers-overload bet
    `overload_press`/`high_line_zone` made and lost. The theory: neither
    prior anti-tiki_taka attempt ever got far enough to test its numbers
    edge because tiki_taka's press won the transition window outright: this
    strategy contests that window symmetrically (our own 3-press) instead
    of trying to dodge it, and uses tiki_taka's own proven-undefeated attack
    engine rather than a fresh pattern.

    Returns a `build_kernel_strategy(motion_controller)`
    callable suitable for `AbstractStrategy`'s constructor argument of the
    same name.
    """

    def _build(motion_controller: MotionController) -> KernelSchedulerStrategy:
        ctx = KernelContext(motion_controller=motion_controller)
        return KernelSchedulerStrategy(
            tactics={
                "attack": GiveAndGoTactic(),
                "press": PressAndContainTactic(),
                "block": BlockShapeTactic(),
            },
            partitioner=_counter_flow_picker,
            outfield_robot_ids=outfield_robot_ids,
            ctx=ctx,
        )

    return _build


def build_high_line_zone_kernel_strategy(outfield_robot_ids: tuple[int, ...]):
    """Zone-defense team built to deny tiki_taka's give-and-go trio the 1v1s
    it's designed around, switching the ball past its thin 2-robot cover.

    Three concurrent slots — `SwitchOfPlayTactic` ("switch"),
    `DecoyOverloadTactic` ("overload"), and `BlockShapeTactic` ("block") —
    allocated by `_high_line_zone_picker`: the whole team holds the zone
    screen when the ball is lost (denying man-marking 1v1s instead of
    running them, unlike tiki_taka's own `ShadowAndMarkTactic` defense);
    3 robots read the weak side and switch there once we have it (tiki_taka's
    defense is always just 2 robots — an imbalance is easy to create); the
    final third hands off to the overload duet to finish.

    Returns a `build_kernel_strategy(motion_controller)`
    callable suitable for `AbstractStrategy`'s constructor argument of the
    same name.
    """

    def _build(motion_controller: MotionController) -> KernelSchedulerStrategy:
        ctx = KernelContext(motion_controller=motion_controller)
        return KernelSchedulerStrategy(
            tactics={
                "switch": SwitchOfPlayTactic(),
                "overload": DecoyOverloadTactic(),
                "block": BlockShapeTactic(),
            },
            partitioner=_high_line_zone_picker,
            outfield_robot_ids=outfield_robot_ids,
            ctx=ctx,
        )

    return _build
