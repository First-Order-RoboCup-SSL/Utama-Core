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
from utama_core.kernel.strategy import Strategy as KernelSchedulerStrategy
from utama_core.kernel.tactic import RobotId
from utama_core.motion_planning.src.common.motion_controller import MotionController
from utama_core.strategy.common.abstract_strategy import (
    AbstractStrategy,
    SpaceRequirements,
)
from utama_core.tactics.goalkeeper import GoalkeeperTactic
from utama_core.tactics.lead_and_support import LeadAndSupportTactic
from utama_core.tactics.shadow_and_mark import ShadowAndMarkTactic
from utama_core.tactics.two_robot_attack import TwoRobotAttackTactic


class KernelStrategy(AbstractStrategy):
    """`AbstractStrategy` subclass that delegates ticking to a `kernel.Strategy`.

    Args:
        build_kernel_strategy: called once, after `load_game`, as
            `build_kernel_strategy(game, motion_controller, rsim_env) ->
            kernel.Strategy`. Deferred to a factory (rather than passed
            pre-built) because the motion controller is only available on
            the blackboard once `StrategyRunner` calls `load_motion_controller`
            — which happens before `load_game` in `StrategyRunner.__init__`,
            so it's safe to read here, but not at `KernelStrategy.__init__`.
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
        self._goalkeeper_mem = self._goalkeeper.make_initial_mem()
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
            rsim_env = self.blackboard.rsim_env
            self._kernel_strategy = self._build_kernel_strategy(game, motion_controller, rsim_env)

    def step(self):
        game = self.blackboard.game

        outfield_commands = self._kernel_strategy.tick(game)
        gk_commands, self._goalkeeper_mem = self._goalkeeper.tick(
            game, self._kernel_strategy._ctx, (self._goalkeeper_id,), self._goalkeeper_mem
        )

        cmd_map: dict[int, RobotCommand] = {}
        cmd_map.update(outfield_commands)
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
        that slot is `committed()` (the actual "why won't this reassign"
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

    Returns a `build_kernel_strategy(game, motion_controller, rsim_env)`
    callable suitable for `KernelStrategy`'s constructor argument of the
    same name.
    """

    def _build(game: Game, motion_controller: MotionController, rsim_env: object | None) -> KernelSchedulerStrategy:
        ctx = KernelContext(motion_controller=motion_controller, rsim_env=rsim_env)
        return KernelSchedulerStrategy(
            tactics={"two_robot_attack": TwoRobotAttackTactic()},
            group_picker=KernelSchedulerStrategy.single_tactic_picker(lambda game, active: "two_robot_attack"),
            outfield_robot_ids=outfield_robot_ids,
            ctx=ctx,
        )

    return _build


def _possession_split_picker(
    game: Game,
    free_robots: frozenset[RobotId],
    prev_partition: Optional[dict[str, frozenset[RobotId]]],
) -> dict[str, frozenset[RobotId]]:
    """Whichever side is closer to the ball decides posture; the split is fixed once decided.

    Deliberately the simplest rule that gives the split-shape scheduler
    something real to react to, not a scored/tunable allocator (see the
    design doc's stance against bid/fitness-scoring machinery) — one signal
    (which team is closer to the ball), two fixed splits. Ties and an
    unreadable proximity lookup default to the more conservative
    (defense-heavy) split.

    Only emits a key for a tactic it is actually assigning free robots to —
    never a zero-robot entry, since `Strategy` treats every key in a
    `GroupPicker`'s return value as "I am claiming this tactic id right now,"
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

    Returns a `build_kernel_strategy(game, motion_controller, rsim_env)`
    callable suitable for `KernelStrategy`'s constructor argument of the
    same name.
    """

    def _build(game: Game, motion_controller: MotionController, rsim_env: object | None) -> KernelSchedulerStrategy:
        ctx = KernelContext(motion_controller=motion_controller, rsim_env=rsim_env)
        return KernelSchedulerStrategy(
            tactics={"attack": LeadAndSupportTactic(), "defense": ShadowAndMarkTactic()},
            group_picker=_possession_split_picker,
            outfield_robot_ids=outfield_robot_ids,
            ctx=ctx,
        )

    return _build
