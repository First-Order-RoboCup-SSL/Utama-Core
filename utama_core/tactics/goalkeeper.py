"""Goalkeeper tactic.

Ported from `utama_strategy.functional.strategies.goalkeeper.GoalkeeperStrategy`.

The goalkeeper (robot 0, by SSL/team convention) is pinned outside the
`Strategy` scheduler entirely — see the "Tactics as Processes" design note,
section 1: "Robot 0 (goalkeeper) is pinned, never scheduled." This class is
therefore not registered with `Strategy` as a normal tactic competing for
the outfield robot pool; a caller ticks it directly for robot 0, once per
game tick, alongside whatever `Strategy` decides for the outfield robots.

It still exposes the same `tick`/`initial_mem`/`committed` shape as any
other tactic (matching `kernel.tactic.Tactic`) purely for consistency of
authoring style — not because the kernel schedules it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from utama_core.engine.context import TickContext
from utama_core.engine.tactic import BaseTactic, RobotId, TacticTag
from utama_core.entities.data.command import RobotCommand
from utama_core.entities.game import Game
from utama_core.motion_planning.src.controllers.pid_controller import PIDController
from utama_core.skills.src.goalkeep import goalkeep
from utama_core.skills.src.utils.move_utils import empty_command


@dataclass
class GoalkeeperMem:
    """Caches the keeper's own dedicated `PIDController` (see
    `GoalkeeperTactic.tick`'s docstring for why the keeper never uses
    `ctx.motion_controller`) — built lazily on first tick since it needs
    `ctx.motion_controller`'s `mode`/`rsim_env` to construct, neither of
    which is available in `initial_mem`."""

    pid_controller: Optional[PIDController] = field(default=None)


class GoalkeeperTactic(BaseTactic[GoalkeeperMem]):
    """One robot, tracks and blocks the ball at the goal line. Never kicks."""

    tag = TacticTag.DEFENSE

    def __init__(self, robot_id: int = 0):
        self.robot_id = robot_id

    def initial_mem(self) -> GoalkeeperMem:
        return GoalkeeperMem()

    def tick(
        self, game: Game, ctx: TickContext, robot_ids: tuple[RobotId, ...], mem: GoalkeeperMem
    ) -> tuple[dict[RobotId, RobotCommand], GoalkeeperMem]:
        # Deliberately never uses ctx.motion_controller (the team's shared
        # scheme, "fpp"/FastPathPlanningController by default): a live
        # tournament trace (2026-08-26, gap #6/#9 validation run —
        # replays/gap6_validation_20260826_124653/counter_flow_vs_tiki_taka_RK.pkl)
        # found the keeper oscillating with a ~1.6s period and ~0.74m
        # amplitude around a completely static target (ball at rest,
        # predicted goal-line intercept fixed to <1e-6 drift) during
        # PREPARE_KICKOFF, never converging. Isolated PID gains alone
        # (same gains, same start/target) converged cleanly in isolation,
        # so the oscillation only reproduces through FastPathPlanner's
        # carrot/detour-side routing — plausibly its persistent per-robot
        # "last chosen detour side" cache flip-flopping against an
        # obstacle the keeper doesn't actually need to route around for a
        # simple hold-a-goal-line-point motion. The keeper's task never
        # needs obstacle-avoidance path planning (it holds a point on its
        # own goal line, inside its own defense area, where no legal
        # opponent or teammate should be routing through), so it gets its
        # own dedicated PIDController instead of sharing the team's
        # scheme — sidesteps the planner entirely rather than debugging
        # its detour logic, and cannot regress any other tactic since
        # nothing else references this controller.
        if mem.pid_controller is None:
            mem.pid_controller = PIDController(ctx.motion_controller.mode, ctx.motion_controller.rsim_env)

        command = goalkeep(game, mem.pid_controller, self.robot_id)
        if command is None:
            command = empty_command(False)
        return {self.robot_id: command}, mem
