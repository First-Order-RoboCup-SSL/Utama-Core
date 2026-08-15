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

from dataclasses import dataclass

from utama_core.entities.data.command import RobotCommand
from utama_core.entities.game import Game
from utama_core.kernel.context import KernelContext
from utama_core.kernel.tactic import BaseTactic, RobotId, TacticTag
from utama_core.skills.src.goalkeep import goalkeep
from utama_core.skills.src.utils.move_utils import empty_command


@dataclass
class GoalkeeperMem:
    """No cross-tick state: goalkeeping recomputes its target from scratch every tick."""


class GoalkeeperTactic(BaseTactic[GoalkeeperMem]):
    """One robot, tracks and blocks the ball at the goal line. Never kicks."""

    tag = TacticTag.DEFENSE

    def __init__(self, robot_id: int = 0):
        self.robot_id = robot_id

    def initial_mem(self) -> GoalkeeperMem:
        return GoalkeeperMem()

    def tick(
        self, game: Game, ctx: KernelContext, robot_ids: tuple[RobotId, ...], mem: GoalkeeperMem
    ) -> tuple[dict[RobotId, RobotCommand], GoalkeeperMem]:
        command = goalkeep(game, ctx.motion_controller, self.robot_id)
        if command is None:
            command = empty_command(False)
        return {self.robot_id: command}, mem
