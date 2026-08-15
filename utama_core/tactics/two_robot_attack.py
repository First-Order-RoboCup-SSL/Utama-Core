"""Two-robot dynamic-role attack tactic.

Ported from `utama_strategy.functional.strategies.two_robot_attack`.
Closest-robot-to-ball becomes passer, the other receiver, then runs a
setup -> pass -> score sequence.

This is the concrete source of the `is_committed()` concept in the kernel
design: role assignment is only safe to re-run during the "setup" phase.
Once the passer has picked up the ball and the pass sequence has started,
re-running closest-to-ball every tick is wrong — the ball is briefly closer
to the receiver than the passer for most of the pass by design — which
would otherwise flip roles mid-pass. The original functional spike guarded
this with an inline phase check
(`if mem.assigned_pair is None or mem.pass_and_score.phase == "setup":`)
before deciding whether to re-run `assign_passer_receiver`. `is_committed()`
is that same check, exposed to the kernel so reassignment of this tactic's
robots is refused (not just role reassignment *within* the tactic) for as
long as it's true.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from utama_core.entities.data.command import RobotCommand
from utama_core.entities.data.object import TeamType
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game
from utama_core.kernel.context import KernelContext
from utama_core.kernel.tactic import BaseTactic, RobotId, TacticTag
from utama_core.tactics._pass_and_score import (
    PassAndScoreMem,
    _pass_exec,
    _score_goal,
    _setup_positions,
    run_setup_phase,
)


def assign_passer_receiver(game: Game, robot_ids: tuple[int, int]) -> tuple[int, int]:
    """Closest-to-ball becomes passer, the other becomes receiver.

    Falls back to `(robot_ids[0], robot_ids[1])` if the ball or a robot
    can't be found.
    """
    closest, _distance = game.proximity_lookup.closest_to_ball(team_type_filter=TeamType.FRIENDLY)
    passer_id = closest.id if closest is not None and closest.id in robot_ids else robot_ids[0]
    receiver_id = next(rid for rid in robot_ids if rid != passer_id)
    return passer_id, receiver_id


@dataclass
class TwoRobotAttackMem:
    pass_and_score: PassAndScoreMem
    assigned_pair: Optional[tuple[int, int]] = None


class TwoRobotAttackTactic(BaseTactic[TwoRobotAttackMem]):
    """Two attacking robots, dynamic passer/receiver role by ball proximity."""

    tag = TacticTag.ATTACK

    def __init__(
        self,
        passer_pos: Vector2D = Vector2D(2, 1.9),
        receiver_pos: Vector2D = Vector2D(2, -1.9),
        dynamic_setup: bool = True,
    ):
        self.passer_pos = passer_pos
        self.receiver_pos = receiver_pos
        self.dynamic_setup = dynamic_setup

    def initial_mem(self) -> TwoRobotAttackMem:
        return TwoRobotAttackMem(pass_and_score=PassAndScoreMem())

    def is_committed(self, game: Game, mem: TwoRobotAttackMem) -> bool:
        if mem is None or mem.assigned_pair is None:
            return False
        return mem.pass_and_score.phase != "setup"

    def tick(
        self, game: Game, ctx: KernelContext, robot_ids: tuple[RobotId, ...], mem: TwoRobotAttackMem
    ) -> tuple[dict[RobotId, RobotCommand], TwoRobotAttackMem]:
        pair = (robot_ids[0], robot_ids[1])

        if mem.assigned_pair is None or mem.pass_and_score.phase == "setup":
            passer_id, receiver_id = assign_passer_receiver(game, pair)
        else:
            passer_id, receiver_id = mem.assigned_pair

        if mem.assigned_pair != (passer_id, receiver_id):
            mem.pass_and_score = PassAndScoreMem()
            mem.assigned_pair = (passer_id, receiver_id)

        if mem.pass_and_score.goal_scored:
            return {}, mem

        inner = mem.pass_and_score
        inner = _setup_positions(
            game, inner, passer_id, receiver_id, self.passer_pos, self.receiver_pos, self.dynamic_setup
        )

        if inner.phase == "setup":
            commands, complete = run_setup_phase(game, ctx, passer_id, receiver_id, inner)
            if complete:
                inner.phase = "pass_then_score"
        elif inner.phase == "pass_then_score":
            commands, pass_complete = _pass_exec(game, ctx, passer_id, receiver_id)
            if pass_complete:
                inner.phase = "score"
        elif inner.phase == "score":
            command, scored = _score_goal(game, ctx, receiver_id)
            commands = {receiver_id: command}
            if scored:
                inner.goal_scored = True
        else:
            commands = {}

        mem.pass_and_score = inner
        return commands, mem
