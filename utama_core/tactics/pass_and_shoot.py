"""Pass-and-shoot attack tactic: one scripted setup -> pass -> shoot sequence
for a fixed pair of robots.

Ported from `utama_strategy.functional.strategies.two_robot_attack` (that
name refers to the source module in Utama-Strategy, unchanged there; this
port was renamed to `pass_and_shoot` to name the actual behavior once
`GiveAndGoTactic` — a repeated-hop pass cycle — made "two robot attack" an
ambiguous name for two different tactics).
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

from utama_core.config.settings import CONTROL_FREQUENCY
from utama_core.engine.context import KernelContext
from utama_core.engine.tactic import BaseTactic, RobotId, TacticTag
from utama_core.entities.data.command import RobotCommand
from utama_core.entities.data.object import TeamType
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game
from utama_core.tactics._pass_and_score import (
    PassAndScoreMem,
    _pass_exec,
    _score_goal,
    _setup_positions,
    run_setup_phase,
)

_REASSIGN_MARGIN_M = 0.3  # metres the non-passer must be closer by before roles flip


def assign_passer_receiver(
    game: Game, robot_ids: tuple[int, int], prev_assignment: Optional[tuple[int, int]] = None
) -> tuple[int, int]:
    """Closest-to-ball becomes passer, the other becomes receiver.

    Falls back to `(robot_ids[0], robot_ids[1])` if the ball or a robot
    can't be found.

    Hysteresis: when both robots start a setup phase near the ball (the
    common case — that's exactly when this gets called), naive per-tick
    closest-to-ball is noisy enough to flip the "closest" robot most ticks.
    Since a flip resets `PassAndScoreMem` (see `tick()`), that thrashing
    means the passer never gets two consecutive ticks to make progress —
    `run_setup_phase` never completes because it's constantly starting
    over. Keep the previous passer unless the other robot is closer by a
    real margin, not just nominally closer.
    """
    ball_pos = game.ball.p.to_2d()
    distances = {
        rid: game.friendly_robots[rid].p.distance_to(ball_pos) for rid in robot_ids if rid in game.friendly_robots
    }

    if not distances:
        return robot_ids[0], next(rid for rid in robot_ids if rid != robot_ids[0])

    if prev_assignment is not None and prev_assignment[0] in distances and prev_assignment[1] in distances:
        prev_passer, prev_receiver = prev_assignment
        if distances[prev_receiver] + _REASSIGN_MARGIN_M >= distances[prev_passer]:
            return prev_passer, prev_receiver

    passer_id = min(distances, key=distances.get)
    receiver_id = next(rid for rid in robot_ids if rid != passer_id)
    return passer_id, receiver_id


@dataclass
class PassAndShootMem:
    pass_and_score: PassAndScoreMem
    assigned_pair: Optional[tuple[int, int]] = None


_PHASE_TIMEOUT_TIME = 12.0  # seconds — generous budget for the full aim+position+kick+catch(+aim+shoot) chain
_PHASE_TIMEOUT_TICKS = round(_PHASE_TIMEOUT_TIME * CONTROL_FREQUENCY)


class PassAndShootTactic(BaseTactic[PassAndShootMem]):
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

    def initial_mem(self) -> PassAndShootMem:
        return PassAndShootMem(pass_and_score=PassAndScoreMem())

    def is_committed(self, game: Game, mem: PassAndShootMem) -> bool:
        if mem is None or mem.assigned_pair is None:
            return False
        return mem.pass_and_score.phase != "setup"

    def tick(
        self, game: Game, ctx: KernelContext, robot_ids: tuple[RobotId, ...], mem: PassAndShootMem
    ) -> tuple[dict[RobotId, RobotCommand], PassAndShootMem]:
        pair = (robot_ids[0], robot_ids[1])

        if mem.assigned_pair is None or mem.pass_and_score.phase == "setup":
            passer_id, receiver_id = assign_passer_receiver(game, pair, prev_assignment=mem.assigned_pair)
        else:
            passer_id, receiver_id = mem.assigned_pair

        if mem.assigned_pair != (passer_id, receiver_id):
            mem.pass_and_score = PassAndScoreMem()
            mem.assigned_pair = (passer_id, receiver_id)

        if mem.pass_and_score.goal_scored:
            # A goal doesn't end the tactic's usefulness for the rest of the
            # match: the ref resets the ball, so start a fresh attempt rather
            # than sitting idle. Re-picking passer/receiver here (rather than
            # keeping the pinned pair) is deliberate: a completed attempt is
            # exactly the point where `pass_and_shoot`'s own docstring says
            # reassignment is safe.
            mem.pass_and_score = PassAndScoreMem()
            mem.assigned_pair = None
            return {}, mem

        inner = mem.pass_and_score

        # A pass/shot attempt that stalls (ball lost, orientation tolerance
        # never satisfied, etc.) previously left the tactic permanently
        # committed with no way back to "setup" — is_committed() only
        # releases once phase == "setup", but nothing ever set it back.
        # Time out of a stuck phase instead of deadlocking for the rest of
        # the match. This budget now also covers "setup" itself: that phase
        # is exempt from is_committed() (see class docstring), so the kernel
        # never reassigns the slot's robots away from it either — a stall
        # here (e.g. the passer can't stabilize dribbling the ball to
        # passer_position) previously had *no* timeout at all, unlike every
        # later phase, and could hold the tactic for the rest of the match
        # with zero progress (observed: 30s straight, has_ball flickering,
        # distance to target never converging). Re-sample setup
        # positions/pairing on timeout rather than retrying the exact same
        # targets that just failed to converge.
        inner.phase_ticks += 1
        if inner.phase_ticks > _PHASE_TIMEOUT_TICKS:
            inner = PassAndScoreMem()
            mem.assigned_pair = None

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
            command, scored, inner.prev_best_shot_y = _score_goal(game, ctx, receiver_id, inner.prev_best_shot_y)
            commands = {receiver_id: command}
            if scored:
                inner.goal_scored = True
        else:
            commands = {}

        mem.pass_and_score = inner
        return commands, mem
