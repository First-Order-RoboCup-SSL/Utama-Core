"""Referee-restart legality override for the kernel model.

`Strategy.tick()` (and `referee_reset.py`) only handle the *scheduling*
question — wiping `mem`/`is_committed()` on a barrier reset, freezing motion
entirely on HALT/STOP. They say nothing about *what a robot should physically
do* during a restart, so left alone, a Tactic's normal logic keeps running
during e.g. an opponent's ball placement — driving straight at the ball,
which is an SSL rule violation, not just a scheduling wrinkle.

The old BT path (`actions.py`, plus a `tree.py` that no longer exists — see
`docs/referee_integration.md`, now stale) already solved this: a priority
Selector matches the current `RefereeCommand` and, when matched, takes over
every friendly robot's command for that tick, bypassing the strategy tree
entirely. This module reuses that same logic (the `*Step` `AbstractBehaviour`
subclasses in `utama_core/custom_referee/actions.py`) rather than
reimplementing keep-out-distance geometry — those classes only ever touch
`blackboard.game`, `blackboard.motion_controller`, and `blackboard.cmd_map`
(verified: no other blackboard key is read), so a tiny duck-typed shim
exposing just those three attributes is enough to drive them outside py_trees.

`HALT`/`STOP`/`TIMEOUT_*` are deliberately NOT handled here: `Strategy.tick()`
already returns `{}` for those via `is_paused` before any tactic (or this
override) would run, which satisfies the same "stop issuing motion" rule more
directly than routing through `StopStep`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from utama_core.custom_referee.actions import (
    BallPlacementOursStep,
    BallPlacementTheirsStep,
    DirectFreeOursStep,
    DirectFreeTheirsStep,
    PrepareKickoffOursStep,
    PrepareKickoffTheirsStep,
    PreparePenaltyOursStep,
    PreparePenaltyTheirsStep,
)
from utama_core.entities.data.command import RobotCommand
from utama_core.entities.game import Game
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.kernel.tactic import RobotId
from utama_core.motion_planning.src.common.motion_controller import MotionController

# Commands the BT path treats as restarts requiring legal-position override
# (i.e. everything in `_REFEREE_STOPPAGE_COMMANDS` except HALT/STOP/TIMEOUT_*,
# which `Strategy.tick()`'s `is_paused` check already handles by issuing no
# commands at all).
_OVERRIDE_COMMANDS = frozenset(
    {
        RefereeCommand.PREPARE_KICKOFF_YELLOW,
        RefereeCommand.PREPARE_KICKOFF_BLUE,
        RefereeCommand.PREPARE_PENALTY_YELLOW,
        RefereeCommand.PREPARE_PENALTY_BLUE,
        RefereeCommand.DIRECT_FREE_YELLOW,
        RefereeCommand.DIRECT_FREE_BLUE,
        RefereeCommand.INDIRECT_FREE_YELLOW,
        RefereeCommand.INDIRECT_FREE_BLUE,
        RefereeCommand.BALL_PLACEMENT_YELLOW,
        RefereeCommand.BALL_PLACEMENT_BLUE,
    }
)


def is_override_command(command: Optional[RefereeCommand]) -> bool:
    return command in _OVERRIDE_COMMANDS


@dataclass
class _BlackboardShim:
    """The only three attributes any `actions.py` Step class reads or writes."""

    game: Game
    motion_controller: MotionController
    cmd_map: dict[RobotId, RobotCommand] = field(default_factory=dict)


class RefereeOverride:
    """Stateful dispatcher mirroring `build_referee_override_tree`'s command→Step mapping.

    Kept as long-lived instances (one `RefereeOverride` per `Strategy`, not
    reconstructed per tick) because `BallPlacementOursStep` carries its own
    cross-tick state (`_release_started_at`/`_placer_id`) exactly like it does
    on the BT path — a fresh instance every tick would lose that state and
    re-trigger the release-delay logic every tick.
    """

    def __init__(self):
        self._ball_placement_ours = BallPlacementOursStep(name="BallPlacementOurs")
        self._ball_placement_ours.setup_()
        self._ball_placement_theirs = BallPlacementTheirsStep(name="BallPlacementTheirs")
        self._kickoff_ours = PrepareKickoffOursStep(name="KickoffOurs")
        self._kickoff_theirs = PrepareKickoffTheirsStep(name="KickoffTheirs")
        self._penalty_ours = PreparePenaltyOursStep(name="PenaltyOurs")
        self._penalty_theirs = PreparePenaltyTheirsStep(name="PenaltyTheirs")
        self._direct_free_ours = DirectFreeOursStep(name="DirectFreeOurs")
        self._direct_free_theirs = DirectFreeTheirsStep(name="DirectFreeTheirs")

    def tick(
        self, game: Game, motion_controller: MotionController, command: RefereeCommand
    ) -> dict[RobotId, RobotCommand]:
        """Compute every friendly robot's command for this restart, or {} if not an override command."""
        step = self._step_for(command, game.my_team_is_yellow)
        if step is None:
            return {}
        shim = _BlackboardShim(game=game, motion_controller=motion_controller)
        step.blackboard = shim
        step.update()
        return dict(shim.cmd_map)

    def _step_for(self, command: RefereeCommand, my_team_is_yellow: bool):
        if command is RefereeCommand.BALL_PLACEMENT_YELLOW:
            return self._ball_placement_ours if my_team_is_yellow else self._ball_placement_theirs
        if command is RefereeCommand.BALL_PLACEMENT_BLUE:
            return self._ball_placement_theirs if my_team_is_yellow else self._ball_placement_ours

        if command is RefereeCommand.PREPARE_KICKOFF_YELLOW:
            return self._kickoff_ours if my_team_is_yellow else self._kickoff_theirs
        if command is RefereeCommand.PREPARE_KICKOFF_BLUE:
            return self._kickoff_theirs if my_team_is_yellow else self._kickoff_ours

        if command is RefereeCommand.PREPARE_PENALTY_YELLOW:
            return self._penalty_ours if my_team_is_yellow else self._penalty_theirs
        if command is RefereeCommand.PREPARE_PENALTY_BLUE:
            return self._penalty_theirs if my_team_is_yellow else self._penalty_ours

        if command in (RefereeCommand.DIRECT_FREE_YELLOW, RefereeCommand.INDIRECT_FREE_YELLOW):
            return self._direct_free_ours if my_team_is_yellow else self._direct_free_theirs
        if command in (RefereeCommand.DIRECT_FREE_BLUE, RefereeCommand.INDIRECT_FREE_BLUE):
            return self._direct_free_theirs if my_team_is_yellow else self._direct_free_ours

        return None
