"""KeepOutRule: SSL rulebook §8.4.3 "Defender Too Close To Ball"."""

from __future__ import annotations

import math
from typing import Optional  # used by BaseRule.check return type

from utama_core.custom_referee.geometry import RefereeGeometry
from utama_core.custom_referee.rules.base_rule import BaseRule, RuleViolation
from utama_core.entities.game.game_frame import GameFrame
from utama_core.entities.referee.referee_command import RefereeCommand

# §8.4.3 applies "during an opponent kick-off or free kick" only. STOP carries
# "no automatic sanction" for being too close, and a robot out of position at a
# penalty is §8.3.4 "Disrespect Procedures" -- a human referee's call, and the
# defending keeper stands on its goal line anyway.
_KICK_COMMANDS = {
    RefereeCommand.DIRECT_FREE_YELLOW,
    RefereeCommand.DIRECT_FREE_BLUE,
    RefereeCommand.PREPARE_KICKOFF_YELLOW,
    RefereeCommand.PREPARE_KICKOFF_BLUE,
}

# §8.4.3: "Each foul has a grace period of 2 seconds per team until it is raised again."
_REGRACE_SECONDS = 2.0


class KeepOutRule(BaseRule):
    """Charges a foul to the defending team when one of its robots stays inside the
    keep-out radius of the ball during the other team's kick-off or free kick.

    Non-stopping: the rulebook's sanction is a foul-counter increment and a reset of
    the kicking team's timer, not a stop. The timer reset needs nothing here: the
    state machine only counts down a free kick or kick-off once every defender is
    clear (`_free_kick_ready`), so an encroaching defender already holds it back.
    Stopping play and re-awarding the free kick, as this rule used to, voided the
    restart -- in sim straight into a FORCE_START scramble.

    A violation is only issued after ``violation_persistence_frames`` consecutive
    frames of encroachment, preventing false positives from transient positions.
    """

    def __init__(
        self,
        radius_meters: float = 0.5,
        violation_persistence_frames: int = 30,
    ) -> None:
        self._radius = radius_meters
        self._persistence = violation_persistence_frames
        self._violation_count: int = 0
        self._last_raised_at: float = -math.inf

    def check(
        self,
        game_frame: GameFrame,
        geometry: RefereeGeometry,
        current_command: RefereeCommand,
        designated_position: Optional[tuple[float, float]] = None,
    ) -> Optional[RuleViolation]:
        if current_command not in _KICK_COMMANDS:
            self._violation_count = 0
            return None

        ball = game_frame.ball
        if ball is None:
            self._violation_count = 0
            return None

        bx, by = ball.p.x, ball.p.y

        kicking_team_is_yellow = _kicking_team_is_yellow(current_command)
        defenders = (
            game_frame.enemy_robots
            if kicking_team_is_yellow == game_frame.my_team_is_yellow
            else game_frame.friendly_robots
        )
        if self._any_robot_encroaching(defenders.values(), bx, by):
            self._violation_count += 1
        else:
            self._violation_count = 0

        if self._violation_count < self._persistence or game_frame.ts - self._last_raised_at < _REGRACE_SECONDS:
            return None
        self._violation_count = 0
        self._last_raised_at = game_frame.ts
        return RuleViolation(
            rule_name="keep_out",
            suggested_command=current_command,
            next_command=None,
            status_message="Defender too close to ball",
            offending_teams=(not kicking_team_is_yellow,),
            is_stopping=False,
        )

    def reset(self) -> None:
        self._violation_count = 0

    def reset_for_new_episode(self) -> None:
        self.reset()
        self._last_raised_at = -math.inf

    def _any_robot_encroaching(self, robots, bx: float, by: float) -> bool:
        return any(math.hypot(r.p.x - bx, r.p.y - by) < self._radius for r in robots)


def _kicking_team_is_yellow(command: RefereeCommand) -> bool:
    """Return True if the kicking team is yellow, False if blue."""
    return command in (RefereeCommand.DIRECT_FREE_YELLOW, RefereeCommand.PREPARE_KICKOFF_YELLOW)
