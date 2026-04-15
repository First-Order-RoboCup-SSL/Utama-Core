"""KeepOutRule: enforces minimum distance to the ball during stoppages."""

from __future__ import annotations

import math
from typing import Optional  # used by BaseRule.check return type

from utama_core.custom_referee.geometry import RefereeGeometry
from utama_core.custom_referee.rules.base_rule import BaseRule, RuleViolation
from utama_core.entities.game.game_frame import GameFrame
from utama_core.entities.referee.referee_command import RefereeCommand

# Commands during which the keep-out circle must be respected.
# STOP is intentionally excluded: during STOP robots are actively clearing the
# ball and the state machine already handles this via StopStep.  Firing a
# keep-out violation during STOP would overwrite next_command (e.g. replacing
# PREPARE_KICKOFF_BLUE with DIRECT_FREE_YELLOW), which is wrong.
# PREPARE_KICKOFF_* and PREPARE_PENALTY_* are excluded for the same reason:
# robots are actively moving to their formation positions during these states.
# The state machine gates progression via _kicker_in_centre_circle /
# _penalty_kicker_ready, so keep-out violations here only cause unnecessary
# sequence resets.
_STOPPAGE_COMMANDS = {
    RefereeCommand.DIRECT_FREE_YELLOW,
    RefereeCommand.DIRECT_FREE_BLUE,
}


class KeepOutRule(BaseRule):
    """Penalises robots that remain inside the keep-out radius around the ball.

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

    def check(
        self,
        game_frame: GameFrame,
        geometry: RefereeGeometry,
        current_command: RefereeCommand,
    ) -> Optional[RuleViolation]:
        if current_command not in _STOPPAGE_COMMANDS:
            self._violation_count = 0
            return None

        ball = game_frame.ball
        if ball is None:
            self._violation_count = 0
            return None

        bx, by = ball.p.x, ball.p.y

        # Determine which team is the *kicking* team (they are exempt).
        # current_command is always in _STOPPAGE_COMMANDS here, so
        # kicking_team_is_yellow is always True or False (never None).
        kicking_team_is_yellow = _kicking_team_is_yellow(current_command)

        # Check non-kicking team robots for encroachment.
        if kicking_team_is_yellow == game_frame.my_team_is_yellow:
            # Friendly is kicking — check enemy only.
            encroaching = self._any_robot_encroaching(game_frame.enemy_robots.values(), bx, by)
        else:
            # Enemy is kicking — check friendly only.
            encroaching = self._any_robot_encroaching(game_frame.friendly_robots.values(), bx, by)
        if encroaching:
            self._violation_count += 1
        else:
            self._violation_count = 0

        if self._violation_count >= self._persistence:
            self._violation_count = 0  # Reset after issuing.
            # Award free kick to the kicking team (they get to retry).
            if kicking_team_is_yellow:
                next_cmd = RefereeCommand.DIRECT_FREE_YELLOW
            else:
                next_cmd = RefereeCommand.DIRECT_FREE_BLUE
            return RuleViolation(
                rule_name="keep_out",
                suggested_command=RefereeCommand.STOP,
                next_command=next_cmd,
                status_message="Keep-out circle violation",
            )

        return None

    def reset(self) -> None:
        self._violation_count = 0

    def _any_robot_encroaching(self, robots, bx: float, by: float) -> bool:
        return any(math.hypot(r.p.x - bx, r.p.y - by) < self._radius for r in robots)


def _kicking_team_is_yellow(command: RefereeCommand) -> bool:
    """Return True if the kicking team is yellow, False if blue.

    Only called when command is in _STOPPAGE_COMMANDS (DIRECT_FREE_* only).
    """
    return command == RefereeCommand.DIRECT_FREE_YELLOW
