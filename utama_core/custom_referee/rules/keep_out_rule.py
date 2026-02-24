"""KeepOutRule: enforces minimum distance to the ball during stoppages."""

from __future__ import annotations

import math
from typing import Optional

from utama_core.custom_referee.geometry import RefereeGeometry
from utama_core.custom_referee.rules.base_rule import BaseRule, RuleViolation
from utama_core.entities.game.game_frame import GameFrame
from utama_core.entities.referee.referee_command import RefereeCommand

# Commands during which the keep-out circle must be respected.
_STOPPAGE_COMMANDS = {
    RefereeCommand.STOP,
    RefereeCommand.DIRECT_FREE_YELLOW,
    RefereeCommand.DIRECT_FREE_BLUE,
    RefereeCommand.PREPARE_KICKOFF_YELLOW,
    RefereeCommand.PREPARE_KICKOFF_BLUE,
    RefereeCommand.PREPARE_PENALTY_YELLOW,
    RefereeCommand.PREPARE_PENALTY_BLUE,
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
        my_team_is_yellow = game_frame.my_team_is_yellow

        # Determine which team is the *kicking* team (they are exempt).
        # During DIRECT_FREE_*, the kicking team is indicated by the command.
        kicking_team_is_yellow = _kicking_team_is_yellow(current_command, my_team_is_yellow)

        # Check non-kicking team robots for encroachment.
        encroaching = False
        if kicking_team_is_yellow is None:
            # STOP: both teams must stay back.
            encroaching = self._any_robot_encroaching(
                game_frame.friendly_robots.values(), bx, by
            ) or self._any_robot_encroaching(game_frame.enemy_robots.values(), bx, by)
        elif kicking_team_is_yellow == my_team_is_yellow:
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
            # Award free kick to the kicking team (or yellow if STOP).
            if kicking_team_is_yellow is None or kicking_team_is_yellow:
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


def _kicking_team_is_yellow(command: RefereeCommand, my_team_is_yellow: bool) -> Optional[bool]:
    """Return True if the kicking team is yellow, False if blue, None if STOP (both stop)."""
    if command in (
        RefereeCommand.DIRECT_FREE_YELLOW,
        RefereeCommand.PREPARE_KICKOFF_YELLOW,
        RefereeCommand.PREPARE_PENALTY_YELLOW,
    ):
        return True
    if command in (
        RefereeCommand.DIRECT_FREE_BLUE,
        RefereeCommand.PREPARE_KICKOFF_BLUE,
        RefereeCommand.PREPARE_PENALTY_BLUE,
    ):
        return False
    return None  # STOP — no kicking team
