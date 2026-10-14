"""PenaltyTimeLimitRule: SSL rulebook §5.3.5 "Penalty Kick".

"If the ball is still in play after 10 seconds, the game is stopped." A goal is
then not awarded and "the game is continued by a goal kick for the defending
team" (§6.2.1), placed by `RefereeGeometry.goal_kick_position` like every other
goal kick.
"""

from __future__ import annotations

from typing import Optional

from utama_core.custom_referee.geometry import RefereeGeometry
from utama_core.custom_referee.rules.base_rule import BaseRule, RuleViolation
from utama_core.entities.game.game_frame import GameFrame
from utama_core.entities.referee.referee_command import RefereeCommand

_ACTIVE_PLAY_COMMANDS = {
    RefereeCommand.NORMAL_START,
    RefereeCommand.FORCE_START,
}


class PenaltyTimeLimitRule(BaseRule):
    """Stops a penalty still in play `max_seconds` after its NORMAL_START and
    gives the defending team a goal kick. A goal, or the ball leaving the field,
    stops play first and ends the penalty."""

    def __init__(self, max_seconds: float = 10.0) -> None:
        self._max_seconds = max_seconds
        self._prev_command: Optional[RefereeCommand] = None
        # Set while a penalty is being taken: (kicking team is yellow, start time).
        self._penalty: Optional[tuple[bool, float]] = None

    def check(
        self,
        game_frame: GameFrame,
        geometry: RefereeGeometry,
        current_command: RefereeCommand,
        designated_position: Optional[tuple[float, float]] = None,
    ) -> Optional[RuleViolation]:
        if current_command == RefereeCommand.NORMAL_START and self._prev_command in (
            RefereeCommand.PREPARE_PENALTY_YELLOW,
            RefereeCommand.PREPARE_PENALTY_BLUE,
        ):
            self._penalty = (self._prev_command == RefereeCommand.PREPARE_PENALTY_YELLOW, game_frame.ts)
        elif current_command not in _ACTIVE_PLAY_COMMANDS:
            self._penalty = None
        self._prev_command = current_command

        if self._penalty is None:
            return None
        kicking_is_yellow, start_ts = self._penalty
        if game_frame.ts - start_ts < self._max_seconds:
            return None

        self._penalty = None
        return RuleViolation(
            rule_name="penalty_time_limit",
            suggested_command=RefereeCommand.STOP,
            next_command=RefereeCommand.DIRECT_FREE_BLUE if kicking_is_yellow else RefereeCommand.DIRECT_FREE_YELLOW,
            status_message=f"Penalty not finished in {self._max_seconds:.0f} s",
            designated_position=self._goal_kick_position(game_frame, geometry, kicking_is_yellow),
        )

    @staticmethod
    def _goal_kick_position(
        game_frame: GameFrame, geometry: RefereeGeometry, kicking_is_yellow: bool
    ) -> tuple[float, float]:
        # The defending goal is the one the kicking team attacks.
        kicking_is_right = game_frame.my_team_is_right == (game_frame.my_team_is_yellow == kicking_is_yellow)
        ball_y = game_frame.ball.p.y if game_frame.ball is not None else 0.0
        return geometry.goal_kick_position(-1.0 if kicking_is_right else 1.0, ball_y)

    def reset(self) -> None:
        # Keeps _prev_command and the running penalty: reset() fires on every
        # command change, including the PREPARE_PENALTY -> NORMAL_START edge
        # this rule arms on and a NORMAL_START -> FORCE_START auto-advance
        # mid-penalty.
        pass

    def reset_for_new_episode(self) -> None:
        self._prev_command = None
        self._penalty = None
