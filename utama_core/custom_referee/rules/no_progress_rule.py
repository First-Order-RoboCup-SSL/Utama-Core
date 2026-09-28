"""NoProgressRule: SSL rulebook §8.1 "No Progress In Game".

"If there is no progress in the game for 5 seconds (Division A) or 10 seconds
(Division B) while both teams are allowed to manipulate the ball, the game is
stopped and continued by a forced start."

"Progress" is taken as the ball moving: the clock restarts whenever the ball has
moved `_PROGRESS_DISTANCE_M` from where it was when the clock last restarted --
the same 0.05 m the rulebook uses for a ball coming into play (§5.4).
"""

from __future__ import annotations

import math
from typing import Optional

from utama_core.config.referee_constants import OPPONENT_DEFENSE_AREA_KEEP_DISTANCE
from utama_core.custom_referee.geometry import RefereeGeometry
from utama_core.custom_referee.rules.base_rule import BaseRule, RuleViolation
from utama_core.entities.game.game_frame import GameFrame
from utama_core.entities.referee.referee_command import RefereeCommand

_ACTIVE_PLAY_COMMANDS = {
    RefereeCommand.NORMAL_START,
    RefereeCommand.FORCE_START,
}

_PROGRESS_DISTANCE_M = 0.05


class NoProgressRule(BaseRule):
    """Stops the game with a forced start after `max_seconds` of live play in which
    the ball has not moved `_PROGRESS_DISTANCE_M`. Nobody is at fault: no foul is
    charged. The ball is placed where it lies, projected clear of the defense areas
    like every other restart position."""

    def __init__(self, max_seconds: float = 10.0) -> None:
        self._max_seconds = max_seconds
        self._anchor: Optional[tuple[float, float]] = None
        self._anchor_ts: float = 0.0

    def check(
        self,
        game_frame: GameFrame,
        geometry: RefereeGeometry,
        current_command: RefereeCommand,
        designated_position: Optional[tuple[float, float]] = None,
    ) -> Optional[RuleViolation]:
        ball = game_frame.ball
        if current_command not in _ACTIVE_PLAY_COMMANDS or ball is None:
            self._anchor = None
            return None

        here = (ball.p.x, ball.p.y)
        if self._anchor is None or math.dist(here, self._anchor) >= _PROGRESS_DISTANCE_M:
            self._anchor = here
            self._anchor_ts = game_frame.ts
            return None
        if game_frame.ts - self._anchor_ts < self._max_seconds:
            return None

        self._anchor = None
        return RuleViolation(
            rule_name="no_progress",
            suggested_command=RefereeCommand.STOP,
            next_command=RefereeCommand.FORCE_START,
            status_message=f"No progress in game for {self._max_seconds:.0f} s",
            designated_position=geometry.legal_restart_position(*here, OPPONENT_DEFENSE_AREA_KEEP_DISTANCE),
        )

    def reset(self) -> None:
        self._anchor = None
