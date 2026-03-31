"""GoalRule: detects when the ball crosses a goal line."""

from __future__ import annotations

import math
from typing import Optional

from utama_core.custom_referee.geometry import RefereeGeometry
from utama_core.custom_referee.rules.base_rule import BaseRule, RuleViolation
from utama_core.entities.game.game_frame import GameFrame
from utama_core.entities.referee.referee_command import RefereeCommand

# Commands that represent active play — goal detection is only relevant here.
_ACTIVE_PLAY_COMMANDS = {
    RefereeCommand.NORMAL_START,
    RefereeCommand.FORCE_START,
}


class GoalRule(BaseRule):
    """Detects goals with a cooldown to suppress duplicate detections."""

    def __init__(self, cooldown_seconds: float = 1.0) -> None:
        self._cooldown = cooldown_seconds
        self._last_goal_time: float = -math.inf

    def check(
        self,
        game_frame: GameFrame,
        geometry: RefereeGeometry,
        current_command: RefereeCommand,
    ) -> Optional[RuleViolation]:
        if current_command not in _ACTIVE_PLAY_COMMANDS:
            return None

        ball = game_frame.ball
        if ball is None:
            return None

        current_time = game_frame.ts

        # Respect cooldown — prevents the same goal being reported for multiple frames.
        if current_time - self._last_goal_time < self._cooldown:
            return None

        bx, by = ball.p.x, ball.p.y
        # Determine which colour team defends each goal from the frame's perspective.
        # my_team_is_right=True  → yellow defends right goal, blue defends left goal.
        # my_team_is_right=False → blue defends right goal, yellow defends left goal.
        yellow_is_right = game_frame.my_team_is_right == game_frame.my_team_is_yellow

        # Right goal: the team defending the right side conceded.
        if geometry.is_in_right_goal(bx, by):
            self._last_goal_time = current_time
            if yellow_is_right:
                # Yellow conceded → Blue scored → Yellow kicks off
                return RuleViolation(
                    rule_name="goal",
                    suggested_command=RefereeCommand.STOP,
                    next_command=RefereeCommand.PREPARE_KICKOFF_YELLOW,
                    status_message="Goal by Blue",
                )
            else:
                # Blue conceded → Yellow scored → Blue kicks off
                return RuleViolation(
                    rule_name="goal",
                    suggested_command=RefereeCommand.STOP,
                    next_command=RefereeCommand.PREPARE_KICKOFF_BLUE,
                    status_message="Goal by Yellow",
                )

        # Left goal: the team defending the left side conceded.
        if geometry.is_in_left_goal(bx, by):
            self._last_goal_time = current_time
            if yellow_is_right:
                # Blue conceded → Yellow scored → Blue kicks off
                return RuleViolation(
                    rule_name="goal",
                    suggested_command=RefereeCommand.STOP,
                    next_command=RefereeCommand.PREPARE_KICKOFF_BLUE,
                    status_message="Goal by Yellow",
                )
            else:
                # Yellow conceded → Blue scored → Yellow kicks off
                return RuleViolation(
                    rule_name="goal",
                    suggested_command=RefereeCommand.STOP,
                    next_command=RefereeCommand.PREPARE_KICKOFF_YELLOW,
                    status_message="Goal by Blue",
                )

        return None

    def reset(self) -> None:
        # Keep last_goal_time across resets so cooldown still applies.
        pass
