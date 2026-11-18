"""BallSpeedRule: enforces SSL's maximum kick-speed limit."""

from __future__ import annotations

import math
from typing import Optional

from utama_core.entities.game.game_frame import GameFrame
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.referee.geometry import RefereeGeometry
from utama_core.referee.rules.base_rule import BaseRule, RuleViolation
from utama_core.referee.rules.last_touch import infer_last_touch_team

# Frames in a row over the limit before it counts. 67-91% of the calls in the round-robins of
# 2026-11-09 were one frame, up to 39 m/s, with the robot's kick at a median 5.3 m/s: vision noise
# in the ball's velocity, not a kick.
_FRAMES_OVER_LIMIT = 2

_ACTIVE_PLAY_COMMANDS = {
    RefereeCommand.NORMAL_START,
    RefereeCommand.FORCE_START,
}


class BallSpeedRule(BaseRule):
    """Penalises the last team to touch the ball if it exceeds the ground-speed
    limit (SSL Division B: 6.5 m/s).

    Only the ball's ground speed (x/y) counts — a bounce's z-velocity is not
    part of the kick-speed rule and would otherwise cause false positives.

    Fires once when the ball has been over the limit for `_FRAMES_OVER_LIMIT`
    frames in a row, not every frame it remains fast, mirroring how a real
    kick is a single event rather than a sustained state.
    """

    def __init__(self, max_speed_mps: float = 6.5) -> None:
        self._max_speed = max_speed_mps
        self._frames_over_limit = 0
        # True = friendly last touched, False = enemy, None = unknown.
        # Maintained colour-blind by `infer_last_touch_team` (see last_touch.py).
        self._last_touch_was_friendly: Optional[bool] = None
        # The ball's velocity on the previous active-play frame (see `infer_last_touch_team`).
        self._prev_ball_v: Optional[tuple[float, float]] = None

    def check(
        self,
        game_frame: GameFrame,
        geometry: RefereeGeometry,
        current_command: RefereeCommand,
        designated_position: Optional[tuple[float, float]] = None,
    ) -> Optional[RuleViolation]:
        if current_command not in _ACTIVE_PLAY_COMMANDS:
            self._frames_over_limit = 0
            self._prev_ball_v = None  # the ball may be moved or placed while play is stopped
            return None

        ball = game_frame.ball
        if ball is None:
            return None

        self._last_touch_was_friendly = infer_last_touch_team(
            game_frame, self._last_touch_was_friendly, self._prev_ball_v
        )
        self._prev_ball_v = (ball.v.x, ball.v.y)

        speed = math.hypot(ball.v.x, ball.v.y)
        self._frames_over_limit = self._frames_over_limit + 1 if speed > self._max_speed else 0

        if self._frames_over_limit != _FRAMES_OVER_LIMIT or self._last_touch_was_friendly is None:
            return None

        # SSL rulebook §8.4.2 "Ball Speed" is a non-stopping foul: the kicking
        # team is charged and the game continues normally. This rule used to
        # stop play with a free kick to the other team.
        my_team_is_yellow = game_frame.my_team_is_yellow
        kicking_team_is_yellow = self._last_touch_was_friendly == my_team_is_yellow
        return RuleViolation(
            rule_name="ball_speed",
            suggested_command=current_command,
            next_command=None,
            status_message=f"Ball speed exceeded {self._max_speed:.1f} m/s",
            offending_teams=(kicking_team_is_yellow,),
            is_stopping=False,
        )

    def reset(self) -> None:
        self._frames_over_limit = 0
        self._last_touch_was_friendly = None
