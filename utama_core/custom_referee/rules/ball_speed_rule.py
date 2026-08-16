"""BallSpeedRule: enforces SSL's maximum kick-speed limit."""

from __future__ import annotations

import math
from typing import Optional

from utama_core.custom_referee.geometry import RefereeGeometry
from utama_core.custom_referee.rules.base_rule import BaseRule, RuleViolation
from utama_core.entities.game.game_frame import GameFrame
from utama_core.entities.referee.referee_command import RefereeCommand

_ACTIVE_PLAY_COMMANDS = {
    RefereeCommand.NORMAL_START,
    RefereeCommand.FORCE_START,
}

# Touches closer than this are trusted enough to attribute a kick to a robot.
_TOUCH_DIST = 0.15  # metres, matches OutOfBoundsRule's fallback threshold


class BallSpeedRule(BaseRule):
    """Penalises the last team to touch the ball if it exceeds the ground-speed
    limit (SSL Division B: 6.5 m/s).

    Only the ball's ground speed (x/y) counts — a bounce's z-velocity is not
    part of the kick-speed rule and would otherwise cause false positives.

    Edge-detected: fires once when the ball crosses above the limit, not
    every frame it remains fast, mirroring how a real kick is a single
    event rather than a sustained state.
    """

    def __init__(self, max_speed_mps: float = 6.5) -> None:
        self._max_speed = max_speed_mps
        self._was_over_limit = False
        # True = friendly last touched, False = enemy, None = unknown.
        self._last_touch_was_friendly: Optional[bool] = None

    def check(
        self,
        game_frame: GameFrame,
        geometry: RefereeGeometry,
        current_command: RefereeCommand,
    ) -> Optional[RuleViolation]:
        if current_command not in _ACTIVE_PLAY_COMMANDS:
            self._was_over_limit = False
            return None

        ball = game_frame.ball
        if ball is None:
            return None

        self._update_last_touch(game_frame, ball.p.x, ball.p.y)

        speed = math.hypot(ball.v.x, ball.v.y)
        is_over_limit = speed > self._max_speed
        rising_edge = is_over_limit and not self._was_over_limit
        self._was_over_limit = is_over_limit

        if not rising_edge or self._last_touch_was_friendly is None:
            return None

        # Non-kicking team gets the free kick.
        my_team_is_yellow = game_frame.my_team_is_yellow
        if self._last_touch_was_friendly:
            next_cmd = RefereeCommand.DIRECT_FREE_BLUE if my_team_is_yellow else RefereeCommand.DIRECT_FREE_YELLOW
        else:
            next_cmd = RefereeCommand.DIRECT_FREE_YELLOW if my_team_is_yellow else RefereeCommand.DIRECT_FREE_BLUE

        return RuleViolation(
            rule_name="ball_speed",
            suggested_command=RefereeCommand.STOP,
            next_command=next_cmd,
            status_message=f"Ball speed exceeded {self._max_speed:.1f} m/s",
        )

    def reset(self) -> None:
        self._was_over_limit = False
        self._last_touch_was_friendly = None

    def _update_last_touch(self, game_frame: GameFrame, bx: float, by: float) -> None:
        """Same approach as `OutOfBoundsRule`: prefer the IR-backed `has_ball`
        flag on friendly robots, fall back to closest-robot-within-touch-dist
        for enemies (whose `has_ball` is itself a positional heuristic, not a
        real sensor reading — see `entities/game/robot.py`)."""
        for robot in game_frame.friendly_robots.values():
            if robot.has_ball:
                self._last_touch_was_friendly = True
                return

        min_dist = math.inf
        closest_is_friendly: Optional[bool] = None

        for robot in game_frame.friendly_robots.values():
            d = math.hypot(robot.p.x - bx, robot.p.y - by)
            if d < min_dist:
                min_dist = d
                closest_is_friendly = True

        for robot in game_frame.enemy_robots.values():
            d = math.hypot(robot.p.x - bx, robot.p.y - by)
            if d < min_dist:
                min_dist = d
                closest_is_friendly = False

        if closest_is_friendly is not None and min_dist <= _TOUCH_DIST:
            self._last_touch_was_friendly = closest_is_friendly
