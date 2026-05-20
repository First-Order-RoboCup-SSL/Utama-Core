"""OutOfBoundsRule: detects when the ball leaves the field."""

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

_INFIELD_OFFSET = 0.1  # metres inside the boundary for free-kick placement


class OutOfBoundsRule(BaseRule):
    """Fires a free kick for the non-touching team when the ball leaves the field."""

    def __init__(self) -> None:
        # Track last robot to have the ball (friendly vs enemy) across frames.
        # True = friendly last touched, False = enemy last touched, None = unknown.
        self._last_touch_was_friendly: Optional[bool] = None

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

        bx, by = ball.p.x, ball.p.y

        # Update last-touch tracking regardless of out-of-bounds state.
        self._update_last_touch(game_frame, bx, by)

        # Only fire when ball is outside field AND not in a goal.
        if geometry.is_in_field(bx, by) or geometry.is_in_left_goal(bx, by) or geometry.is_in_right_goal(bx, by):
            return None

        # Determine which team gets the free kick (non-touching team).
        free_kick_cmd = self._assign_free_kick(game_frame)
        placement = self._nearest_infield_point(bx, by, geometry)

        return RuleViolation(
            rule_name="out_of_bounds",
            suggested_command=RefereeCommand.STOP,
            next_command=free_kick_cmd,
            status_message="Ball out of bounds",
            designated_position=placement,
        )

    def reset(self) -> None:
        self._last_touch_was_friendly = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_last_touch(self, game_frame: GameFrame, bx: float, by: float) -> None:
        """Update last-touch tracking based on robot proximity / has_ball flag."""
        # Check friendly robots first (has_ball from IR sensor is reliable).
        for robot in game_frame.friendly_robots.values():
            if robot.has_ball:
                self._last_touch_was_friendly = True
                return

        # Fall back to closest robot proximity.
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

        # Only update if a robot was actually close enough to plausibly touch (≤ 0.15 m).
        if closest_is_friendly is not None and min_dist <= 0.15:
            self._last_touch_was_friendly = closest_is_friendly

    def _assign_free_kick(self, game_frame: GameFrame) -> RefereeCommand:
        """Return the free-kick command for the non-touching team."""
        my_team_is_yellow = game_frame.my_team_is_yellow

        # Non-touching team gets the free kick.
        if self._last_touch_was_friendly is None:
            # Unknown last touch: give to yellow by default.
            return RefereeCommand.DIRECT_FREE_YELLOW

        if self._last_touch_was_friendly:
            # Friendly last touched → enemy gets free kick.
            if my_team_is_yellow:
                return RefereeCommand.DIRECT_FREE_BLUE
            else:
                return RefereeCommand.DIRECT_FREE_YELLOW
        else:
            # Enemy last touched → friendly gets free kick.
            if my_team_is_yellow:
                return RefereeCommand.DIRECT_FREE_YELLOW
            else:
                return RefereeCommand.DIRECT_FREE_BLUE

    @staticmethod
    def _nearest_infield_point(bx: float, by: float, geometry: RefereeGeometry) -> tuple[float, float]:
        """Return the nearest point on the field boundary, offset inward."""
        # Clamp to field bounds and shift inward.
        px = max(-geometry.half_length, min(geometry.half_length, bx))
        py = max(-geometry.half_width, min(geometry.half_width, by))

        # If clamped on x boundary, offset inward along x.
        if abs(bx) > geometry.half_length:
            sign = 1.0 if bx > 0 else -1.0
            px = sign * (geometry.half_length - _INFIELD_OFFSET)

        # If clamped on y boundary, offset inward along y.
        if abs(by) > geometry.half_width:
            sign = 1.0 if by > 0 else -1.0
            py = sign * (geometry.half_width - _INFIELD_OFFSET)

        return (px, py)
