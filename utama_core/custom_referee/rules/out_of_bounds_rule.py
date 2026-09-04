"""OutOfBoundsRule: detects when the ball leaves the field."""

from __future__ import annotations

from typing import Optional

from utama_core.config.referee_constants import OPPONENT_DEFENSE_AREA_KEEP_DISTANCE
from utama_core.custom_referee.geometry import RefereeGeometry
from utama_core.custom_referee.rules.base_rule import BaseRule, RuleViolation
from utama_core.custom_referee.rules.last_touch import infer_last_touch_team
from utama_core.entities.game.game_frame import GameFrame
from utama_core.entities.referee.referee_command import RefereeCommand

_ACTIVE_PLAY_COMMANDS = {
    RefereeCommand.NORMAL_START,
    RefereeCommand.FORCE_START,
}

_INFIELD_OFFSET = 0.25  # metres inside the boundary for a playable free-kick placement


class OutOfBoundsRule(BaseRule):
    """Fires a free kick for the non-touching team when the ball leaves the field."""

    def __init__(self) -> None:
        # Last team to have the ball: True = friendly, False = enemy, None = unknown.
        # Maintained colour-blind by `infer_last_touch_team` (see last_touch.py).
        self._last_touch_was_friendly: Optional[bool] = None

    def check(
        self,
        game_frame: GameFrame,
        geometry: RefereeGeometry,
        current_command: RefereeCommand,
        designated_position: Optional[tuple[float, float]] = None,
    ) -> Optional[RuleViolation]:
        if current_command not in _ACTIVE_PLAY_COMMANDS:
            return None

        ball = game_frame.ball
        if ball is None:
            return None

        bx, by = ball.p.x, ball.p.y

        # Update last-touch tracking regardless of out-of-bounds state.
        self._last_touch_was_friendly = infer_last_touch_team(game_frame, self._last_touch_was_friendly)

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
            status_message=(
                "Ball out of bounds" if free_kick_cmd is not None else "Ball out of bounds (last touch unknown)"
            ),
            designated_position=placement,
        )

    def reset(self) -> None:
        self._last_touch_was_friendly = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _assign_free_kick(self, game_frame: GameFrame) -> Optional[RefereeCommand]:
        """Return the free-kick command for the non-touching team.

        Returns None when the last touch cannot be attributed at all
        (only possible with an empty frame — a real match always has
        robots, so `infer_last_touch_team` resolves the touch).
        """
        my_team_is_yellow = game_frame.my_team_is_yellow

        if self._last_touch_was_friendly is None:
            # No colour-bias default: leave the restart unresolved instead
            # of awarding the ball to a hardcoded team.
            return None

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
        """Return the nearest point on the field boundary, offset inward, and
        clear of both defense areas.

        The boundary offset alone (`_INFIELD_OFFSET` = 0.25m) is shallower
        than a defense area's depth (`half_defense_depth`, 0.5m on the
        standard field) -- a ball going out near either goal line routinely
        projects to a point still inside that defense area. Found live,
        tiki_taka_plus_vs_zone_fluid (2026-09-04): out-of-bounds near the
        left goal line placed a `DIRECT_FREE_YELLOW` at (-4.25, 0.58),
        squarely inside the left defense area, which immediately fired
        "Yellow too close to opponent defense area"/"Yellow attacker in
        blue defense area" and churned into a second stoppage. Same failure
        mode `RefereeGeometry.legal_restart_position`'s docstring documents
        for every other rule that derives a restart position from the
        ball's raw position -- this rule's own boundary projection was
        wrongly assumed exempt (see that docstring's example), since it
        clamps the field boundary but never the defense-area one. Run the
        boundary-clamped point through the same shared projection every
        other rule already uses.
        """
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

        return geometry.legal_restart_position(px, py, OPPONENT_DEFENSE_AREA_KEEP_DISTANCE)
