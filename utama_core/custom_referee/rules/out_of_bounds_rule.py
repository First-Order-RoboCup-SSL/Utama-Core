"""OutOfBoundsRule: detects when the ball leaves the field."""

from __future__ import annotations

import math
from typing import Optional

from utama_core.config.referee_constants import OPPONENT_DEFENSE_AREA_KEEP_DISTANCE
from utama_core.custom_referee.geometry import CORNER_INFIELD_OFFSET, RefereeGeometry
from utama_core.custom_referee.rules.base_rule import BaseRule, RuleViolation
from utama_core.custom_referee.rules.last_touch import infer_last_touch_team
from utama_core.entities.game.game_frame import GameFrame
from utama_core.entities.referee.referee_command import RefereeCommand

_ACTIVE_PLAY_COMMANDS = {
    RefereeCommand.NORMAL_START,
    RefereeCommand.FORCE_START,
}

_INFIELD_OFFSET = 0.25  # metres inside the boundary for a playable free-kick placement
# A corner exit is close to two boundary lines at once, not one -- the single-edge
# offset above leaves only _INFIELD_OFFSET of clearance on EACH line simultaneously
# (as little as 0.08m observed live, see _nearest_infield_point's docstring), which
# is robot-body scale. Deeper offset used only when both axes are being clamped.
_CORNER_INFIELD_OFFSET = CORNER_INFIELD_OFFSET


class OutOfBoundsRule(BaseRule):
    """Fires a free kick for the non-touching team when the ball leaves the field."""

    def __init__(self) -> None:
        # Last team to have the ball: True = friendly, False = enemy, None = unknown.
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
            self._prev_ball_v = None  # the ball may be moved or placed while play is stopped
            return None

        ball = game_frame.ball
        if ball is None:
            return None

        bx, by = ball.p.x, ball.p.y

        # Update last-touch tracking regardless of out-of-bounds state.
        self._last_touch_was_friendly = infer_last_touch_team(
            game_frame, self._last_touch_was_friendly, self._prev_ball_v
        )
        self._prev_ball_v = (ball.v.x, ball.v.y)

        # Only fire when ball is outside field AND not in a goal.
        if geometry.is_in_field(bx, by) or geometry.is_in_left_goal(bx, by) or geometry.is_in_right_goal(bx, by):
            return None

        # Determine which team gets the free kick (non-touching team).
        free_kick_cmd = self._assign_free_kick(game_frame)
        # Over a goal line: a corner kick when the goal's own team touched it last.
        crossed_friendly_goal_line = (bx > 0) == game_frame.my_team_is_right
        corner_kick = self._last_touch_was_friendly == crossed_friendly_goal_line
        placement = self._nearest_infield_point(bx, by, geometry, corner_kick)

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
        self._prev_ball_v = None

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
    def _nearest_infield_point(
        bx: float, by: float, geometry: RefereeGeometry, corner_kick: bool = False
    ) -> tuple[float, float]:
        """Return the nearest point on the field boundary, offset inward, and
        clear of both defense areas.

        A ball over a goal line instead restarts in the corner nearer `by`, as
        §6.2.1–2 place goal and corner kicks: `_CORNER_INFIELD_OFFSET` from both
        lines for a corner kick (`corner_kick`), `RefereeGeometry.goal_kick_position`
        for a goal kick. Placing it where it crossed, pushed 1 m off
        the box, gave the attackers a free kick 2 m in front of goal: 521 of them
        and 150 goals in tournament_20261004_204810.

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

        The two per-axis clamps below are independent by construction (one
        only ever moves x, the other only ever moves y), which is fine when
        the ball exits near the middle of one edge -- but a ball exiting
        near a CORNER is close to both edges simultaneously, and each axis
        only insets itself `_INFIELD_OFFSET` from its OWN edge, ignorant of
        how close the other axis already sits to its own edge. The result
        can be as little as `_INFIELD_OFFSET` from both boundary lines at
        once (robot-body scale), not `_INFIELD_OFFSET` from the nearer one
        with headroom on the other. Found live, 2026-09-04 (four real
        tournament matches, e.g. `counter_flow_vs_counter_press`:
        58 out-of-bounds events in one 65s match, every restart placed
        0.08-0.41m from BOTH the goal line and the sideline at once): an
        ordinary post-restart drift (0.2-0.3 m/s, far below a shot) was
        enough to send the ball back out one of the two nearby lines,
        re-triggering the same restart in roughly the same corner,
        repeating for most or all of the match. Fixed the same way the
        defense-area gap above was: when a corner is detected (both axes
        clamped), inset BOTH axes by `_CORNER_INFIELD_OFFSET` (deeper than
        the single-edge `_INFIELD_OFFSET`, since a corner restart needs
        clearance on two sides at once, not one) instead of leaving
        whichever axis wasn't the "nearer" one at its raw clamped value.
        """
        near_x_boundary = abs(bx) > geometry.half_length
        if near_x_boundary:
            if not corner_kick:
                return geometry.goal_kick_position(math.copysign(1.0, bx), by)
            px = math.copysign(geometry.half_length - _CORNER_INFIELD_OFFSET, bx)
            py = math.copysign(geometry.half_width - _CORNER_INFIELD_OFFSET, by)
            return geometry.legal_restart_position(px, py, OPPONENT_DEFENSE_AREA_KEEP_DISTANCE)
        near_y_boundary = abs(by) > geometry.half_width
        # Corner detection must look at the RAW exit position on the axis
        # that wasn't clamped too, not just "was this axis itself out of
        # bounds" -- a ball can exit purely over the sideline (by > half_width)
        # while its x is still technically in-field but already close to the
        # goal line (e.g. bx=4.31 with half_length=4.5): near_x_boundary is
        # False there, yet the eventual x-clamp-free placement still sits
        # right next to that edge. Found live, 2026-09-04: this exact shape
        # was most of the traced counter_flow_vs_counter_press corner-loop
        # exits (sideline-only exits with x already within a few tenths of
        # the goal line). So "is this a corner" checks proximity to BOTH
        # edges using the raw (bx, by), regardless of which one triggered
        # the out-of-bounds call.
        near_corner = (geometry.half_length - abs(bx) < _CORNER_INFIELD_OFFSET) and (
            geometry.half_width - abs(by) < _CORNER_INFIELD_OFFSET
        )
        # A corner exit is close to both edges at once; give both axes the
        # deeper corner offset so the restart isn't left hugging one line
        # while only the other gets inset. A single-edge exit keeps the
        # shallower offset, unchanged from before.
        offset = _CORNER_INFIELD_OFFSET if near_corner else _INFIELD_OFFSET

        # Clamp to field bounds and shift inward.
        px = max(-geometry.half_length, min(geometry.half_length, bx))
        py = max(-geometry.half_width, min(geometry.half_width, by))

        # If a corner pushed the offset deeper than the raw in-field x already
        # sat from the goal line, offset inward along x.
        if near_corner and geometry.half_length - abs(bx) < offset:
            sign = 1.0 if bx > 0 else -1.0
            px = sign * (geometry.half_length - offset)

        # Mirror for y.
        if near_y_boundary or (near_corner and geometry.half_width - abs(by) < offset):
            sign = 1.0 if by > 0 else -1.0
            py = sign * (geometry.half_width - offset)

        return geometry.legal_restart_position(px, py, OPPONENT_DEFENSE_AREA_KEEP_DISTANCE)
