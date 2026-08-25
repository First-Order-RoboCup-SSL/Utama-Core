"""KeeperHeldBallRule: enforces the max-hold-time limit for a ball resting in
a team's own defense area (SSL rulebook §8.4.1 "Keeper Held Ball").

"The ball must not be kept in the defense area for more than 5 seconds
(Division A) or 10 seconds (Division B)." Despite the name, the rulebook
condition is purely geometric (ball inside the defending team's own defense
area), not "the keeper is holding it" — a stray ball sitting untouched in
the box for the full duration fouls just the same as one the keeper is
actively controlling. This rule tracks continuous dwell time only; it does
not consult `has_ball` at all.
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


class KeeperHeldBallRule(BaseRule):
    """Fires a free kick to the attacking team once the ball has continuously
    dwelt inside the defending team's own defense area for longer than
    ``max_hold_seconds``.

    Dwell time is measured via ``game_frame.ts`` deltas (not a wall-clock/tick
    count) so it stays correct under variable tick rates. The clock resets the
    moment the ball leaves the defense area, or when play stops being active
    (``current_command`` outside ``_ACTIVE_PLAY_COMMANDS``) — a stoppage that
    happens to catch the ball mid-hold isn't a continuation of the same hold
    once play resumes.
    """

    def __init__(self, max_hold_seconds: float = 10.0) -> None:
        self._max_hold_seconds = max_hold_seconds
        # Timestamp the ball most recently *entered* either defense area,
        # per side (True = yellow's own area, False = blue's own area).
        # None means "not currently inside that area".
        self._entered_at: dict[bool, Optional[float]] = {True: None, False: None}

    def check(
        self,
        game_frame: GameFrame,
        geometry: RefereeGeometry,
        current_command: RefereeCommand,
        designated_position: Optional[tuple[float, float]] = None,
    ) -> Optional[RuleViolation]:
        if current_command not in _ACTIVE_PLAY_COMMANDS:
            self._entered_at = {True: None, False: None}
            return None

        ball = game_frame.ball
        if ball is None:
            self._entered_at = {True: None, False: None}
            return None

        bx, by = ball.p.x, ball.p.y

        # yellow_is_right: which physical side (left/right defense area)
        # belongs to yellow, from this frame's perspective — same idiom as
        # DefenseAreaRule.
        yellow_is_right = game_frame.my_team_is_right == game_frame.my_team_is_yellow
        in_yellow_own_area = (
            geometry.is_in_right_defense_area(bx, by) if yellow_is_right else geometry.is_in_left_defense_area(bx, by)
        )
        in_blue_own_area = (
            geometry.is_in_left_defense_area(bx, by) if yellow_is_right else geometry.is_in_right_defense_area(bx, by)
        )

        violation = None
        for is_yellow_area, currently_in in ((True, in_yellow_own_area), (False, in_blue_own_area)):
            if not currently_in:
                self._entered_at[is_yellow_area] = None
                continue
            if self._entered_at[is_yellow_area] is None:
                self._entered_at[is_yellow_area] = game_frame.ts
            held_for = game_frame.ts - self._entered_at[is_yellow_area]
            if held_for > self._max_hold_seconds and violation is None:
                # Held team is charged the foul; free kick to the other team,
                # taken from the ball's current position.
                next_cmd = RefereeCommand.DIRECT_FREE_BLUE if is_yellow_area else RefereeCommand.DIRECT_FREE_YELLOW
                violation = RuleViolation(
                    rule_name="keeper_held_ball",
                    suggested_command=RefereeCommand.STOP,
                    next_command=next_cmd,
                    status_message=(
                        f"Ball held in {'yellow' if is_yellow_area else 'blue'} defense area "
                        f"over {self._max_hold_seconds:.0f}s"
                    ),
                    designated_position=(bx, by),
                    offending_teams=(is_yellow_area,),
                )
                self._entered_at[is_yellow_area] = None  # reset after issuing

        return violation

    def reset(self) -> None:
        self._entered_at = {True: None, False: None}
