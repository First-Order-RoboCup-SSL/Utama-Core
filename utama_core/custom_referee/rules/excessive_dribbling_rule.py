"""ExcessiveDribblingRule: enforces the 1m dribble-distance cap (SSL rulebook
§8.4.1 "Excessive Dribbling").

"A robot must not dribble the ball further than 1 meter, measured linearly
from the ball location where the dribbling started. A robot begins
dribbling when it makes contact with the ball and stops dribbling when
there is an observable separation between the ball and the robot. Dribblers
can still be used to dribble large distances with the ball as long as the
robot periodically loses possession, such as kicking the ball ahead of it
as human soccer players often do."

Uses `has_ball` deliberately (unlike Pushing/Crashing elsewhere in this
module) — dribbling is specifically about dribbler-mouth contact, which is
exactly what `has_ball` measures (real IR sensor for friendly robots,
sim-contact-derived for enemy).
"""

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


class ExcessiveDribblingRule(BaseRule):
    """Fires a free kick to the opposing team when a robot's continuous
    `has_ball` streak carries the ball more than `max_dribble_meters` (linear)
    from where that streak began.

    Per-robot dribble origin is captured on the has_ball rising edge and
    cleared the instant has_ball drops — "kicking the ball ahead of it" (a
    deliberate touch-release-touch pattern) legitimately resets the origin
    each time, per the rulebook's explicit carve-out, even if the robot's
    net displacement across several such touches exceeds 1m.
    """

    def __init__(self, max_dribble_meters: float = 1.0) -> None:
        self._max_dribble_meters = max_dribble_meters
        # robot_id -> (is_friendly, origin_x, origin_y) for the currently
        # open dribble streak, keyed separately per side since friendly and
        # enemy robots occupy independent id spaces.
        self._origin: dict[tuple[bool, int], tuple[float, float]] = {}

    def check(
        self,
        game_frame: GameFrame,
        geometry: RefereeGeometry,
        current_command: RefereeCommand,
        designated_position: Optional[tuple[float, float]] = None,
    ) -> Optional[RuleViolation]:
        if current_command not in _ACTIVE_PLAY_COMMANDS:
            self._origin = {}
            return None

        ball = game_frame.ball
        if ball is None:
            self._origin = {}
            return None

        bx, by = ball.p.x, ball.p.y
        my_team_is_yellow = game_frame.my_team_is_yellow

        violation = None
        live_keys: set[tuple[bool, int]] = set()
        for is_friendly, robots in ((True, game_frame.friendly_robots), (False, game_frame.enemy_robots)):
            for robot in robots.values():
                key = (is_friendly, robot.id)
                if not robot.has_ball:
                    continue
                live_keys.add(key)
                if key not in self._origin:
                    self._origin[key] = (bx, by)
                    continue
                ox, oy = self._origin[key]
                dist = math.hypot(bx - ox, by - oy)
                if dist > self._max_dribble_meters and violation is None:
                    robot_is_yellow = is_friendly == my_team_is_yellow
                    next_cmd = RefereeCommand.DIRECT_FREE_BLUE if robot_is_yellow else RefereeCommand.DIRECT_FREE_YELLOW
                    violation = RuleViolation(
                        rule_name="excessive_dribbling",
                        suggested_command=RefereeCommand.STOP,
                        next_command=next_cmd,
                        status_message=f"Excessive dribbling: {dist:.2f}m > {self._max_dribble_meters:.1f}m",
                        designated_position=(bx, by),
                        offending_teams=(robot_is_yellow,),
                    )
                    # Close out this streak so it doesn't refire every tick
                    # while the robot continues carrying the ball past the
                    # limit — the free kick + STOP already ends the streak.
                    del self._origin[key]
                    live_keys.discard(key)

        # Clear origins for any robot whose has_ball dropped this tick —
        # observable separation ends the dribble streak (rulebook: "stops
        # dribbling when there is an observable separation between the ball
        # and the robot"), and the *next* rising edge starts a fresh one.
        for key in list(self._origin):
            if key not in live_keys:
                del self._origin[key]

        return violation

    def reset(self) -> None:
        self._origin = {}
