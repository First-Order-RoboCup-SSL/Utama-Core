"""RobotStopSpeedRule: enforces the max robot speed during STOP (SSL rulebook
§8.4.3 "Robot Stop Speed").

"A robot must not move faster than 1.5 meters per second during stop. A
violation of this rule is only counted once per robot and stoppage. There
is a grace period of 2 seconds for the robots to slow down. This rule does
not apply to ball placement."

Only `RefereeCommand.STOP` triggers this rule — `BALL_PLACEMENT_*` is a
separate command in this codebase's enum and is never STOP, so the
rulebook's placement exemption holds automatically without an explicit
check; robots are also naturally allowed to move fast again once any other
command (PREPARE_*, DIRECT_FREE_*, NORMAL_START, ...) takes over.

A robot still outside `BALL_KEEP_OUT_DISTANCE` from the ball is exempt from
the speed check regardless of the grace clock: `RefereeOverride`'s
`_clear_to_legal_positions` (actions.py) actively drives any robot caught
inside that radius at STOP-entry back out at full motion-controller speed,
so a robot that entered STOP deep inside the keep-out zone can still be
legitimately mid-clear past the 2s grace mark. Fouling it for that would
penalize the robot for complying with the referee's own override — the
rule should only fire once a robot has reached (or already was at) a legal
distance from the ball and *then* moves too fast.
"""

from __future__ import annotations

import math
from typing import Optional

from utama_core.config.referee_constants import BALL_KEEP_OUT_DISTANCE
from utama_core.custom_referee.geometry import RefereeGeometry
from utama_core.custom_referee.rules.base_rule import BaseRule, RuleViolation
from utama_core.entities.game.game_frame import GameFrame
from utama_core.entities.referee.referee_command import RefereeCommand


class RobotStopSpeedRule(BaseRule):
    """Fouls a robot that exceeds `max_speed_mps` during STOP, once per robot
    per stoppage, after a `grace_seconds` grace period from STOP's start.

    Reported with `RuleViolation.is_stopping=False`: there is no "resume
    with a free kick" language for this rule (unlike Pushing or Multiple
    Defenders) — the game is already in STOP, so the only effect should be
    the foul-counter/yellow-card side effect `_handle_foul` applies before
    checking `is_stopping`. `command`/`next_command` must stay untouched, or
    the state machine would read a spurious transition out of STOP.
    """

    _GRACE_SECONDS = 2.0

    def __init__(self, max_speed_mps: float = 1.5, grace_seconds: float = _GRACE_SECONDS) -> None:
        self._max_speed = max_speed_mps
        self._grace_seconds = grace_seconds
        self._stop_entered_at: Optional[float] = None
        # Robot ids (friendly/enemy, keyed like ExcessiveDribblingRule) already
        # charged for this stoppage — at most one foul per robot per STOP.
        self._already_charged: set[tuple[bool, int]] = set()

    def check(
        self,
        game_frame: GameFrame,
        geometry: RefereeGeometry,
        current_command: RefereeCommand,
        designated_position: Optional[tuple[float, float]] = None,
    ) -> Optional[RuleViolation]:
        if current_command != RefereeCommand.STOP:
            self._stop_entered_at = None
            self._already_charged = set()
            return None

        if self._stop_entered_at is None:
            self._stop_entered_at = game_frame.ts

        if game_frame.ts - self._stop_entered_at < self._grace_seconds:
            return None

        ball = game_frame.ball
        my_team_is_yellow = game_frame.my_team_is_yellow
        for is_friendly, robots in ((True, game_frame.friendly_robots), (False, game_frame.enemy_robots)):
            for robot in robots.values():
                key = (is_friendly, robot.id)
                if key in self._already_charged:
                    continue
                if ball is not None and math.hypot(robot.p.x - ball.p.x, robot.p.y - ball.p.y) < BALL_KEEP_OUT_DISTANCE:
                    # Still inside the keep-out zone: may still be being
                    # actively driven out by RefereeOverride, not evidence of
                    # non-compliance.
                    continue
                speed = math.hypot(robot.v.x, robot.v.y)
                if speed <= self._max_speed:
                    continue
                self._already_charged.add(key)
                robot_is_yellow = is_friendly == my_team_is_yellow
                return RuleViolation(
                    rule_name="robot_stop_speed",
                    suggested_command=current_command,
                    next_command=None,
                    status_message=f"Robot exceeded {self._max_speed:.1f} m/s during STOP",
                    offending_teams=(robot_is_yellow,),
                    is_stopping=False,
                )

        return None

    def reset(self) -> None:
        self._stop_entered_at = None
        self._already_charged = set()
