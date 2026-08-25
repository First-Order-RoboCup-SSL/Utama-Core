"""DefenseAreaStoppageRule: enforces standoff from the OPPONENT defense area
during stoppages, before the ball has entered play (SSL rulebook §8.4.1,
"Robot Too Close To Opponent Defense Area").

Distinct from `DefenseAreaRule`, which only checks *active play*
(NORMAL_START/FORCE_START) and enforces a different pair of rules
(too-many-defenders occupancy, attacker-in-own-area encroachment). This
rule instead covers STOP and free-kick-pending commands, before
NORMAL_START has actually been reached for that restart.

"During stop and free kicks, before the ball has entered play, all robots
have to keep at least 0.2 meters distance to the opponent defense area.
There is a grace period of 2 seconds for the robots to move away from the
opponent defense area. The game is immediately halted after the second
such foul committed by the same team while the game is stopped or during a
free kick, before the ball has entered play. If the first foul is
committed during a free kick, the game is still stopped regularly. The
grace period is restarted after the first foul of the same team. Both
fouls count towards the foul counter. There are no individual fouls per
robot."
"""

from __future__ import annotations

from typing import Optional

from utama_core.custom_referee.geometry import RefereeGeometry
from utama_core.custom_referee.rules.base_rule import BaseRule, RuleViolation
from utama_core.entities.game.game_frame import GameFrame
from utama_core.entities.referee.referee_command import RefereeCommand

# "Before the ball has entered play" — STOP itself, plus a free kick that
# hasn't yet transitioned to NORMAL_START. Kickoff/penalty prepare commands
# are deliberately excluded: robots are still moving into formation there
# and haven't been given a live ball to chase into the box yet, and the
# rulebook's own wording ("free kicks") doesn't name them.
_STOPPAGE_COMMANDS = {
    RefereeCommand.STOP,
    RefereeCommand.DIRECT_FREE_YELLOW,
    RefereeCommand.DIRECT_FREE_BLUE,
}

_MIN_DISTANCE = 0.2  # metres
_GRACE_SECONDS = 2.0


class DefenseAreaStoppageRule(BaseRule):
    """Team-level (not per-robot) encroachment on the OPPONENT defense area
    during STOP/free-kick-pending. First offense per team: STOP + regular
    restart continues (rulebook: "the game is still stopped regularly").
    Second offense by the *same* team while still in this stoppage window:
    HALT immediately.
    """

    def __init__(self, min_distance_meters: float = _MIN_DISTANCE, grace_seconds: float = _GRACE_SECONDS) -> None:
        self._min_distance = min_distance_meters
        self._grace = grace_seconds
        # Per team: wall-clock time encroachment was first observed this
        # stoppage window (None = currently clear). Reset by reset() on any
        # command transition — "there are no individual fouls per robot"
        # implies this is scoped to one continuous stoppage window, not
        # persisted across restarts.
        self._encroaching_since: dict[bool, Optional[float]] = {True: None, False: None}
        # How many times each team has already been fouled THIS stoppage
        # window (used for the "halted after the 2nd foul" escalation).
        self._foul_count: dict[bool, int] = {True: 0, False: 0}

    def check(
        self,
        game_frame: GameFrame,
        geometry: RefereeGeometry,
        current_command: RefereeCommand,
        designated_position: Optional[tuple[float, float]] = None,
    ) -> Optional[RuleViolation]:
        if current_command not in _STOPPAGE_COMMANDS:
            self._encroaching_since = {True: None, False: None}
            self._foul_count = {True: 0, False: 0}
            return None

        now = game_frame.ts
        # yellow_is_right: which physical side yellow defends, from this
        # frame's perspective — same idiom as DefenseAreaRule.
        yellow_is_right = game_frame.my_team_is_right == game_frame.my_team_is_yellow
        dist_to_yellow_defense = (
            geometry.distance_to_right_defense_area if yellow_is_right else geometry.distance_to_left_defense_area
        )
        dist_to_blue_defense = (
            geometry.distance_to_left_defense_area if yellow_is_right else geometry.distance_to_right_defense_area
        )

        yellow_robots = (
            game_frame.friendly_robots.values() if game_frame.my_team_is_yellow else game_frame.enemy_robots.values()
        )
        blue_robots = (
            game_frame.enemy_robots.values() if game_frame.my_team_is_yellow else game_frame.friendly_robots.values()
        )

        # Yellow robots must stay clear of BLUE's defense area (the
        # opponent's), and vice versa.
        yellow_violating = any(dist_to_blue_defense(r.p.x, r.p.y) < self._min_distance for r in yellow_robots)
        blue_violating = any(dist_to_yellow_defense(r.p.x, r.p.y) < self._min_distance for r in blue_robots)

        violation = self._check_team(True, yellow_violating, now, current_command)
        if violation is not None:
            return violation
        return self._check_team(False, blue_violating, now, current_command)

    def _check_team(
        self, is_yellow: bool, violating: bool, now: float, current_command: RefereeCommand
    ) -> Optional[RuleViolation]:
        if not violating:
            self._encroaching_since[is_yellow] = None
            return None

        if self._encroaching_since[is_yellow] is None:
            self._encroaching_since[is_yellow] = now
            return None

        if (now - self._encroaching_since[is_yellow]) < self._grace:
            return None

        # Grace period expired while still encroaching — foul.
        self._foul_count[is_yellow] += 1
        # "The grace period is restarted after the first foul of the same
        # team" — re-arm so a continuously-encroaching robot doesn't refoul
        # every single tick once the grace window has elapsed once.
        self._encroaching_since[is_yellow] = now

        team_name = "Yellow" if is_yellow else "Blue"
        if self._foul_count[is_yellow] >= 2:
            return RuleViolation(
                rule_name="defense_area_stoppage",
                suggested_command=RefereeCommand.HALT,
                next_command=None,
                status_message=f"{team_name} too close to opponent defense area during stoppage (2nd foul — HALT)",
                offending_teams=(is_yellow,),
                counts_toward_foul_counter=True,
            )

        # First offense: "the game is still stopped regularly" — i.e. keep
        # the current stoppage/free-kick command flowing rather than
        # halting, but still charge the foul counter.
        return RuleViolation(
            rule_name="defense_area_stoppage",
            suggested_command=current_command,
            next_command=None,
            status_message=f"{team_name} too close to opponent defense area during stoppage",
            offending_teams=(is_yellow,),
            counts_toward_foul_counter=True,
            is_stopping=False,
        )

    def reset(self) -> None:
        self._encroaching_since = {True: None, False: None}
        self._foul_count = {True: 0, False: 0}
