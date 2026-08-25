"""BallPlacementInterferenceRule: enforces standoff from the placement line
during BALL_PLACEMENT_* (SSL rulebook §8.4.3).

"During ball placement, all robots of the non-placing team have to keep at
least 0.5 meters distance to the line between the ball and the placement
position (the forbidden area forms a stadium shape). If a robot of the
non-placing team is too close to the line ... for more than 2 seconds, it
commits a foul. In this case, 10 seconds are added to the ball placement
timer. Only one interference foul per ball placement phase counts towards
the foul counter, but the placement timer is always incremented."

KNOWN GAP: there is no ball-placement deadline/timer in `GameStateMachine`
today to actually add 10 seconds to (`TeamInfo.ball_placement_failures` is a
failure *count*, not an elapsed-time timer — checked, and it's never
incremented by any existing code either). This rule correctly detects the
foul, attributes it to the interfering team, and enforces "only the first
interference per placement phase counts toward the foul counter" — but the
"+10s to the timer" side effect has no real timer to extend into yet. Flag
this in `status_message` and leave a clear TODO rather than inventing new
state-machine deadline plumbing unprompted; whoever adds a real placement
timeout should wire it to `interference_committed_this_phase` (below) the
same way `counts_toward_foul_counter` is already gated on it here.
"""

from __future__ import annotations

from typing import Optional

from utama_core.custom_referee.geometry import RefereeGeometry
from utama_core.custom_referee.rules.base_rule import BaseRule, RuleViolation
from utama_core.entities.game.game_frame import GameFrame
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.global_utils.math_utils import distance_point_to_segment

_BALL_PLACEMENT_COMMANDS = {
    RefereeCommand.BALL_PLACEMENT_YELLOW,
    RefereeCommand.BALL_PLACEMENT_BLUE,
}


class BallPlacementInterferenceRule(BaseRule):
    """Penalises non-placing-team robots that linger inside the 0.5m
    "stadium" zone around the ball-to-placement-target line during
    BALL_PLACEMENT_*.

    Mirrors `KeepOutRule`'s persistence-frame pattern (rulebook here uses a
    2-second wall-clock grace instead of a frame count, since interference
    is specifically time-gated in the spec text — tracked as
    `_over_since: Optional[float]` rather than a frame counter so it's
    robust to variable tick rate).
    """

    _GRACE_SECONDS = 2.0
    _STADIUM_RADIUS = 0.5  # metres

    def __init__(
        self,
        stadium_radius_meters: float = _STADIUM_RADIUS,
        grace_seconds: float = _GRACE_SECONDS,
    ) -> None:
        self._radius = stadium_radius_meters
        self._grace = grace_seconds
        self._over_since: Optional[float] = None
        # Only the *first* interference foul in a placement phase counts
        # toward the foul counter (rulebook, quoted above) — reset whenever
        # we leave BALL_PLACEMENT_* (see reset()).
        self._counted_this_phase = False

    def check(
        self,
        game_frame: GameFrame,
        geometry: RefereeGeometry,
        current_command: RefereeCommand,
        designated_position: Optional[tuple[float, float]] = None,
    ) -> Optional[RuleViolation]:
        if current_command not in _BALL_PLACEMENT_COMMANDS or designated_position is None:
            self._over_since = None
            return None

        ball = game_frame.ball
        if ball is None:
            self._over_since = None
            return None

        ball_pos = (ball.p.x, ball.p.y)

        # The placing team is the one named by the command; the other
        # team's robots are the ones subject to the standoff.
        placing_team_is_yellow = current_command == RefereeCommand.BALL_PLACEMENT_YELLOW
        if placing_team_is_yellow == game_frame.my_team_is_yellow:
            non_placing_robots = game_frame.enemy_robots.values()
        else:
            non_placing_robots = game_frame.friendly_robots.values()

        interfering = any(
            distance_point_to_segment((r.p.x, r.p.y), ball_pos, designated_position) < self._radius
            for r in non_placing_robots
        )

        if not interfering:
            self._over_since = None
            return None

        now = game_frame.ts
        if self._over_since is None:
            self._over_since = now
            return None

        if (now - self._over_since) < self._grace:
            return None

        # Foul confirmed. Reset the timer so a continuously-interfering
        # robot doesn't refire every tick — same "only counts once per
        # phase" idea KeepOutRule uses, but here we still want the
        # placement-timer-extension side effect to fire on repeat
        # violations (see rulebook: "the placement timer is always
        # incremented" even when the foul counter isn't) — so re-arm
        # _over_since rather than latching forever, but only charge the
        # foul counter once per phase via _counted_this_phase.
        self._over_since = now
        offender_is_yellow = not placing_team_is_yellow
        first_in_phase = not self._counted_this_phase
        self._counted_this_phase = True

        return RuleViolation(
            rule_name="ball_placement_interference",
            suggested_command=current_command,
            next_command=None,
            status_message=(
                "Ball placement interference"
                + ("" if first_in_phase else " (repeat — foul counter already charged this phase)")
                + " — NOTE: no placement-timer deadline exists yet to add 10s to (see module docstring)"
            ),
            offending_teams=(offender_is_yellow,),
            counts_toward_foul_counter=first_in_phase,
            is_stopping=False,
        )

    def reset(self) -> None:
        self._over_since = None
        self._counted_this_phase = False
