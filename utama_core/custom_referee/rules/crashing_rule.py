"""CrashingRule: SSL rulebook §8.4.2 "Crashing" (non-stopping foul).

"At the moment of collision of two robots of different teams, the
difference of the speed vectors of both robots is taken and projected
onto the line that is defined by the position of both robots. If the
length of this projection is greater than 1.5 meters per second, the
faster robot committed a foul. If the absolute robot speed difference is
less than 0.3 meters per second, both conduct a foul."

"Faster" here means whichever robot has the larger *closing* speed into
the other (hit the other one harder), not absolute ground speed — worked
examples:
  - A stationary, B closing at 2 m/s: closing speeds (0, 2), difference
    magnitude 2 m/s > 1.5 -> B (the closing one) fouls.
  - A and B each closing at 1 m/s (head-on): closing speeds (1, 1),
    difference magnitude ~0 < 0.3 -> both foul (SSL calls this out
    explicitly — a genuine head-on collision at matched speed is nobody's
    fault alone).
  - A closing at 1.5 m/s, B retreating at -0.5 m/s: closing speeds
    (1.5, -0.5), difference magnitude 2.0 -> A fouls (clearly the one
    that closed the gap).
`RobotPairContact.projected_velocity_difference` is exactly this
same-line projected difference by construction; the per-side
`closing_speed_*_into_*` properties tell us *which* robot was faster so
we can attribute fault correctly (the raw projected difference is signed
by team, not by "which one was faster").

Purely position/velocity-based (see `robot_contact.py`'s module docstring
for why `has_ball` is deliberately not used here).
"""

from __future__ import annotations

import math
from typing import Optional

from utama_core.custom_referee.geometry import RefereeGeometry
from utama_core.custom_referee.rules.base_rule import BaseRule, RuleViolation
from utama_core.custom_referee.rules.robot_contact import find_robot_pair_contacts
from utama_core.entities.game.game_frame import GameFrame
from utama_core.entities.referee.referee_command import RefereeCommand

_ACTIVE_PLAY_COMMANDS = {
    RefereeCommand.NORMAL_START,
    RefereeCommand.FORCE_START,
}

_FAULT_SPEED_THRESHOLD = 1.5  # m/s — above this, the faster robot alone fouls
_BOTH_FAULT_THRESHOLD = 0.3  # m/s — below this closing-speed difference, both foul

# Per §8.4.2's preamble: "The same no stop foul cannot be triggered again
# until the foul condition has stopped being violated or there has been 2
# seconds since the foul was first triggered." Re-checked as an OR below.
_RETRIGGER_COOLDOWN_SECONDS = 2.0


class CrashingRule(BaseRule):
    """Detects robot-robot collisions exceeding the closing-speed fault
    thresholds, edge-triggered on the tick contact first begins (not a
    sustained-contact check like PushingRule — a crash is a single event
    at the moment of collision, not an ongoing state).
    """

    def __init__(
        self,
        fault_speed_threshold_mps: float = _FAULT_SPEED_THRESHOLD,
        both_fault_threshold_mps: float = _BOTH_FAULT_THRESHOLD,
        retrigger_cooldown_seconds: float = _RETRIGGER_COOLDOWN_SECONDS,
    ) -> None:
        self._fault_speed_threshold = fault_speed_threshold_mps
        self._both_fault_threshold = both_fault_threshold_mps
        self._retrigger_cooldown = retrigger_cooldown_seconds
        # Pairs in contact as of the previous tick — used to find the
        # contact rising edge ("moment of collision").
        self._prev_contact_pairs: set[tuple[int, int]] = set()
        # (friendly_id, enemy_id) -> game_frame.ts when this pair last fired
        # a violation, so it isn't re-raised every tick of one long contact.
        self._last_fired_ts: dict[tuple[int, int], float] = {}

    def check(
        self,
        game_frame: GameFrame,
        geometry: RefereeGeometry,
        current_command: RefereeCommand,
        designated_position: Optional[tuple[float, float]] = None,
    ) -> Optional[RuleViolation]:
        if current_command not in _ACTIVE_PLAY_COMMANDS:
            self._prev_contact_pairs.clear()
            self._last_fired_ts.clear()
            return None

        now = game_frame.ts
        current_pairs: set[tuple[int, int]] = set()
        violation: Optional[RuleViolation] = None

        for contact in find_robot_pair_contacts(game_frame):
            key = (contact.friendly.id, contact.enemy.id)
            current_pairs.add(key)

            is_rising_edge = key not in self._prev_contact_pairs
            last_fired = self._last_fired_ts.get(key)
            # Preamble condition, checked as an OR: either this is a fresh
            # contact (condition "stopped being violated" and has now
            # started again), or enough time has passed since it last fired
            # while contact never broke (sustained scraping).
            cooldown_elapsed = last_fired is not None and (now - last_fired) >= self._retrigger_cooldown
            if not is_rising_edge and not cooldown_elapsed:
                continue
            if violation is not None:
                # Only one violation reported per tick, matching every
                # other rule in this package — still record this pair as
                # "fired now" below so it doesn't immediately re-fire next
                # tick once this one's been handled.
                pass

            friendly_closing = contact.closing_speed_friendly_into_enemy
            enemy_closing = contact.closing_speed_enemy_into_friendly
            closing_diff = friendly_closing - enemy_closing
            magnitude = abs(closing_diff)

            if magnitude < self._both_fault_threshold:
                fault = "both"
            elif magnitude > self._fault_speed_threshold:
                fault = "friendly" if closing_diff > 0 else "enemy"
            else:
                # Between the two thresholds: a real collision, but neither
                # "clearly one robot's fault alone" nor "clearly matched" —
                # the rulebook only defines the two named bands, so no
                # foul is raised in the gap between them.
                fault = None

            self._last_fired_ts[key] = now

            if fault is None or violation is not None:
                continue

            my_team_is_yellow = game_frame.my_team_is_yellow
            if fault == "both":
                violation = RuleViolation(
                    rule_name="crashing",
                    suggested_command=current_command,
                    next_command=None,
                    status_message="Crashing — matched closing speed, both teams at fault",
                    offending_teams=(True, False),
                    is_stopping=False,
                )
            else:
                faster_is_friendly = fault == "friendly"
                faster_is_yellow = faster_is_friendly == my_team_is_yellow
                violation = RuleViolation(
                    rule_name="crashing",
                    suggested_command=current_command,
                    next_command=None,
                    status_message="Crashing foul",
                    offending_teams=(faster_is_yellow,),
                    is_stopping=False,
                )

        self._prev_contact_pairs = current_pairs
        return violation

    def reset(self) -> None:
        self._prev_contact_pairs.clear()
        self._last_fired_ts.clear()
