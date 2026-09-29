"""CrashingRule: SSL rulebook §8.4.2 "Crashing" (non-stopping foul).

"At the moment of collision of two robots of different teams, the
difference of the speed vectors of both robots is taken and projected
onto the line that is defined by the position of both robots. If the
length of this projection is greater than 1.5 meters per second, the
faster robot committed a foul. If the absolute robot speed difference is
less than 0.3 meters per second, both conduct a foul."

A contact is a crash only when that projection exceeds 1.5 m/s; "faster"
then compares the two robots' own speeds, and within 0.3 m/s both are at
fault. As in TIGERs AutoReferee's `BotCollisionDetector` (the referee real
matches use), each robot's velocity is first shortened by what it can brake in
0.1 s (`_BRAKE_LOOKAHEAD_S` at `_BRAKE_DECELERATION`, 0.4 m/s), floored at 0.
Worked examples:
  - A stationary, B at 2 m/s into it: projection 1.6 > 1.5, speeds differ by
    1.6 -> B fouls.
  - Head-on, each at 1.2 m/s: projection 0.8 + 0.8 = 1.6 > 1.5, speeds equal
    -> both foul. Each at 1 m/s (1.2 after braking) is no crash.
  - Two robots resting against each other: projection ~0 -> no crash.
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
# TIGERs' `botBrakeLookahead` (0.1 s) at our robots' MAX_ACCELERATION (4 m/s^2 in
# every `RobotParams`). Without it, round-robin at 5be4df48 had 226 crashes where
# TIGERs' rule over the same frames finds 8: mostly two robots each at about 1 m/s.
_BRAKE_LOOKAHEAD_S = 0.1
_BRAKE_DECELERATION = 4.0  # m/s^2

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

            # A crash is > threshold along the line between the robots; only then does
            # the 0.3 m/s band decide between the faster robot and both (TIGERs
            # AutoReferee's BotCollisionDetector reads the rule the same way).
            friendly_v = _braked_velocity(contact.friendly.v)
            enemy_v = _braked_velocity(contact.enemy.v)
            ux, uy = contact.direction_enemy_to_friendly
            crash_speed = (enemy_v[0] - friendly_v[0]) * ux + (enemy_v[1] - friendly_v[1]) * uy
            if abs(crash_speed) <= self._fault_speed_threshold:
                fault = None
            else:
                speed_diff = math.hypot(*friendly_v) - math.hypot(*enemy_v)
                if abs(speed_diff) < self._both_fault_threshold:
                    fault = "both"
                else:
                    fault = "friendly" if speed_diff > 0 else "enemy"

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
                    offending_robots=(
                        (my_team_is_yellow, contact.friendly.id),
                        (not my_team_is_yellow, contact.enemy.id),
                    ),
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
                    offending_robots=(
                        (faster_is_yellow, contact.friendly.id if faster_is_friendly else contact.enemy.id),
                    ),
                    is_stopping=False,
                )

        self._prev_contact_pairs = current_pairs
        return violation

    def reset(self) -> None:
        self._prev_contact_pairs.clear()
        self._last_fired_ts.clear()


def _braked_velocity(v) -> tuple[float, float]:
    """`v` shortened by what the robot brakes in `_BRAKE_LOOKAHEAD_S`, never reversed."""
    speed = math.hypot(v.x, v.y)
    if speed == 0.0:
        return (0.0, 0.0)
    scale = max(speed - _BRAKE_DECELERATION * _BRAKE_LOOKAHEAD_S, 0.0) / speed
    return (v.x * scale, v.y * scale)
