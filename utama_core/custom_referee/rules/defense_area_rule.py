"""DefenseAreaRule: detects illegal entry into defense areas."""

from __future__ import annotations

from typing import Optional

from utama_core.custom_referee.geometry import RefereeGeometry
from utama_core.custom_referee.rules.base_rule import BaseRule, RuleViolation
from utama_core.entities.game.game_frame import GameFrame
from utama_core.entities.game.robot import Robot
from utama_core.entities.referee.referee_command import RefereeCommand


# Contact distance for "this defender is the one who touched the ball" —
# matches `has_ball` semantics elsewhere (dribbler contact), but this rule
# needs to identify *which* extra-defender robot touched it, not just
# whether any friendly/enemy robot did, so it checks `has_ball` directly
# per-robot rather than importing the whole-frame `infer_last_touch_team`
# machinery (last_touch.py), which answers a different question (which
# *team*, colour-blind, across the whole field) than this rule needs
# (which specific robot, already known to be inside its own box).
def _defenders_touching_ball(robots: list[Robot]) -> bool:
    return any(r.has_ball for r in robots)


_ACTIVE_PLAY_COMMANDS = {
    RefereeCommand.NORMAL_START,
    RefereeCommand.FORCE_START,
}


def _split_by_color(game_frame: GameFrame):
    """Return (yellow_robots, blue_robots) as dict_values of Robot."""
    if game_frame.my_team_is_yellow:
        return game_frame.friendly_robots.values(), game_frame.enemy_robots.values()
    else:
        return game_frame.enemy_robots.values(), game_frame.friendly_robots.values()


class DefenseAreaRule(BaseRule):
    """Detects attacker encroachment or too many defenders in either defense area.

    Checks both teams symmetrically regardless of which team is "friendly",
    so enforcement is correct whether CustomReferee is stepped from yellow or
    blue perspective.
    """

    def __init__(self, max_defenders: int = 1, attacker_infringement: bool = True) -> None:
        self._max_defenders = max_defenders
        self._attacker_infringement = attacker_infringement

    def check(
        self,
        game_frame: GameFrame,
        geometry: RefereeGeometry,
        current_command: RefereeCommand,
        designated_position: Optional[tuple[float, float]] = None,
    ) -> Optional[RuleViolation]:
        if current_command not in _ACTIVE_PLAY_COMMANDS:
            return None

        # Derive which side each color defends from the caller's perspective.
        # yellow_is_right is True when yellow defends the right goal.
        yellow_is_right = game_frame.my_team_is_right == game_frame.my_team_is_yellow

        in_yellow_defense = geometry.is_in_right_defense_area if yellow_is_right else geometry.is_in_left_defense_area
        in_blue_defense = geometry.is_in_left_defense_area if yellow_is_right else geometry.is_in_right_defense_area

        yellow_robots, blue_robots = (list(g) for g in _split_by_color(game_frame))

        # --- Yellow defense area ---
        # "Multiple Defenders" (rulebook §8.4.1): the rule as written is
        # "any non-keeper robot touches the ball while entirely inside its
        # own defense area" — it has nothing to do with occupancy count.
        # This rule has no way to know which specific robot is the keeper
        # (`TeamInfo.goalkeeper` isn't threaded into `BaseRule.check()`,
        # only `game_frame`/`geometry`/`current_command`/
        # `designated_position` are), so `max_defenders` (default 1) is
        # used as-is, as an occupancy proxy for "at most the keeper is
        # allowed in here" — the real fix is gating on ball-touch by
        # whichever defender is *beyond* that allowance, not on ball-touch
        # by any occupant (which could wrongly flag the keeper itself).
        # Sanction is a penalty kick, not a free kick, and "the foul
        # counter is not increased" (counts_toward_foul_counter=False).
        yellow_in_own_area = [r for r in yellow_robots if in_yellow_defense(r.p.x, r.p.y)]
        if len(yellow_in_own_area) > self._max_defenders and _defenders_touching_ball(yellow_in_own_area):
            return RuleViolation(
                rule_name="defense_area",
                suggested_command=RefereeCommand.STOP,
                next_command=RefereeCommand.PREPARE_PENALTY_BLUE,
                status_message="Extra yellow defender touched ball inside own defense area",
                counts_toward_foul_counter=False,
            )

        if self._attacker_infringement:
            for r in blue_robots:
                if in_yellow_defense(r.p.x, r.p.y):
                    return RuleViolation(
                        rule_name="defense_area",
                        suggested_command=RefereeCommand.STOP,
                        next_command=RefereeCommand.DIRECT_FREE_YELLOW,
                        status_message="Blue attacker in yellow defense area",
                    )

        # --- Blue defense area --- (mirror of the yellow branch above; see
        # its comment for the ball-touch/occupancy-proxy reasoning.)
        blue_in_own_area = [r for r in blue_robots if in_blue_defense(r.p.x, r.p.y)]
        if len(blue_in_own_area) > self._max_defenders and _defenders_touching_ball(blue_in_own_area):
            return RuleViolation(
                rule_name="defense_area",
                suggested_command=RefereeCommand.STOP,
                next_command=RefereeCommand.PREPARE_PENALTY_YELLOW,
                status_message="Extra blue defender touched ball inside own defense area",
                counts_toward_foul_counter=False,
            )

        if self._attacker_infringement:
            for r in yellow_robots:
                if in_blue_defense(r.p.x, r.p.y):
                    return RuleViolation(
                        rule_name="defense_area",
                        suggested_command=RefereeCommand.STOP,
                        next_command=RefereeCommand.DIRECT_FREE_BLUE,
                        status_message="Yellow attacker in blue defense area",
                    )

        return None
