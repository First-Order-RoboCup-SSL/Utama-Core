"""DefenseAreaRule: detects illegal entry into defense areas."""

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
        n_yellow_defenders = sum(1 for r in yellow_robots if in_yellow_defense(r.p.x, r.p.y))
        if n_yellow_defenders > self._max_defenders:
            return RuleViolation(
                rule_name="defense_area",
                suggested_command=RefereeCommand.STOP,
                next_command=RefereeCommand.DIRECT_FREE_BLUE,
                status_message="Too many yellow defenders in own area",
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

        # --- Blue defense area ---
        n_blue_defenders = sum(1 for r in blue_robots if in_blue_defense(r.p.x, r.p.y))
        if n_blue_defenders > self._max_defenders:
            return RuleViolation(
                rule_name="defense_area",
                suggested_command=RefereeCommand.STOP,
                next_command=RefereeCommand.DIRECT_FREE_YELLOW,
                status_message="Too many blue defenders in own area",
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
