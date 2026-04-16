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


class DefenseAreaRule(BaseRule):
    """Detects attacker encroachment or too many defenders in the defense area."""

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

        my_team_is_right = game_frame.my_team_is_right
        my_team_is_yellow = game_frame.my_team_is_yellow

        # Determine which geometry helper corresponds to "my" defense area.
        if my_team_is_right:
            in_my_defense = geometry.is_in_right_defense_area
            # in_opp_defense = geometry.is_in_left_defense_area
        else:
            in_my_defense = geometry.is_in_left_defense_area
            # in_opp_defense = geometry.is_in_right_defense_area

        # Check: too many friendly defenders in own area.
        n_friendly_in_own = sum(1 for r in game_frame.friendly_robots.values() if in_my_defense(r.p.x, r.p.y))
        if n_friendly_in_own > self._max_defenders:
            # Opponent gets a free kick.
            free_kick_cmd = RefereeCommand.DIRECT_FREE_BLUE if my_team_is_yellow else RefereeCommand.DIRECT_FREE_YELLOW
            return RuleViolation(
                rule_name="defense_area",
                suggested_command=RefereeCommand.STOP,
                next_command=free_kick_cmd,
                status_message="Too many defenders in own area",
            )

        # Check: enemy attacker inside our defense area.
        if self._attacker_infringement:
            for robot in game_frame.enemy_robots.values():
                if in_my_defense(robot.p.x, robot.p.y):
                    # Defending team (friendly) gets the free kick.
                    free_kick_cmd = (
                        RefereeCommand.DIRECT_FREE_YELLOW if my_team_is_yellow else RefereeCommand.DIRECT_FREE_BLUE
                    )
                    return RuleViolation(
                        rule_name="defense_area",
                        suggested_command=RefereeCommand.STOP,
                        next_command=free_kick_cmd,
                        status_message="Attacker in defense area",
                    )

        return None
