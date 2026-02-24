"""CustomReferee: orchestrates rule checking and state management."""

from __future__ import annotations

from typing import List, Optional

from utama_core.custom_referee.geometry import RefereeGeometry
from utama_core.custom_referee.profiles.profile_loader import (
    RefereeProfile,
    load_profile,
)
from utama_core.custom_referee.rules.base_rule import BaseRule, RuleViolation
from utama_core.custom_referee.rules.defense_area_rule import DefenseAreaRule
from utama_core.custom_referee.rules.goal_rule import GoalRule
from utama_core.custom_referee.rules.keep_out_rule import KeepOutRule
from utama_core.custom_referee.rules.out_of_bounds_rule import OutOfBoundsRule
from utama_core.custom_referee.state_machine import GameStateMachine
from utama_core.entities.data.referee import RefereeData
from utama_core.entities.game.game_frame import GameFrame
from utama_core.entities.referee.referee_command import RefereeCommand


def _build_active_rules(rules_cfg) -> List[BaseRule]:
    """Construct the ordered list of active rules from a RulesConfig."""
    active: List[BaseRule] = []

    # Priority order: GoalRule → OutOfBoundsRule → DefenseAreaRule → KeepOutRule
    if rules_cfg.goal_detection.enabled:
        active.append(GoalRule(cooldown_seconds=rules_cfg.goal_detection.cooldown_seconds))

    if rules_cfg.out_of_bounds.enabled:
        active.append(OutOfBoundsRule())

    if rules_cfg.defense_area.enabled:
        active.append(
            DefenseAreaRule(
                max_defenders=rules_cfg.defense_area.max_defenders,
                attacker_infringement=rules_cfg.defense_area.attacker_infringement,
            )
        )

    if rules_cfg.keep_out.enabled:
        active.append(
            KeepOutRule(
                radius_meters=rules_cfg.keep_out.radius_meters,
                violation_persistence_frames=rules_cfg.keep_out.violation_persistence_frames,
            )
        )

    return active


class CustomReferee:
    """Stateful referee that operates on ``GameFrame`` objects.

    Works uniformly across Real, grSim, and RSim modes because it does not
    depend on any network receiver or simulator-specific data.

    Usage::

        referee = CustomReferee.from_profile_name("strict_ai")
        ref_data = referee.step(game_frame, time.time())
    """

    def __init__(
        self,
        profile: RefereeProfile,
        n_robots_yellow: int = 3,
        n_robots_blue: int = 3,
    ) -> None:
        self._geometry: RefereeGeometry = profile.geometry
        self._rules: List[BaseRule] = _build_active_rules(profile.rules)
        self._state = GameStateMachine(
            half_duration_seconds=profile.game.half_duration_seconds,
            kickoff_team=profile.game.kickoff_team,
            n_robots_yellow=n_robots_yellow,
            n_robots_blue=n_robots_blue,
            force_start_after_goal=profile.game.force_start_after_goal,
            stop_duration_seconds=profile.game.stop_duration_seconds,
        )

    @classmethod
    def from_profile_name(
        cls,
        name: str,
        n_robots_yellow: int = 3,
        n_robots_blue: int = 3,
    ) -> "CustomReferee":
        """Convenience constructor: load profile by built-in name or file path."""
        profile = load_profile(name)
        return cls(profile, n_robots_yellow=n_robots_yellow, n_robots_blue=n_robots_blue)

    # ------------------------------------------------------------------
    # Main loop interface
    # ------------------------------------------------------------------

    def step(self, game_frame: GameFrame, current_time: float) -> RefereeData:
        """Evaluate all rules and advance the state machine by one tick.

        First matching rule (in priority order) wins; subsequent rules are
        not evaluated.
        """
        violation: Optional[RuleViolation] = None
        for rule in self._rules:
            result = rule.check(game_frame, self._geometry, self._state.command)
            if result is not None:
                violation = result
                break

        # Notify rules of any command transition so they can reset internal state.
        if violation is not None:
            for rule in self._rules:
                rule.reset()

        return self._state.step(current_time, violation)

    def set_command(self, command: RefereeCommand, timestamp: float) -> None:
        """Manual override — for operator use or test scripting."""
        self._state.set_command(command, timestamp)

    # ------------------------------------------------------------------
    # Properties (read-only access for callers that need to inspect state)
    # ------------------------------------------------------------------

    @property
    def geometry(self) -> RefereeGeometry:
        return self._geometry
