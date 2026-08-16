"""Base class and violation dataclass for all referee rules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from utama_core.custom_referee.geometry import RefereeGeometry
from utama_core.entities.game.game_frame import GameFrame
from utama_core.entities.referee.referee_command import RefereeCommand


@dataclass(frozen=True)
class RuleViolation:
    """Describes a detected rule infringement and the appropriate response."""

    rule_name: str
    suggested_command: RefereeCommand
    next_command: Optional[RefereeCommand]
    status_message: str
    designated_position: Optional[tuple[float, float]] = None


class BaseRule(ABC):
    """Abstract base class for all modular referee rules."""

    @abstractmethod
    def check(
        self,
        game_frame: GameFrame,
        geometry: RefereeGeometry,
        current_command: RefereeCommand,
    ) -> Optional[RuleViolation]:
        """Check for a rule violation in the current game frame.

        Returns a RuleViolation if one is detected, otherwise None.
        """
        ...

    def reset(self) -> None:
        """Called when a command transition occurs; reset internal state.

        NOT the same as a full episode reset — cooldowns/timestamps that
        should persist *across* command transitions (e.g. `GoalRule`'s
        cooldown, which must survive the STOP that follows a goal) are
        deliberately kept here. Use `reset_for_new_episode()` to also clear
        those.
        """
        pass

    def reset_for_new_episode(self) -> None:
        """Called by `CustomReferee.reset()` when starting a fresh episode
        (e.g. for RL training reusing one referee instance). Clears
        everything `reset()` does, plus any state that normally survives
        command transitions (cooldown timestamps, etc.) — a new episode's
        clock starts fresh, so nothing from the previous episode should
        carry over. Default implementation just calls `reset()`; override
        when a rule keeps state that `reset()` intentionally preserves.
        """
        self.reset()
