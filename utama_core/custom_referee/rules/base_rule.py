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
    """Describes a detected rule infringement and the appropriate response.

    `offending_teams`: which team(s) (`True` = yellow, `False` = blue) the
    foul counter (SSL rulebook §8.4: "every third increase ... causes a
    yellow card") should be charged against. Empty tuple for a violation
    that isn't attributable to either team's fault (e.g. a goal, or a
    symmetric-force Pushing case where "no team is at fault" per the
    rulebook) — `_apply_violation` skips counter incrementing entirely in
    that case. A tuple can hold both booleans for a foul the rulebook
    charges to both sides at once (e.g. Crashing's <0.3 m/s
    absolute-speed-difference branch).

    `counts_toward_foul_counter`: some violations intentionally bypass the
    counter even though a team is at fault — e.g. Multiple Defenders, which
    the rulebook explicitly carves out ("The foul counter is not
    increased."). Defaults to True since that's the common case.

    `is_stopping`: SSL rulebook §8.4.1 vs §8.4.2 — a stopping foul (the
    default, matching every rule that existed before this field was added)
    causes the state machine to actually transition `command`/
    `next_command` per `suggested_command`/`next_command` below. A
    *non*-stopping foul (e.g. Crashing) must apply its foul-counter/card
    side effect without perturbing the live command at all — "the game
    continues normally". Set False for those; `suggested_command` is still
    required by the dataclass shape but is ignored by `_apply_violation`
    when `is_stopping` is False (pass the rule's own `current_command`
    through unused, or any placeholder — it is never read in that path).
    """

    rule_name: str
    suggested_command: RefereeCommand
    next_command: Optional[RefereeCommand]
    status_message: str
    designated_position: Optional[tuple[float, float]] = None
    offending_teams: tuple[bool, ...] = ()
    counts_toward_foul_counter: bool = True
    is_stopping: bool = True


class BaseRule(ABC):
    """Abstract base class for all modular referee rules."""

    @abstractmethod
    def check(
        self,
        game_frame: GameFrame,
        geometry: RefereeGeometry,
        current_command: RefereeCommand,
        designated_position: Optional[tuple[float, float]] = None,
    ) -> Optional[RuleViolation]:
        """Check for a rule violation in the current game frame.

        `designated_position`: the state machine's current ball-placement
        target (`GameStateMachine.ball_placement_target`), one tick stale
        like every other value a rule observes here — not otherwise
        reachable from `GameFrame`/`RefereeData`, which `CustomReferee`
        never populates internally. Optional/defaulted so existing rules
        that don't need it are unaffected. Only `BallPlacementInterferenceRule`
        uses it today.

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
