"""GameStateMachine: owns all mutable game state for the CustomReferee."""

from __future__ import annotations

import logging
import math
import time
from typing import Optional

from utama_core.custom_referee.rules.base_rule import RuleViolation
from utama_core.entities.data.referee import RefereeData
from utama_core.entities.game.team_info import TeamInfo
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.entities.referee.stage import Stage

logger = logging.getLogger(__name__)

_TRANSITION_COOLDOWN = 0.3  # seconds — prevents command oscillation


class GameStateMachine:
    """Owns score, command, and stage.  Produces ``RefereeData`` each tick."""

    def __init__(
        self,
        half_duration_seconds: float,
        kickoff_team: str,
        n_robots_yellow: int,
        n_robots_blue: int,
        initial_stage: Stage = Stage.NORMAL_FIRST_HALF_PRE,
        initial_command: RefereeCommand = RefereeCommand.HALT,
        force_start_after_goal: bool = False,
        stop_duration_seconds: float = 3.0,
    ) -> None:
        self.command = initial_command
        self.command_counter = 0
        self.command_timestamp = 0.0

        self.stage = initial_stage
        self.stage_start_time = time.time()  # initialise to now so timer is correct immediately
        self.stage_duration = half_duration_seconds

        self.yellow_team = TeamInfo(
            name="Yellow",
            score=0,
            red_cards=0,
            yellow_card_times=[],
            yellow_cards=0,
            timeouts=4,
            timeout_time=300,
            goalkeeper=0,
            foul_counter=0,
            ball_placement_failures=0,
            can_place_ball=True,
            max_allowed_bots=n_robots_yellow,
            bot_substitution_intent=False,
            bot_substitution_allowed=True,
            bot_substitutions_left=5,
        )
        self.blue_team = TeamInfo(
            name="Blue",
            score=0,
            red_cards=0,
            yellow_card_times=[],
            yellow_cards=0,
            timeouts=4,
            timeout_time=300,
            goalkeeper=0,
            foul_counter=0,
            ball_placement_failures=0,
            can_place_ball=True,
            max_allowed_bots=n_robots_blue,
            bot_substitution_intent=False,
            bot_substitution_allowed=True,
            bot_substitutions_left=5,
        )

        self.next_command: Optional[RefereeCommand] = None
        self.ball_placement_target: Optional[tuple[float, float]] = None

        # Kickoff team initialised from profile.
        self._kickoff_team_is_yellow = kickoff_team.lower() == "yellow"

        # Arcade auto-advance: after stop_duration_seconds in STOP following a
        # goal, automatically issue FORCE_START instead of waiting for operator.
        self._force_start_after_goal = force_start_after_goal
        self._stop_duration_seconds = stop_duration_seconds
        self._stop_entered_time: float = -math.inf  # wall time when STOP was last entered

        # Cooldown: don't process a new violation within this window.
        self._last_transition_time: float = -math.inf

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, current_time: float, violation: Optional[RuleViolation]) -> RefereeData:
        """Process one tick.  Apply violation if not in cooldown.  Return RefereeData."""
        if violation is not None and self._can_transition(current_time):
            self._apply_violation(violation, current_time)

        # Arcade mode: auto-advance from STOP → FORCE_START after stop_duration_seconds.
        if (
            self._force_start_after_goal
            and self.command == RefereeCommand.STOP
            and self.next_command
            in (
                RefereeCommand.PREPARE_KICKOFF_YELLOW,
                RefereeCommand.PREPARE_KICKOFF_BLUE,
            )
            and (current_time - self._stop_entered_time) >= self._stop_duration_seconds
        ):
            self.command = RefereeCommand.FORCE_START
            self.command_counter += 1
            self.command_timestamp = current_time
            self.next_command = None
            self._last_transition_time = current_time
            logger.info("Auto-advanced STOP → FORCE_START after goal (arcade mode)")

        return self._generate_referee_data(current_time)

    def set_command(self, command: RefereeCommand, timestamp: float) -> None:
        """Manual override — for operator use or test scripting."""
        self.command = command
        self.command_counter += 1
        self.command_timestamp = timestamp

        # Advance PRE stages to their active counterpart when play begins.
        _PRE_TO_ACTIVE = {
            Stage.NORMAL_FIRST_HALF_PRE: Stage.NORMAL_FIRST_HALF,
            Stage.NORMAL_SECOND_HALF_PRE: Stage.NORMAL_SECOND_HALF,
            Stage.EXTRA_FIRST_HALF_PRE: Stage.EXTRA_FIRST_HALF,
            Stage.EXTRA_SECOND_HALF_PRE: Stage.EXTRA_SECOND_HALF,
        }
        if command in (RefereeCommand.NORMAL_START, RefereeCommand.FORCE_START):
            active = _PRE_TO_ACTIVE.get(self.stage)
            if active is not None:
                self.advance_stage(active, timestamp)

        logger.info("Referee command manually set to: %s", command.name)

    def advance_stage(self, new_stage: Stage, timestamp: float) -> None:
        """Advance the game stage."""
        logger.info("Stage %s → %s", self.stage.name, new_stage.name)
        self.stage = new_stage
        self.stage_start_time = timestamp

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _can_transition(self, current_time: float) -> bool:
        return (current_time - self._last_transition_time) >= _TRANSITION_COOLDOWN

    def _apply_violation(self, violation: RuleViolation, current_time: float) -> None:
        """Update state in response to a detected violation."""
        if violation.rule_name == "goal":
            self._handle_goal(violation, current_time)
        else:
            self._handle_foul(violation, current_time)

        self._last_transition_time = current_time

    def _handle_goal(self, violation: RuleViolation, current_time: float) -> None:
        # Determine scorer from next_command (loser gets the kickoff).
        if violation.next_command == RefereeCommand.PREPARE_KICKOFF_BLUE:
            # Blue gets kickoff → yellow scored.
            self.yellow_team.increment_score()
            logger.info(
                "Goal by Yellow! Score: Yellow %d – Blue %d",
                self.yellow_team.score,
                self.blue_team.score,
            )
        elif violation.next_command == RefereeCommand.PREPARE_KICKOFF_YELLOW:
            # Yellow gets kickoff → blue scored.
            self.blue_team.increment_score()
            logger.info(
                "Goal by Blue! Score: Yellow %d – Blue %d",
                self.yellow_team.score,
                self.blue_team.score,
            )

        self.command = RefereeCommand.STOP
        self.command_counter += 1
        self.command_timestamp = current_time
        self.next_command = violation.next_command
        self.ball_placement_target = (0.0, 0.0)
        self._stop_entered_time = current_time

    def _handle_foul(self, violation: RuleViolation, current_time: float) -> None:
        self.command = violation.suggested_command
        self.command_counter += 1
        self.command_timestamp = current_time
        self.next_command = violation.next_command
        self.ball_placement_target = violation.designated_position
        logger.info(
            "Foul detected: %s → %s (next: %s)",
            violation.rule_name,
            violation.suggested_command.name,
            violation.next_command.name if violation.next_command else "None",
        )

    def _generate_referee_data(self, current_time: float) -> RefereeData:
        stage_time_left = max(0.0, self.stage_duration - (current_time - self.stage_start_time))
        return RefereeData(
            source_identifier="custom_referee",
            time_sent=current_time,
            time_received=current_time,
            referee_command=self.command,
            referee_command_timestamp=self.command_timestamp,
            stage=self.stage,
            stage_time_left=stage_time_left,
            blue_team=self.blue_team,
            yellow_team=self.yellow_team,
            designated_position=self.ball_placement_target,
            blue_team_on_positive_half=None,
            next_command=self.next_command,
            current_action_time_remaining=None,
        )
