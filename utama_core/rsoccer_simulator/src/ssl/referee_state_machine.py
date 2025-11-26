"""Embedded referee state machine for RSim environment.

This module implements a referee system for the RSim SSL environment that
generates RefereeData synchronously with simulation steps, maintaining
interface compatibility with network-based referee systems.
"""

import logging
from typing import Optional

import numpy as np

from utama_core.entities.data.referee import RefereeData
from utama_core.entities.game.team_info import TeamInfo
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.entities.referee.stage import Stage
from utama_core.rsoccer_simulator.src.Entities import Frame

logger = logging.getLogger(__name__)


class RefereeStateMachine:
    """Manages referee state and generates RefereeData for RSim.

    This class detects game events from simulation state, tracks scores and
    timers, and generates valid RefereeData objects compatible with the
    network-based referee system used in GRSIM/REAL modes.

    Attributes:
        stage: Current game stage (NORMAL_FIRST_HALF, etc.)
        command: Current referee command (HALT, STOP, NORMAL_START, etc.)
        command_counter: Increments each time command changes
        command_timestamp: Timestamp when current command was issued
        stage_start_time: When current stage started
        stage_duration: Duration of current stage in seconds
        yellow_team: Team info for yellow team (score, cards, etc.)
        blue_team: Team info for blue team
    """

    def __init__(
        self,
        n_robots_blue: int,
        n_robots_yellow: int,
        field,
        initial_stage: Stage = Stage.NORMAL_FIRST_HALF_PRE,
        initial_command: RefereeCommand = RefereeCommand.HALT,
    ):
        """Initialize referee state machine.

        Args:
            n_robots_blue: Number of blue robots
            n_robots_yellow: Number of yellow robots
            field: Field object with dimensions
            initial_stage: Starting game stage
            initial_command: Starting referee command
        """
        self.n_robots_blue = n_robots_blue
        self.n_robots_yellow = n_robots_yellow
        self.field = field

        # Field dimensions (meters)
        self.field_half_length = field.length / 2
        self.field_half_width = field.width / 2

        # State tracking
        self.stage = initial_stage
        self.command = initial_command
        self.command_counter = 0
        self.command_timestamp = 0.0

        # Timers
        self.stage_start_time = 0.0
        self.stage_duration = 300.0  # 5 minutes per half
        self.action_timeout = None

        # Team info
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

        # Game state
        self.ball_last_touched_by = None
        self.ball_placement_target = None
        self.next_command = None
        self.goal_scored_by = None

        # Event detection state
        self.last_ball_position = None
        self.last_goal_time = 0.0
        self.goal_cooldown = 0.5  # seconds before detecting another goal

        logger.info(
            "RefereeStateMachine initialized: stage=%s, command=%s",
            self.stage.name,
            self.command.name,
        )

    def update(self, frame: Frame, current_time: float) -> None:
        """Update referee state based on simulation frame.

        This should be called every simulation step.

        Args:
            frame: Current simulation frame with ball and robot positions
            current_time: Current simulation time in seconds
        """
        # Detect and process game events
        self._detect_and_process_events(frame, current_time)

        # Update timers
        self._update_timers(current_time)

    def _detect_and_process_events(self, frame: Frame, current_time: float) -> None:
        """Detect game events and update state accordingly.

        Args:
            frame: Current simulation frame
            current_time: Current simulation time
        """
        # Goal detection (with cooldown to prevent multiple detections)
        if current_time - self.last_goal_time > self.goal_cooldown:
            if self._is_goal(frame):
                self._process_goal(current_time)

    def _is_goal(self, frame: Frame) -> bool:
        """Detect if ball is in goal.

        Args:
            frame: Current simulation frame

        Returns:
            True if goal scored, False otherwise
        """
        ball = frame.ball
        # goal_depth = 0.2  # meters behind goal line
        goal_width = self.field.goal_width / 2  # half width

        # Left goal (yellow defends) - negative x
        if ball.x < -self.field_half_length and abs(ball.y) < goal_width:
            self.goal_scored_by = "blue"
            logger.info("Goal scored by blue team!")
            return True

        # Right goal (blue defends) - positive x
        if ball.x > self.field_half_length and abs(ball.y) < goal_width:
            self.goal_scored_by = "yellow"
            logger.info("Goal scored by yellow team!")
            return True

        return False

    def _process_goal(self, current_time: float) -> None:
        """Process a goal event.

        Updates score, sets STOP command, and prepares kickoff for opposite team.

        Args:
            current_time: Time when goal was scored
        """
        if self.goal_scored_by == "yellow":
            self.yellow_team = self.yellow_team._replace(score=self.yellow_team.score + 1)
            self.next_command = RefereeCommand.PREPARE_KICKOFF_BLUE
            logger.info("Yellow scored! Score: Yellow %d - Blue %d", self.yellow_team.score, self.blue_team.score)
        elif self.goal_scored_by == "blue":
            self.blue_team = self.blue_team._replace(score=self.blue_team.score + 1)
            self.next_command = RefereeCommand.PREPARE_KICKOFF_YELLOW
            logger.info("Blue scored! Score: Yellow %d - Blue %d", self.yellow_team.score, self.blue_team.score)

        # Set STOP command after goal
        self.command = RefereeCommand.STOP
        self.command_counter += 1
        self.command_timestamp = current_time
        self.last_goal_time = current_time

        logger.info(
            "Referee command: STOP (after goal), next: %s", self.next_command.name if self.next_command else "None"
        )

    def _update_timers(self, current_time: float) -> None:
        """Update stage and action timers.

        Args:
            current_time: Current simulation time
        """
        # Stage timer automatically counts down based on elapsed time
        # No action needed here, calculated in _generate_referee_data()
        pass

    def _generate_referee_data(self, current_time: float) -> RefereeData:
        """Generate RefereeData from current state.

        Args:
            current_time: Current simulation time

        Returns:
            RefereeData object with current referee state
        """
        stage_time_left = max(0, self.stage_duration - (current_time - self.stage_start_time))

        return RefereeData(
            source_identifier="rsim-embedded",
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
            current_action_time_remaining=self.action_timeout,
        )

    def get_referee_data(self, current_time: float) -> RefereeData:
        """Get current referee data without updating state.

        Args:
            current_time: Current simulation time

        Returns:
            RefereeData object with current referee state
        """
        return self._generate_referee_data(current_time)

    def set_command(self, command: RefereeCommand, timestamp: float = None) -> None:
        """Manually set referee command (for testing/scenarios).

        Args:
            command: Referee command to set
            timestamp: Optional timestamp, uses current command timestamp if None
        """
        self.command = command
        self.command_counter += 1
        if timestamp is not None:
            self.command_timestamp = timestamp
        logger.info("Referee command manually set to: %s", command.name)

    def advance_stage(self, new_stage: Stage, timestamp: float) -> None:
        """Manually advance to a new stage.

        Args:
            new_stage: Stage to advance to
            timestamp: Timestamp when stage change occurs
        """
        logger.info("Stage advancing from %s to %s", self.stage.name, new_stage.name)
        self.stage = new_stage
        self.stage_start_time = timestamp

        # Set appropriate duration for new stage
        if new_stage in [Stage.NORMAL_FIRST_HALF, Stage.NORMAL_SECOND_HALF]:
            self.stage_duration = 300.0  # 5 minutes
        elif new_stage in [Stage.EXTRA_FIRST_HALF, Stage.EXTRA_SECOND_HALF]:
            self.stage_duration = 150.0  # 2.5 minutes
        else:
            self.stage_duration = 0.0
