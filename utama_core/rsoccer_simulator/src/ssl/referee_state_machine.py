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

# Phase 2: Command Sequencing and Game Flow Constants
STOP_TO_KICKOFF_DELAY = 2.0  # seconds before transitioning from STOP to PREPARE_KICKOFF
KICKOFF_PREP_TIMEOUT = 10.0  # seconds to prepare for kickoff before auto-start
FREE_KICK_TIMEOUT = 5.0  # seconds for free kick before auto-start
BALL_OUT_MARGIN = 0.3  # meters outside field lines before ball considered out
BALL_TOUCH_THRESHOLD = 0.5  # meters to determine last ball toucher

# Phase 3: Foul Detection Constants
DEFENSE_AREA_VIOLATION_COOLDOWN = 1.0  # seconds between detections
ROBOT_RADIUS = 0.09  # meters (SSL standard)
COLLISION_DISTANCE_THRESHOLD = 0.18  # meters (2x robot radius)
COLLISION_VELOCITY_THRESHOLD = 0.5  # m/s minimum velocity for collision
COLLISION_COOLDOWN = 1.0  # seconds
MAX_BALL_HOLD_TIME = 10.0  # seconds (SSL rule)
BALL_PROXIMITY_THRESHOLD = 0.15  # meters to consider "holding"

# Phase 3: Ball Placement Constants
BALL_PLACEMENT_TIMEOUT = 10.0  # seconds
BALL_PLACEMENT_SUCCESS_DISTANCE = 0.1  # meters
MAX_PLACEMENT_FAILURES = 5

# Phase 3: Card System Constants
FOULS_FOR_YELLOW_CARD = 3
YELLOW_CARD_TIME = 120  # seconds (2 minutes)
YELLOW_CARDS_FOR_RED = 3


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

        # Phase 2: Command transition timers
        self.stop_command_start_time = None
        self.kickoff_prep_start_time = None
        self.free_kick_start_time = None
        self.action_timer_start = None
        self.action_timer_duration = None

        # Phase 3: Foul detection state
        self.last_defense_area_violation_time = 0.0
        self.last_collision_time = 0.0
        self.ball_holder_team = None
        self.ball_hold_start_time = None
        self.ball_placement_start_time = None
        self.placing_team = None

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
        # 1. Detect and process game events
        self._detect_and_process_events(frame, current_time)

        # 2. Update command transitions (Phase 2)
        self._update_command_transitions(frame, current_time)

        # 3. Update timers
        self._update_timers(current_time)

        # 4. Check stage transitions (Phase 2)
        self._check_stage_transition(current_time)

    def _detect_and_process_events(self, frame: Frame, current_time: float) -> None:
        """Detect game events and update state accordingly.

        Args:
            frame: Current simulation frame
            current_time: Current simulation time
        """
        # PHASE 1: Goal detection (with cooldown to prevent multiple detections)
        if current_time - self.last_goal_time > self.goal_cooldown:
            if self._is_goal(frame):
                self._process_goal(current_time)

        # PHASE 2: Ball out detection
        if self.command == RefereeCommand.NORMAL_START:
            if self._is_ball_out_of_bounds(frame):
                self._process_ball_out(frame, current_time)

        # PHASE 3: Defense area violations
        fouling_team = self._detect_defense_area_violations(frame, current_time)
        if fouling_team:
            self._process_defense_area_violation(fouling_team, current_time)

        # PHASE 3: Robot collisions
        collision_team = self._detect_robot_collisions(frame, current_time)
        if collision_team:
            self._process_collision(collision_team, current_time)

        # PHASE 3: Ball holding violations
        holding_team = self._detect_ball_holding(frame, current_time)
        if holding_team:
            self._process_ball_holding_violation(holding_team, current_time)

        # PHASE 3: Check ball placement progress
        self._check_ball_placement_progress(frame, current_time)

        # PHASE 3: Check and issue cards
        self._check_and_issue_cards(current_time)

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
            self.yellow_team.increment_score()
            self.next_command = RefereeCommand.PREPARE_KICKOFF_BLUE
            logger.info("Yellow scored! Score: Yellow %d - Blue %d", self.yellow_team.score, self.blue_team.score)
        elif self.goal_scored_by == "blue":
            self.blue_team.increment_score()
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
        # Phase 2: Update action timeout countdown
        if self.action_timer_start and self.action_timer_duration:
            elapsed = current_time - self.action_timer_start
            self.action_timeout = max(0, int((self.action_timer_duration - elapsed) * 1000))

            # Clear timer when expired
            if elapsed >= self.action_timer_duration:
                self.action_timer_start = None
                self.action_timer_duration = None
                self.action_timeout = None

        # Phase 3: Update yellow card timers
        self._update_yellow_card_timers(current_time)

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

    def reset(
        self,
        initial_stage: Stage = Stage.NORMAL_FIRST_HALF_PRE,
        initial_command: RefereeCommand = RefereeCommand.HALT,
    ):
        """Reset referee state to initial conditions.

        Called when the simulation is reset (between test episodes or matches).
        This ensures clean state for new episodes without time paradoxes.

        Args:
            initial_stage: Starting game stage
            initial_command: Starting referee command
        """
        # Reset state tracking
        self.stage = initial_stage
        self.command = initial_command
        self.command_counter = 0
        self.command_timestamp = 0.0

        # Reset timers
        self.stage_start_time = 0.0
        self.stage_duration = 300.0
        self.action_timeout = None

        # Reset team scores and states (yellow team)
        self.yellow_team.score = 0
        self.yellow_team.foul_counter = 0
        self.yellow_team.yellow_cards = 0
        self.yellow_team.yellow_card_times = []
        self.yellow_team.red_cards = 0
        self.yellow_team.ball_placement_failures = 0
        self.yellow_team.can_place_ball = True
        self.yellow_team.max_allowed_bots = self.n_robots_yellow

        # Reset team scores and states (blue team)
        self.blue_team.score = 0
        self.blue_team.foul_counter = 0
        self.blue_team.yellow_cards = 0
        self.blue_team.yellow_card_times = []
        self.blue_team.red_cards = 0
        self.blue_team.ball_placement_failures = 0
        self.blue_team.can_place_ball = True
        self.blue_team.max_allowed_bots = self.n_robots_blue

        # Reset game state
        self.ball_last_touched_by = None
        self.ball_placement_target = None
        self.next_command = None
        self.goal_scored_by = None

        # Reset event detection state
        self.last_ball_position = None
        self.last_goal_time = 0.0

        # Reset Phase 2 command transition timers
        self.stop_command_start_time = None
        self.kickoff_prep_start_time = None
        self.free_kick_start_time = None
        self.action_timer_start = None
        self.action_timer_duration = None

        # Reset Phase 3 foul detection state
        self.last_defense_area_violation_time = 0.0
        self.last_collision_time = 0.0
        self.ball_holder_team = None
        self.ball_hold_start_time = None
        self.ball_placement_start_time = None
        self.placing_team = None

        logger.info("Referee state machine reset to initial conditions")

    # ========== PHASE 2: COMMAND SEQUENCING METHODS ==========

    def _update_command_transitions(self, frame: Frame, current_time: float):
        """Handle automatic command transitions with timeouts.

        Args:
            frame: Current simulation frame
            current_time: Current simulation time
        """
        # STOP → PREPARE_KICKOFF (after 2s delay)
        if self.command == RefereeCommand.STOP:
            if self.stop_command_start_time is None:
                self.stop_command_start_time = current_time
            elif current_time - self.stop_command_start_time >= STOP_TO_KICKOFF_DELAY:
                if self.next_command:
                    self.command = self.next_command
                    self.command_counter += 1
                    self.command_timestamp = current_time
                    self.kickoff_prep_start_time = current_time
                    self.stop_command_start_time = None
                    self.next_command = RefereeCommand.NORMAL_START
                    logger.info("Command transition: STOP → %s", self.command.name)

        # PREPARE_KICKOFF → NORMAL_START (after 10s or when ready)
        elif self.command in [RefereeCommand.PREPARE_KICKOFF_YELLOW, RefereeCommand.PREPARE_KICKOFF_BLUE]:
            if self.kickoff_prep_start_time is None:
                self.kickoff_prep_start_time = current_time

            elapsed = current_time - self.kickoff_prep_start_time
            if elapsed >= KICKOFF_PREP_TIMEOUT or self._is_kickoff_ready(frame):
                self.command = RefereeCommand.NORMAL_START
                self.command_counter += 1
                self.command_timestamp = current_time
                self.kickoff_prep_start_time = None
                logger.info("Command transition: PREPARE_KICKOFF → NORMAL_START")

        # FREE_KICK → NORMAL_START (after 5s timeout)
        elif self.command in [
            RefereeCommand.DIRECT_FREE_YELLOW,
            RefereeCommand.DIRECT_FREE_BLUE,
        ]:
            if self.free_kick_start_time is None:
                self.free_kick_start_time = current_time
            elif current_time - self.free_kick_start_time >= FREE_KICK_TIMEOUT:
                self.command = RefereeCommand.NORMAL_START
                self.command_counter += 1
                self.command_timestamp = current_time
                self.free_kick_start_time = None
                logger.info("Command transition: FREE_KICK → NORMAL_START")

    def _is_kickoff_ready(self, frame: Frame) -> bool:
        """Check if teams are ready for kickoff (simplified version).

        Args:
            frame: Current simulation frame

        Returns:
            True if ready for kickoff, False otherwise
        """
        # Simple readiness: ball is near center and stationary
        ball = frame.ball
        ball_near_center = np.sqrt(ball.x**2 + ball.y**2) < 0.2
        ball_stationary = np.sqrt(ball.v_x**2 + ball.v_y**2) < 0.05
        return ball_near_center and ball_stationary

    # ========== PHASE 2: BALL OUT DETECTION METHODS ==========

    def _is_ball_out_of_bounds(self, frame: Frame) -> bool:
        """Detect if ball has left the field.

        Args:
            frame: Current simulation frame

        Returns:
            True if ball is out of bounds, False otherwise
        """
        ball = frame.ball
        return (
            abs(ball.x) > self.field_half_length + BALL_OUT_MARGIN
            or abs(ball.y) > self.field_half_width + BALL_OUT_MARGIN
        )

    def _get_last_ball_toucher(self, frame: Frame) -> Optional[str]:
        """Determine which team last touched the ball.

        Args:
            frame: Current simulation frame

        Returns:
            Team color ("yellow" or "blue") or None if unclear
        """
        min_dist = float("inf")
        last_toucher = None

        for robot_id, robot in frame.robots_blue.items():
            dist = np.sqrt((robot.x - frame.ball.x) ** 2 + (robot.y - frame.ball.y) ** 2)
            if dist < min_dist:
                min_dist = dist
                last_toucher = "blue"

        for robot_id, robot in frame.robots_yellow.items():
            dist = np.sqrt((robot.x - frame.ball.x) ** 2 + (robot.y - frame.ball.y) ** 2)
            if dist < min_dist:
                min_dist = dist
                last_toucher = "yellow"

        return last_toucher if min_dist < BALL_TOUCH_THRESHOLD else None

    def _process_ball_out(self, frame: Frame, current_time: float):
        """Handle ball out of bounds event.

        Args:
            frame: Current simulation frame
            current_time: Current simulation time
        """
        last_toucher = self._get_last_ball_toucher(frame)

        if last_toucher == "yellow":
            self.command = RefereeCommand.DIRECT_FREE_BLUE
            self.ball_last_touched_by = "yellow"
        else:
            self.command = RefereeCommand.DIRECT_FREE_YELLOW
            self.ball_last_touched_by = "blue"

        self.command_counter += 1
        self.command_timestamp = current_time
        self.free_kick_start_time = current_time

        # Set ball placement position (simplified - just inside field)
        ball = frame.ball
        placement_x = np.clip(ball.x, -self.field_half_length + 0.2, self.field_half_length - 0.2)
        placement_y = np.clip(ball.y, -self.field_half_width + 0.2, self.field_half_width - 0.2)
        self.ball_placement_target = (placement_x, placement_y)

        logger.info("Ball out! Free kick awarded to %s", "blue" if last_toucher == "yellow" else "yellow")

    # ========== PHASE 2: STAGE ADVANCEMENT METHOD ==========

    def _check_stage_transition(self, current_time: float):
        """Check and handle automatic stage transitions.

        Args:
            current_time: Current simulation time
        """
        stage_time_left = self.stage_duration - (current_time - self.stage_start_time)

        if stage_time_left <= 0:
            # Determine next stage
            if self.stage == Stage.NORMAL_FIRST_HALF_PRE:
                new_stage = Stage.NORMAL_FIRST_HALF
            elif self.stage == Stage.NORMAL_FIRST_HALF:
                new_stage = Stage.NORMAL_HALF_TIME
            elif self.stage == Stage.NORMAL_HALF_TIME:
                new_stage = Stage.NORMAL_SECOND_HALF_PRE
            elif self.stage == Stage.NORMAL_SECOND_HALF_PRE:
                new_stage = Stage.NORMAL_SECOND_HALF
            elif self.stage == Stage.NORMAL_SECOND_HALF:
                new_stage = Stage.POST_GAME
            else:
                return  # No transition

            self.advance_stage(new_stage, current_time)
            logger.info("Stage advanced to %s", new_stage.name)

    # ========== PHASE 3: DEFENSE AREA VIOLATION DETECTION ==========

    def _is_in_defense_area(self, robot_x: float, robot_y: float, defending_team: str) -> bool:
        """Check if a position is inside a defense area.

        Args:
            robot_x: Robot x position
            robot_y: Robot y position
            defending_team: Team that defends this area ("yellow" or "blue")

        Returns:
            True if position is inside defense area, False otherwise
        """
        # SSL: Defense area extends 1.0m from goal line, 2.0m wide
        defense_depth = 1.0  # meters from goal line
        defense_half_width = 1.0  # half width of defense area

        if defending_team == "yellow":
            # Yellow defends left goal (negative x)
            in_x_range = robot_x < -self.field_half_length + defense_depth
            in_y_range = abs(robot_y) < defense_half_width
            return in_x_range and in_y_range
        else:
            # Blue defends right goal (positive x)
            in_x_range = robot_x > self.field_half_length - defense_depth
            in_y_range = abs(robot_y) < defense_half_width
            return in_x_range and in_y_range

    def _detect_defense_area_violations(self, frame: Frame, current_time: float) -> Optional[str]:
        """Detect robots illegally in defense area.

        Args:
            frame: Current simulation frame
            current_time: Current simulation time

        Returns:
            Fouling team color or None
        """
        # Only check during active play
        if self.command != RefereeCommand.NORMAL_START:
            return None

        # Cooldown to prevent duplicate detections
        if current_time - self.last_defense_area_violation_time < DEFENSE_AREA_VIOLATION_COOLDOWN:
            return None

        # Check yellow robots in blue defense area
        for robot_id, robot in frame.robots_yellow.items():
            if self._is_in_defense_area(robot.x, robot.y, "blue"):
                # Check if robot has the ball (allowed)
                if not robot.infrared:
                    return "yellow"

        # Check blue robots in yellow defense area
        for robot_id, robot in frame.robots_blue.items():
            if self._is_in_defense_area(robot.x, robot.y, "yellow"):
                if not robot.infrared:
                    return "blue"

        return None

    def _process_defense_area_violation(self, fouling_team: str, current_time: float):
        """Handle defense area violation.

        Args:
            fouling_team: Team that committed the foul
            current_time: Current simulation time
        """
        # Award free kick to defending team
        if fouling_team == "yellow":
            self.command = RefereeCommand.DIRECT_FREE_BLUE
            self.yellow_team.foul_counter += 1
        else:
            self.command = RefereeCommand.DIRECT_FREE_YELLOW
            self.blue_team.foul_counter += 1

        self.command_counter += 1
        self.command_timestamp = current_time
        self.free_kick_start_time = current_time
        self.last_defense_area_violation_time = current_time

        logger.info("Defense area violation by %s", fouling_team)

    # ========== PHASE 3: ROBOT COLLISION DETECTION ==========

    def _detect_robot_collisions(self, frame: Frame, current_time: float) -> Optional[str]:
        """Detect high-speed robot collisions.

        Args:
            frame: Current simulation frame
            current_time: Current simulation time

        Returns:
            Fouling team color or None
        """
        if self.command != RefereeCommand.NORMAL_START:
            return None

        if current_time - self.last_collision_time < COLLISION_COOLDOWN:
            return None

        # Check all robot pairs
        for yellow_id, yellow_robot in frame.robots_yellow.items():
            for blue_id, blue_robot in frame.robots_blue.items():
                # Calculate distance
                dx = yellow_robot.x - blue_robot.x
                dy = yellow_robot.y - blue_robot.y
                distance = np.sqrt(dx**2 + dy**2)

                if distance < COLLISION_DISTANCE_THRESHOLD:
                    # Calculate velocities
                    yellow_speed = np.sqrt(yellow_robot.v_x**2 + yellow_robot.v_y**2)
                    blue_speed = np.sqrt(blue_robot.v_x**2 + blue_robot.v_y**2)

                    # High-speed collision detected
                    if yellow_speed > COLLISION_VELOCITY_THRESHOLD or blue_speed > COLLISION_VELOCITY_THRESHOLD:
                        # Blame the faster robot (simplified)
                        return "yellow" if yellow_speed > blue_speed else "blue"

        return None

    def _process_collision(self, fouling_team: str, current_time: float):
        """Handle robot collision.

        Args:
            fouling_team: Team at fault
            current_time: Current simulation time
        """
        self.command = RefereeCommand.STOP
        self.command_counter += 1
        self.command_timestamp = current_time
        self.last_collision_time = current_time

        # Increment foul counter
        if fouling_team == "yellow":
            self.yellow_team.foul_counter += 1
            self.next_command = RefereeCommand.DIRECT_FREE_BLUE
        else:
            self.blue_team.foul_counter += 1
            self.next_command = RefereeCommand.DIRECT_FREE_YELLOW

        logger.info("Collision detected, %s at fault", fouling_team)

    # ========== PHASE 3: BALL HOLDING DETECTION ==========

    def _detect_ball_holding(self, frame: Frame, current_time: float) -> Optional[str]:
        """Track ball possession and detect holding violations.

        Args:
            frame: Current simulation frame
            current_time: Current simulation time

        Returns:
            Fouling team color or None
        """
        if self.command != RefereeCommand.NORMAL_START:
            return None

        # Find robot with ball
        current_holder = None
        for robot_id, robot in frame.robots_yellow.items():
            if robot.infrared:
                current_holder = "yellow"
                break

        if not current_holder:
            for robot_id, robot in frame.robots_blue.items():
                if robot.infrared:
                    current_holder = "blue"
                    break

        # Update tracking
        if current_holder != self.ball_holder_team:
            # Possession changed
            self.ball_holder_team = current_holder
            self.ball_hold_start_time = current_time if current_holder else None
        elif current_holder and self.ball_hold_start_time:
            # Check if held too long
            hold_duration = current_time - self.ball_hold_start_time
            if hold_duration > MAX_BALL_HOLD_TIME:
                return current_holder

        return None

    def _process_ball_holding_violation(self, fouling_team: str, current_time: float):
        """Handle ball holding violation.

        Args:
            fouling_team: Team holding ball too long
            current_time: Current simulation time
        """
        # Award indirect free kick (but using DIRECT_FREE for simplicity)
        if fouling_team == "yellow":
            self.command = RefereeCommand.DIRECT_FREE_BLUE
            self.yellow_team.foul_counter += 1
        else:
            self.command = RefereeCommand.DIRECT_FREE_YELLOW
            self.blue_team.foul_counter += 1

        self.command_counter += 1
        self.command_timestamp = current_time
        self.free_kick_start_time = current_time

        # Reset tracking
        self.ball_holder_team = None
        self.ball_hold_start_time = None

        logger.info("Ball holding violation by %s", fouling_team)

    # ========== PHASE 3: BALL PLACEMENT (SIMPLIFIED) ==========

    def _check_ball_placement_progress(self, frame: Frame, current_time: float):
        """Monitor ball placement progress.

        Args:
            frame: Current simulation frame
            current_time: Current simulation time
        """
        if self.command not in [RefereeCommand.BALL_PLACEMENT_YELLOW, RefereeCommand.BALL_PLACEMENT_BLUE]:
            return

        if not self.ball_placement_target:
            return

        # Check if ball is at target
        ball = frame.ball
        target_x, target_y = self.ball_placement_target
        distance = np.sqrt((ball.x - target_x) ** 2 + (ball.y - target_y) ** 2)

        if distance < BALL_PLACEMENT_SUCCESS_DISTANCE:
            # Success - resume play
            self.command = self.next_command if self.next_command else RefereeCommand.NORMAL_START
            self.command_counter += 1
            self.command_timestamp = current_time
            self.ball_placement_target = None
            self.ball_placement_start_time = None
            logger.info("Ball placement successful")
        elif current_time - self.ball_placement_start_time > BALL_PLACEMENT_TIMEOUT:
            # Timeout - placement failed
            self._handle_placement_failure(current_time)

    def _handle_placement_failure(self, current_time: float):
        """Handle ball placement failure.

        Args:
            current_time: Current simulation time
        """
        if self.placing_team == "yellow":
            self.yellow_team.ball_placement_failures += 1
            if self.yellow_team.ball_placement_failures >= MAX_PLACEMENT_FAILURES:
                self.yellow_team.can_place_ball = False
                self.yellow_team.ball_placement_failures_reached = True
        else:
            self.blue_team.ball_placement_failures += 1
            if self.blue_team.ball_placement_failures >= MAX_PLACEMENT_FAILURES:
                self.blue_team.can_place_ball = False
                self.blue_team.ball_placement_failures_reached = True

        # Award ball to opponent
        self.command = RefereeCommand.FORCE_START
        self.command_counter += 1
        self.command_timestamp = current_time
        self.ball_placement_target = None
        self.ball_placement_start_time = None

        logger.info("Ball placement failed for %s", self.placing_team)

    # ========== PHASE 3: YELLOW/RED CARD SYSTEM ==========

    def _check_and_issue_cards(self, current_time: float):
        """Check foul counters and issue yellow/red cards.

        Args:
            current_time: Current simulation time
        """
        # Check yellow team
        if self.yellow_team.foul_counter >= FOULS_FOR_YELLOW_CARD:
            self._issue_yellow_card("yellow", current_time)
            self.yellow_team.foul_counter = 0  # Reset after card

        # Check blue team
        if self.blue_team.foul_counter >= FOULS_FOR_YELLOW_CARD:
            self._issue_yellow_card("blue", current_time)
            self.blue_team.foul_counter = 0

        # Check for red cards (3 yellow cards)
        if self.yellow_team.yellow_cards >= YELLOW_CARDS_FOR_RED:
            self._issue_red_card("yellow")

        if self.blue_team.yellow_cards >= YELLOW_CARDS_FOR_RED:
            self._issue_red_card("blue")

    def _issue_yellow_card(self, team: str, current_time: float):
        """Issue a yellow card to a team.

        Args:
            team: Team receiving yellow card
            current_time: Current simulation time
        """
        if team == "yellow":
            self.yellow_team.yellow_cards += 1
            self.yellow_team.yellow_card_times.append(int(current_time * 1000))  # microseconds
            # Temporarily reduce allowed robots
            self.yellow_team.max_allowed_bots = max(0, self.yellow_team.max_allowed_bots - 1)
        else:
            self.blue_team.yellow_cards += 1
            self.blue_team.yellow_card_times.append(int(current_time * 1000))
            self.blue_team.max_allowed_bots = max(0, self.blue_team.max_allowed_bots - 1)

        logger.warning("Yellow card issued to %s", team)

    def _issue_red_card(self, team: str):
        """Issue a red card (permanent robot removal).

        Args:
            team: Team receiving red card
        """
        if team == "yellow":
            self.yellow_team.red_cards += 1
            self.yellow_team.max_allowed_bots = max(0, self.yellow_team.max_allowed_bots - 1)
        else:
            self.blue_team.red_cards += 1
            self.blue_team.max_allowed_bots = max(0, self.blue_team.max_allowed_bots - 1)

        logger.error("Red card issued to %s", team)

    def _update_yellow_card_timers(self, current_time: float):
        """Decrement yellow card times and restore robots when expired.

        Args:
            current_time: Current simulation time
        """
        # Update yellow team cards
        expired_cards = 0
        for i, card_time in enumerate(self.yellow_team.yellow_card_times):
            elapsed = (current_time * 1000) - card_time  # to microseconds
            if elapsed >= YELLOW_CARD_TIME * 1000:
                expired_cards += 1

        if expired_cards > 0:
            # Remove expired cards
            self.yellow_team.yellow_card_times = self.yellow_team.yellow_card_times[expired_cards:]
            # Restore robots
            self.yellow_team.max_allowed_bots = min(
                self.n_robots_yellow, self.yellow_team.max_allowed_bots + expired_cards
            )

        # Update blue team cards (similar logic)
        expired_cards = 0
        for i, card_time in enumerate(self.blue_team.yellow_card_times):
            elapsed = (current_time * 1000) - card_time
            if elapsed >= YELLOW_CARD_TIME * 1000:
                expired_cards += 1

        if expired_cards > 0:
            self.blue_team.yellow_card_times = self.blue_team.yellow_card_times[expired_cards:]
            self.blue_team.max_allowed_bots = min(self.n_robots_blue, self.blue_team.max_allowed_bots + expired_cards)
