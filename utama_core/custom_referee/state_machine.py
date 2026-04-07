"""GameStateMachine: owns all mutable game state for the CustomReferee."""

from __future__ import annotations

import copy
import logging
import math
import time
from typing import Optional

from utama_core.custom_referee.geometry import RefereeGeometry
from utama_core.custom_referee.profiles.profile_loader import AutoAdvanceConfig
from utama_core.custom_referee.rules.base_rule import RuleViolation
from utama_core.entities.data.referee import RefereeData
from utama_core.entities.game.game_frame import GameFrame
from utama_core.entities.game.team_info import TeamInfo
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.entities.referee.stage import Stage

logger = logging.getLogger(__name__)

_TRANSITION_COOLDOWN = 0.3  # seconds — prevents command oscillation
_BALL_CLEAR_DIST = 0.5  # metres — all robots must be this far from ball before queued restart
_KICKER_READY_DIST = 0.3  # metres — kicker must be within this distance to trigger free kick start
_PLACEMENT_DONE_DIST = 0.15  # metres — ball within this dist of target → placement complete
_AUTO_ADVANCE_DELAY = 2.0  # seconds — readiness must be sustained this long before play starts


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
        prepare_duration_seconds: float = 3.0,
        kickoff_timeout_seconds: float = 10.0,
        geometry: Optional[RefereeGeometry] = None,
        auto_advance: Optional[AutoAdvanceConfig] = None,
        initial_time: Optional[float] = None,
    ) -> None:
        self.command = initial_command
        self.command_counter = 0
        self.command_timestamp = 0.0

        self.stage = initial_stage
        self.stage_start_time = (
            time.time() if initial_time is None else initial_time
        )
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
        self.status_message: Optional[str] = None

        # Kickoff team initialised from profile.
        self._kickoff_team_is_yellow = kickoff_team.lower() == "yellow"

        # Arcade auto-advance: after stop_duration_seconds in STOP following a
        # goal, automatically issue FORCE_START instead of waiting for operator.
        self._force_start_after_goal = force_start_after_goal
        self._stop_duration_seconds = stop_duration_seconds
        self._stop_entered_time: float = -math.inf  # wall time when STOP was last entered

        # Auto-advance timings.
        self._prepare_duration_seconds = prepare_duration_seconds
        self._kickoff_timeout_seconds = kickoff_timeout_seconds
        self._prepare_entered_time: float = -math.inf  # wall time when PREPARE_KICKOFF was entered
        self._normal_start_time: float = -math.inf  # wall time when NORMAL_START was entered

        # Ball position snapshot at NORMAL_START — used to detect if the ball has moved.
        self._ball_pos_at_normal_start: Optional[tuple[float, float]] = None

        # Timestamps for sustained-readiness countdown before play-starting advances.
        # Set to math.inf when condition is not yet met; fire when elapsed >= _AUTO_ADVANCE_DELAY.
        self._advance2_ready_since: float = math.inf  # PREPARE_* → NORMAL_START
        self._advance3_ready_since: float = math.inf  # DIRECT_FREE_* → NORMAL_START
        self._advance4_ready_since: float = math.inf  # BALL_PLACEMENT_* → next_command

        # Field geometry (used for readiness checks).
        self._geometry: Optional[RefereeGeometry] = geometry

        # Cooldown: don't process a new violation within this window.
        self._last_transition_time: float = -math.inf

        # Per-transition enable flags (default: all on).
        self._auto_advance = auto_advance if auto_advance is not None else AutoAdvanceConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    _PREPARE_KICKOFF_COMMANDS = frozenset(
        {
            RefereeCommand.PREPARE_KICKOFF_YELLOW,
            RefereeCommand.PREPARE_KICKOFF_BLUE,
        }
    )
    _DIRECT_FREE_COMMANDS = frozenset(
        {
            RefereeCommand.DIRECT_FREE_YELLOW,
            RefereeCommand.DIRECT_FREE_BLUE,
        }
    )
    _PREPARE_PENALTY_COMMANDS = frozenset(
        {
            RefereeCommand.PREPARE_PENALTY_YELLOW,
            RefereeCommand.PREPARE_PENALTY_BLUE,
        }
    )
    _BALL_PLACEMENT_COMMANDS = frozenset(
        {
            RefereeCommand.BALL_PLACEMENT_YELLOW,
            RefereeCommand.BALL_PLACEMENT_BLUE,
        }
    )

    def step(
        self,
        current_time: float,
        violation: Optional[RuleViolation],
        game_frame: Optional["GameFrame"] = None,
    ) -> RefereeData:
        """Process one tick.  Apply violation if not in cooldown.  Return RefereeData."""
        if violation is not None and self._can_transition(current_time):
            self._apply_violation(violation, current_time)

        # ----------------------------------------------------------------
        # Auto-advance 1: STOP → next queued restart
        # Fires when all robots are ≥ _BALL_CLEAR_DIST from the ball.
        # ----------------------------------------------------------------
        if (
            self._auto_advance.stop_to_next_command
            and self.command == RefereeCommand.STOP
            and self.next_command in self._NEEDS_STOP_FIRST
            and game_frame is not None
            and self._all_robots_clear(game_frame)
        ):
            logger.info("All robots clear — auto-advancing STOP → %s", self.next_command.name)
            self.command = self.next_command
            self.command_counter += 1
            self.command_timestamp = current_time
            if self.command in self._BALL_PLACEMENT_COMMANDS:
                self.next_command = RefereeCommand.NORMAL_START
                self._advance4_ready_since = math.inf
            elif self.command in self._DIRECT_FREE_COMMANDS:
                self.next_command = RefereeCommand.NORMAL_START
                self._advance3_ready_since = math.inf
            elif self.command in self._PREPARE_KICKOFF_COMMANDS or self.command in self._PREPARE_PENALTY_COMMANDS:
                self.next_command = RefereeCommand.NORMAL_START
                self._prepare_entered_time = current_time
                self._advance2_ready_since = math.inf
            self._last_transition_time = current_time

        # ----------------------------------------------------------------
        # Auto-advance 2a: PREPARE_KICKOFF_* → NORMAL_START
        # Fires after prepare_duration_seconds AND one attacker is inside
        # the centre circle, sustained for _AUTO_ADVANCE_DELAY seconds.
        # ----------------------------------------------------------------
        elif self._auto_advance.prepare_kickoff_to_normal and self.command in self._PREPARE_KICKOFF_COMMANDS:
            ready = (
                (current_time - self._prepare_entered_time) >= self._prepare_duration_seconds
                and game_frame is not None
                and self._kicker_in_centre_circle(self.command, game_frame)
            )
            if ready:
                if self._advance2_ready_since == math.inf:
                    self._advance2_ready_since = current_time
                    logger.debug("Advance 2 countdown started (%s)", self.command.name)
                elif (current_time - self._advance2_ready_since) >= _AUTO_ADVANCE_DELAY:
                    logger.info(
                        "Kicker in centre circle — auto-advancing %s → NORMAL_START",
                        self.command.name,
                    )
                    self.command = RefereeCommand.NORMAL_START
                    self.command_counter += 1
                    self.command_timestamp = current_time
                    self.next_command = None
                    self.status_message = None
                    self._normal_start_time = current_time
                    self._ball_pos_at_normal_start = (
                        (game_frame.ball.p.x, game_frame.ball.p.y) if game_frame.ball is not None else None
                    )
                    self._advance2_ready_since = math.inf
                    self._last_transition_time = current_time
            else:
                self._advance2_ready_since = math.inf

        # ----------------------------------------------------------------
        # Auto-advance 2b: PREPARE_PENALTY_* → NORMAL_START
        # Fires after prepare_duration_seconds when the kicker reaches the
        # penalty mark, sustained for _AUTO_ADVANCE_DELAY seconds.
        # ----------------------------------------------------------------
        elif self._auto_advance.prepare_penalty_to_normal and self.command in self._PREPARE_PENALTY_COMMANDS:
            ready = (
                (current_time - self._prepare_entered_time) >= self._prepare_duration_seconds
                and game_frame is not None
                and self._penalty_kicker_ready(self.command, game_frame)
            )
            if ready:
                if self._advance2_ready_since == math.inf:
                    self._advance2_ready_since = current_time
                    logger.debug("Advance 2 countdown started (%s)", self.command.name)
                elif (current_time - self._advance2_ready_since) >= _AUTO_ADVANCE_DELAY:
                    logger.info(
                        "Kicker at penalty mark — auto-advancing %s → NORMAL_START",
                        self.command.name,
                    )
                    self.command = RefereeCommand.NORMAL_START
                    self.command_counter += 1
                    self.command_timestamp = current_time
                    self.next_command = None
                    self.status_message = None
                    self._normal_start_time = current_time
                    self._ball_pos_at_normal_start = (
                        (game_frame.ball.p.x, game_frame.ball.p.y) if game_frame.ball is not None else None
                    )
                    self._advance2_ready_since = math.inf
                    self._last_transition_time = current_time
            else:
                self._advance2_ready_since = math.inf

        # ----------------------------------------------------------------
        # Auto-advance 3: DIRECT_FREE_* → NORMAL_START
        # Fires when the kicker is within _KICKER_READY_DIST of the ball
        # AND all defending robots are ≥ _BALL_CLEAR_DIST away, sustained
        # for _AUTO_ADVANCE_DELAY seconds.
        # ----------------------------------------------------------------
        elif self._auto_advance.direct_free_to_normal and self.command in self._DIRECT_FREE_COMMANDS:
            ready = game_frame is not None and self._free_kick_ready(self.command, game_frame)
            if ready:
                if self._advance3_ready_since == math.inf:
                    self._advance3_ready_since = current_time
                    logger.debug("Advance 3 countdown started (%s)", self.command.name)
                elif (current_time - self._advance3_ready_since) >= _AUTO_ADVANCE_DELAY:
                    logger.info("Free kick ready — auto-advancing %s → NORMAL_START", self.command.name)
                    self.command = RefereeCommand.NORMAL_START
                    self.command_counter += 1
                    self.command_timestamp = current_time
                    self.next_command = None
                    self.status_message = None
                    self._normal_start_time = current_time
                    self._ball_pos_at_normal_start = (
                        (game_frame.ball.p.x, game_frame.ball.p.y) if game_frame.ball is not None else None
                    )
                    self._advance3_ready_since = math.inf
                    self._last_transition_time = current_time
            else:
                self._advance3_ready_since = math.inf

        # ----------------------------------------------------------------
        # Auto-advance 4: BALL_PLACEMENT_* → next_command
        # Fires when ball reaches within _PLACEMENT_DONE_DIST of target,
        # sustained for _AUTO_ADVANCE_DELAY seconds.
        # ----------------------------------------------------------------
        elif self._auto_advance.ball_placement_to_next and self.command in self._BALL_PLACEMENT_COMMANDS:
            ready = self.next_command is not None and game_frame is not None and self._ball_placement_done(game_frame)
            if ready:
                if self._advance4_ready_since == math.inf:
                    self._advance4_ready_since = current_time
                    logger.debug("Advance 4 countdown started (%s)", self.command.name)
                elif (current_time - self._advance4_ready_since) >= _AUTO_ADVANCE_DELAY:
                    logger.info(
                        "Ball placement complete — auto-advancing %s → %s",
                        self.command.name,
                        self.next_command.name,
                    )
                    self.command = self.next_command
                    self.command_counter += 1
                    self.command_timestamp = current_time
                    self.next_command = None
                    self._advance4_ready_since = math.inf
                    self.status_message = None
                    self._last_transition_time = current_time
            else:
                self._advance4_ready_since = math.inf

        # ----------------------------------------------------------------
        # Auto-advance 5: NORMAL_START → FORCE_START
        # Fires after kickoff_timeout_seconds if the ball hasn't moved ≥5 cm.
        # ----------------------------------------------------------------
        elif (
            self._auto_advance.normal_start_to_force
            and self.command == RefereeCommand.NORMAL_START
            and self._ball_pos_at_normal_start is not None
            and (current_time - self._normal_start_time) >= self._kickoff_timeout_seconds
            and game_frame is not None
            and game_frame.ball is not None
            and not self._ball_has_moved(game_frame)
        ):
            logger.info("Kickoff/free-kick timeout — auto-advancing NORMAL_START → FORCE_START")
            self.command = RefereeCommand.FORCE_START
            self.command_counter += 1
            self.command_timestamp = current_time
            self.next_command = None
            self.status_message = None
            self._ball_pos_at_normal_start = None
            self._last_transition_time = current_time

        # ----------------------------------------------------------------
        # Legacy force-start path: STOP → FORCE_START after goal
        # ----------------------------------------------------------------
        elif (
            self._force_start_after_goal
            and self.command == RefereeCommand.STOP
            and self.next_command in self._PREPARE_KICKOFF_COMMANDS
            and (current_time - self._stop_entered_time) >= self._stop_duration_seconds
        ):
            self.command = RefereeCommand.FORCE_START
            self.command_counter += 1
            self.command_timestamp = current_time
            self.next_command = None
            self.status_message = None
            self._last_transition_time = current_time
            logger.info("Auto-advanced STOP → FORCE_START after goal (force-start profile mode)")

        return self._generate_referee_data(current_time)

    def _all_robots_clear(self, game_frame: "GameFrame") -> bool:
        """Return True if every robot on both teams is ≥ _BALL_CLEAR_DIST from the ball."""
        ball = game_frame.ball
        if ball is None:
            return True
        bx, by = ball.p.x, ball.p.y
        for r in list(game_frame.friendly_robots.values()) + list(game_frame.enemy_robots.values()):
            if math.hypot(r.p.x - bx, r.p.y - by) < _BALL_CLEAR_DIST:
                return False
        return True

    def _kicker_in_centre_circle(self, command: RefereeCommand, game_frame: "GameFrame") -> bool:
        """Return True if at least one robot of the attacking team is inside the centre circle."""
        r = self._geometry.center_circle_radius if self._geometry is not None else 0.5
        kicking_is_yellow = command == RefereeCommand.PREPARE_KICKOFF_YELLOW
        attackers = (
            game_frame.friendly_robots if kicking_is_yellow == game_frame.my_team_is_yellow else game_frame.enemy_robots
        )
        return any(math.hypot(robot.p.x, robot.p.y) <= r for robot in attackers.values())

    def _penalty_kicker_ready(self, command: RefereeCommand, game_frame: "GameFrame") -> bool:
        """Return True when an attacking robot is within the ready radius of the penalty mark."""
        kicking_is_yellow = command == RefereeCommand.PREPARE_PENALTY_YELLOW
        attackers = (
            game_frame.friendly_robots if kicking_is_yellow == game_frame.my_team_is_yellow else game_frame.enemy_robots
        )
        if not attackers:
            return False

        half_length = self._geometry.half_length if self._geometry is not None else 4.5
        yellow_is_right = game_frame.my_team_is_right == game_frame.my_team_is_yellow
        if kicking_is_yellow:
            goal_sign = -1.0 if yellow_is_right else 1.0
        else:
            goal_sign = 1.0 if yellow_is_right else -1.0
        penalty_mark_x = goal_sign * half_length * 0.5

        closest = min(math.hypot(robot.p.x - penalty_mark_x, robot.p.y) for robot in attackers.values())
        return closest <= _KICKER_READY_DIST

    def _free_kick_ready(self, command: RefereeCommand, game_frame: "GameFrame") -> bool:
        """Return True when a free kick is ready to start:
        - The kicker (closest attacker to ball) is within _KICKER_READY_DIST of the ball.
        - All defending robots are ≥ _BALL_CLEAR_DIST from the ball.
        """
        ball = game_frame.ball
        if ball is None:
            return False
        bx, by = ball.p.x, ball.p.y

        kicking_is_yellow = command == RefereeCommand.DIRECT_FREE_YELLOW
        attackers = (
            game_frame.friendly_robots if kicking_is_yellow == game_frame.my_team_is_yellow else game_frame.enemy_robots
        )
        defenders = (
            game_frame.enemy_robots if kicking_is_yellow == game_frame.my_team_is_yellow else game_frame.friendly_robots
        )

        # Check defending robots are all clear.
        if any(math.hypot(r.p.x - bx, r.p.y - by) < _BALL_CLEAR_DIST for r in defenders.values()):
            return False

        # Check at least one attacker is close to the ball (kicker in position).
        if not attackers:
            return False
        closest = min(math.hypot(r.p.x - bx, r.p.y - by) for r in attackers.values())
        return closest <= _KICKER_READY_DIST

    def _ball_has_moved(self, game_frame: "GameFrame") -> bool:
        """Return True if the ball has moved ≥ 0.05 m since NORMAL_START."""
        if self._ball_pos_at_normal_start is None or game_frame.ball is None:
            return False
        ox, oy = self._ball_pos_at_normal_start
        return math.hypot(game_frame.ball.p.x - ox, game_frame.ball.p.y - oy) >= 0.05

    def _ball_placement_done(self, game_frame: "GameFrame") -> bool:
        """Return True when the ball is within _PLACEMENT_DONE_DIST of the placement target."""
        if self.ball_placement_target is None or game_frame.ball is None:
            return False
        tx, ty = self.ball_placement_target
        return math.hypot(game_frame.ball.p.x - tx, game_frame.ball.p.y - ty) <= _PLACEMENT_DONE_DIST

    # Commands that require robots to clear the ball before they take effect.
    # In a real match these are always preceded by STOP.
    _NEEDS_STOP_FIRST = frozenset(
        {
            RefereeCommand.PREPARE_KICKOFF_YELLOW,
            RefereeCommand.PREPARE_KICKOFF_BLUE,
            RefereeCommand.DIRECT_FREE_YELLOW,
            RefereeCommand.DIRECT_FREE_BLUE,
            RefereeCommand.PREPARE_PENALTY_YELLOW,
            RefereeCommand.PREPARE_PENALTY_BLUE,
            RefereeCommand.BALL_PLACEMENT_YELLOW,
            RefereeCommand.BALL_PLACEMENT_BLUE,
        }
    )

    def set_command(self, command: RefereeCommand, timestamp: float) -> None:
        """Manual override — for operator use or test scripting.

        If *command* is a set-piece command (kickoff, free kick, penalty,
        ball placement) and the game is not already in STOP or HALT, a STOP
        is issued first and the requested command is stored as ``next_command``
        so the operator (or a script) can advance to it after robots have
        cleared the ball.  This mirrors real-match game-controller behaviour
        and prevents robots from receiving a PREPARE_KICKOFF while they are
        still within the keep-out zone around the ball.
        """
        _ALREADY_STOPPED = (RefereeCommand.STOP, RefereeCommand.HALT)

        if command in self._NEEDS_STOP_FIRST and self.command not in _ALREADY_STOPPED:
            # Insert STOP; park the real command as next_command.
            logger.info("Inserting STOP before %s so robots can clear the ball", command.name)
            self.command = RefereeCommand.STOP
            self.command_counter += 1
            self.command_timestamp = timestamp
            self.next_command = command
            self.status_message = None
            self._stop_entered_time = timestamp
            return

        # NORMAL_START while in STOP with a pending set-piece: advance to the
        # set-piece first so robots can form up.  Auto-advance will then issue
        # NORMAL_START after prepare_duration_seconds.
        if (
            command == RefereeCommand.NORMAL_START
            and self.command == RefereeCommand.STOP
            and self.next_command in self._NEEDS_STOP_FIRST
        ):
            logger.info(
                "Manually advancing STOP → %s (auto NORMAL_START in %.1f s)",
                self.next_command.name,
                self._prepare_duration_seconds,
            )
            self.command = self.next_command
            self.command_counter += 1
            self.command_timestamp = timestamp
            self.next_command = RefereeCommand.NORMAL_START
            self.status_message = None
            self._prepare_entered_time = timestamp
            return

        self.command = command
        self.command_counter += 1
        self.command_timestamp = timestamp
        self.status_message = None
        self._advance2_ready_since = math.inf
        self._advance3_ready_since = math.inf
        self._advance4_ready_since = math.inf

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
        self.status_message = violation.status_message
        self._stop_entered_time = current_time

    def _handle_foul(self, violation: RuleViolation, current_time: float) -> None:
        self.command = violation.suggested_command
        self.command_counter += 1
        self.command_timestamp = current_time
        self.next_command = violation.next_command
        self.ball_placement_target = violation.designated_position
        self.status_message = violation.status_message
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
            blue_team=copy.copy(self.blue_team),
            yellow_team=copy.copy(self.yellow_team),
            designated_position=self.ball_placement_target,
            blue_team_on_positive_half=None,
            next_command=self.next_command,
            current_action_time_remaining=None,
            status_message=self.status_message,
        )
