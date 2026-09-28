"""GameStateMachine: owns all mutable game state for the CustomReferee."""

from __future__ import annotations

import copy
import logging
import math
from typing import Optional

from utama_core.config.field_params import STANDARD_FIELD_DIMS
from utama_core.config.referee_constants import PENALTY_MARK_HALF_FIELD_RATIO
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
# Seconds a STOP can wait on _all_robots_clear() before advancing anyway. A
# real GC operator would eventually force it through if a robot never backs
# off; without this, one robot that fails to clear the ball (whatever the
# root cause -- e.g. a motion-planner convergence stall near a stationary
# ball) freezes the STOP forever, since _all_robots_clear() has no other way
# to become true. Found live, 2026-09-01: a full_match_tournament.py replay
# (counter_press_vs_tiki_taka_RK.pkl) froze at t=58s on exactly this and
# never played another tick for the remaining 540s of a 600s match -- caught
# by the stuck-window detector, not by score (it stayed a correct 0-0, just
# a dead one).
_STOP_CLEAR_TIMEOUT_SECONDS = 15.0
# Seconds BALL_PLACEMENT_* can wait on _ball_placement_done() before the
# state machine gives up and advances anyway, marking the placement failed.
# Mirrors _STOP_CLEAR_TIMEOUT_SECONDS's rationale exactly: without this, a
# placement target the placer can never actually reach (e.g. genuinely
# outside the robot-reachable area, or blocked by an opponent camped on it)
# is an unbounded wait rather than a recoverable event, since
# _ball_placement_done() gates purely on ball-to-target distance with no
# other way to become true. A real GC operator/rulebook (SSL rules §5.3.3)
# would eventually rule the placement failed and hand the restart to the
# other team; this is the sim-tournament equivalent, done automatically.
# Comfortably under _STOP_CLEAR_TIMEOUT_SECONDS so a placement failure is
# never mistaken for the slower STOP-clear case, and under any watchdog's
# own stall-detection window so this always fires first.
_BALL_PLACEMENT_TIMEOUT_SECONDS = 10.0


class GameStateMachine:
    """Owns score, command, and stage.  Produces ``RefereeData`` each tick."""

    def __init__(
        self,
        half_duration_seconds: float,
        kickoff_team: str,
        n_robots_yellow: int,
        n_robots_blue: int,
        initial_stage: Stage = Stage.NORMAL_FIRST_HALF_PRE,
        force_start_after_goal: bool = False,
        stop_duration_seconds: float = 3.0,
        prepare_duration_seconds: float = 3.0,
        kickoff_timeout_seconds: float = 10.0,
        geometry: Optional[RefereeGeometry] = None,
        auto_advance: Optional[AutoAdvanceConfig] = None,
    ) -> None:
        # Config, fixed for the lifetime of this instance — re-applied by
        # reset() but never itself reset.
        self._half_duration_seconds = half_duration_seconds
        self._kickoff_team = kickoff_team
        self._n_robots_yellow = n_robots_yellow
        self._n_robots_blue = n_robots_blue
        self._initial_stage = initial_stage
        self._force_start_after_goal = force_start_after_goal
        self._stop_duration_seconds = stop_duration_seconds
        self._prepare_duration_seconds = prepare_duration_seconds
        self._kickoff_timeout_seconds = kickoff_timeout_seconds
        self._geometry: Optional[RefereeGeometry] = geometry
        self._auto_advance = auto_advance if auto_advance is not None else AutoAdvanceConfig()

        self._reset_state()

    def _reset_state(self) -> None:
        """(Re-)initialise all per-episode mutable state to its starting values.

        Called once from `__init__` and again from `reset()` — kept as a
        single method so the two can never drift apart. Config passed to
        `__init__` (durations, robot counts, kickoff team, geometry,
        auto-advance flags) is untouched here; only state that changes over
        the course of a game is reset.
        """
        self.command = RefereeCommand.HALT
        self.command_counter = 0
        self.command_timestamp = 0.0

        self.stage = self._initial_stage
        # Seeded by seed_clock() after the first valid game frame is available.
        self.stage_start_time: Optional[float] = None
        self.stage_duration = self._half_duration_seconds

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
            max_allowed_bots=self._n_robots_yellow,
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
            max_allowed_bots=self._n_robots_blue,
            bot_substitution_intent=False,
            bot_substitution_allowed=True,
            bot_substitutions_left=5,
        )

        self.next_command: Optional[RefereeCommand] = None
        self.ball_placement_target: Optional[tuple[float, float]] = None
        self._post_ball_placement_command: Optional[RefereeCommand] = None
        self.status_message: Optional[str] = None

        # Kickoff team initialised from profile.
        self._kickoff_team_is_yellow = self._kickoff_team.lower() == "yellow"

        # Arcade auto-advance: after stop_duration_seconds in STOP following a
        # goal, automatically issue FORCE_START instead of waiting for operator.
        self._stop_entered_time: float = -math.inf  # wall time when STOP was last entered

        # Auto-advance timings.
        self._prepare_entered_time: float = -math.inf  # wall time when PREPARE_KICKOFF was entered
        self._normal_start_time: float = -math.inf  # wall time when NORMAL_START was entered

        # Ball position snapshot at NORMAL_START — used to detect if the ball has moved.
        self._ball_pos_at_normal_start: Optional[tuple[float, float]] = None

        # Timestamps for sustained-readiness countdown before play-starting advances.
        # Set to math.inf when condition is not yet met; fire when elapsed >= _AUTO_ADVANCE_DELAY.
        self._advance2_ready_since: float = math.inf  # PREPARE_* → NORMAL_START
        self._advance3_ready_since: float = math.inf  # DIRECT_FREE_* → NORMAL_START
        self._advance4_ready_since: float = math.inf  # BALL_PLACEMENT_* → next_command

        # Cooldown: don't process a new violation within this window.
        self._last_transition_time: float = -math.inf

    def reset(self) -> None:
        """Restore all per-episode state (score, command, stage, timers) to
        its starting values, for reuse across RL episodes without
        constructing a new `GameStateMachine`.

        Does not reset `_geometry` (set separately via `override_geometry`)
        or any of the config passed to `__init__` (durations, robot counts,
        kickoff team, auto-advance flags) — those describe the deployment,
        not the episode.
        """
        self._reset_state()

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
        if self.stage_start_time is None:
            self.stage_start_time = current_time

        if violation is not None and self._can_transition(current_time):
            self._apply_violation(violation, current_time)

        # ----------------------------------------------------------------
        # Auto-advance 1: STOP → next queued restart
        # Fires when all robots are ≥ _BALL_CLEAR_DIST from the ball, or --
        # regardless -- once _STOP_CLEAR_TIMEOUT_SECONDS has elapsed, so a
        # single robot that never clears (e.g. stuck oscillating next to a
        # stationary ball) can't freeze the STOP forever.
        # ----------------------------------------------------------------
        if (
            self._auto_advance.stop_to_next_command
            and self.command == RefereeCommand.STOP
            and self.next_command in self._NEEDS_STOP_FIRST
            and game_frame is not None
            and (
                self._all_robots_clear(game_frame)
                or (current_time - self._stop_entered_time) >= _STOP_CLEAR_TIMEOUT_SECONDS
            )
        ):
            if not self._all_robots_clear(game_frame):
                logger.warning(
                    "STOP clear timeout (%.1fs) — auto-advancing STOP → %s despite an uncleared robot",
                    _STOP_CLEAR_TIMEOUT_SECONDS,
                    self.next_command.name,
                )
            else:
                logger.info("All robots clear — auto-advancing STOP → %s", self.next_command.name)
            self.command = self.next_command
            self.command_counter += 1
            self.command_timestamp = current_time
            if self.command in self._BALL_PLACEMENT_COMMANDS:
                self.next_command = self._post_ball_placement_command or RefereeCommand.NORMAL_START
                self._post_ball_placement_command = None
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
            timed_out = (
                not ready
                and self.next_command is not None
                and (current_time - self.command_timestamp) >= _BALL_PLACEMENT_TIMEOUT_SECONDS
            )
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
                    self._advance_past_ball_placement(current_time)
            elif timed_out:
                # The placer has had _BALL_PLACEMENT_TIMEOUT_SECONDS and still
                # hasn't gotten the ball within _PLACEMENT_DONE_DIST -- e.g. a
                # target the placer can never physically reach. Record the
                # failure (mirrors real-GC/SSL-rulebook §5.3.3 behaviour) and
                # advance anyway rather than waiting forever; see
                # _BALL_PLACEMENT_TIMEOUT_SECONDS's docstring.
                placing_team = (
                    self.yellow_team if self.command == RefereeCommand.BALL_PLACEMENT_YELLOW else self.blue_team
                )
                placing_team.ball_placement_failures = (placing_team.ball_placement_failures or 0) + 1
                placing_team.can_place_ball = False
                logger.warning(
                    "Ball placement timeout (%.1fs) — auto-advancing %s → %s despite an unplaced ball "
                    "(failure #%d for %s)",
                    _BALL_PLACEMENT_TIMEOUT_SECONDS,
                    self.command.name,
                    self.next_command.name,
                    placing_team.ball_placement_failures,
                    placing_team.name,
                )
                self._advance_past_ball_placement(current_time)
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
        r = self._geometry.center_circle_radius if self._geometry is not None else 0.5  # fallback for standalone use
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

        half_length = (
            self._geometry.half_length if self._geometry is not None else STANDARD_FIELD_DIMS.full_field_half_length
        )
        yellow_is_right = game_frame.my_team_is_right == game_frame.my_team_is_yellow
        if kicking_is_yellow:
            goal_sign = -1.0 if yellow_is_right else 1.0
        else:
            goal_sign = 1.0 if yellow_is_right else -1.0
        penalty_mark_x = goal_sign * half_length * PENALTY_MARK_HALF_FIELD_RATIO

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

    def _advance_past_ball_placement(self, current_time: float) -> None:
        """Shared BALL_PLACEMENT_* → next_command transition for Auto-advance
        4's two exits (placement genuinely completed, or timed out per
        _BALL_PLACEMENT_TIMEOUT_SECONDS) — both hand off to whatever restart
        was already queued in exactly the same way; only the caller's log
        message and TeamInfo bookkeeping differ.
        """
        completed_command = self.next_command
        self.command = completed_command
        self.command_counter += 1
        self.command_timestamp = current_time
        if completed_command in self._DIRECT_FREE_COMMANDS:
            self.next_command = RefereeCommand.NORMAL_START
            self._advance3_ready_since = math.inf
        elif completed_command in self._PREPARE_KICKOFF_COMMANDS or completed_command in self._PREPARE_PENALTY_COMMANDS:
            self.next_command = RefereeCommand.NORMAL_START
            self._prepare_entered_time = current_time
            self._advance2_ready_since = math.inf
        else:
            self.next_command = None
        self._advance4_ready_since = math.inf
        self.status_message = None
        self._last_transition_time = current_time

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

    @staticmethod
    def _ball_placement_command_for(restart_command: Optional[RefereeCommand]) -> Optional[RefereeCommand]:
        """Return the ball-placement command for the team that owns a restart."""
        if restart_command in (
            RefereeCommand.PREPARE_KICKOFF_YELLOW,
            RefereeCommand.PREPARE_PENALTY_YELLOW,
            RefereeCommand.DIRECT_FREE_YELLOW,
            RefereeCommand.INDIRECT_FREE_YELLOW,
        ):
            return RefereeCommand.BALL_PLACEMENT_YELLOW
        if restart_command in (
            RefereeCommand.PREPARE_KICKOFF_BLUE,
            RefereeCommand.PREPARE_PENALTY_BLUE,
            RefereeCommand.DIRECT_FREE_BLUE,
            RefereeCommand.INDIRECT_FREE_BLUE,
        ):
            return RefereeCommand.BALL_PLACEMENT_BLUE
        return None

    def seed_clock(self, timestamp: float) -> None:
        """Align all internal timers to *timestamp*.

        Resets every timer that was initialised before the real vision clock
        was known (construction time) so that durations are measured from the
        correct timebase.  Safe to call only once, immediately after the first
        valid game frame is available.
        """
        if self.stage_start_time is None:
            self.stage_start_time = timestamp
        self.command_timestamp = timestamp
        # Push all "entered-at" sentinels to the new timebase so that
        # cooldowns / durations are not immediately satisfied.
        self._last_transition_time = timestamp - _TRANSITION_COOLDOWN  # allow transitions immediately
        self._stop_entered_time = timestamp - self._stop_duration_seconds  # won't auto-advance yet
        self._prepare_entered_time = timestamp - self._prepare_duration_seconds  # same

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
            self._post_ball_placement_command = None
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
            if self.command in self._BALL_PLACEMENT_COMMANDS:
                self.next_command = self._post_ball_placement_command or RefereeCommand.NORMAL_START
                self._post_ball_placement_command = None
                self._advance4_ready_since = math.inf
            else:
                self.next_command = RefereeCommand.NORMAL_START
            self.status_message = None
            self._prepare_entered_time = timestamp
            return

        self.command = command
        self.command_counter += 1
        self.command_timestamp = timestamp
        self._post_ball_placement_command = None
        self.status_message = None
        self._advance2_ready_since = math.inf
        self._advance3_ready_since = math.inf
        self._advance4_ready_since = math.inf

        if command in (RefereeCommand.STOP, RefereeCommand.HALT):
            self._stop_entered_time = timestamp
        elif command in (
            RefereeCommand.PREPARE_KICKOFF_YELLOW,
            RefereeCommand.PREPARE_KICKOFF_BLUE,
            RefereeCommand.PREPARE_PENALTY_YELLOW,
            RefereeCommand.PREPARE_PENALTY_BLUE,
        ):
            self._prepare_entered_time = timestamp

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

    def force_command(
        self,
        command: RefereeCommand,
        timestamp: float,
        ball_placement_target: Optional[tuple[float, float]] = None,
    ) -> None:
        """Directly set the command, bypassing the STOP-first guard.

        For god-mode / test use only — skips the normal safety interlock that
        inserts STOP before set-piece commands.
        """
        self.command = command
        self.command_counter += 1
        self.command_timestamp = timestamp
        self._post_ball_placement_command = None
        self.status_message = None
        self._advance2_ready_since = math.inf
        self._advance3_ready_since = math.inf
        self._advance4_ready_since = math.inf
        if ball_placement_target is not None:
            self.ball_placement_target = ball_placement_target
        if command in self._BALL_PLACEMENT_COMMANDS:
            # Auto-advance 4 requires next_command to be set.
            self.next_command = RefereeCommand.NORMAL_START
        else:
            self.next_command = None
        logger.info("Referee command force-set to: %s", command.name)

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

        # A non-stopping foul (§8.4.2 — "the game continues normally") makes
        # no command transition at all, so it must not consume the
        # transition cooldown either: doing so would wrongly block a real
        # transition (a goal, a stopping foul) detected up to
        # _TRANSITION_COOLDOWN seconds later for no reason connected to it.
        if violation.is_stopping:
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
        self.next_command = self._ball_placement_command_for(violation.next_command) or violation.next_command
        self._post_ball_placement_command = (
            violation.next_command if self.next_command in self._BALL_PLACEMENT_COMMANDS else None
        )
        self.ball_placement_target = (0.0, 0.0)
        self.status_message = violation.status_message
        self._stop_entered_time = current_time

    def _handle_foul(self, violation: RuleViolation, current_time: float) -> None:
        if violation.counts_toward_foul_counter:
            for is_yellow in violation.offending_teams:
                team = self.yellow_team if is_yellow else self.blue_team
                if team.increment_foul_counter():
                    logger.info("Yellow card: %s (3rd foul: %s)", team.name, violation.rule_name)

        if not violation.is_stopping:
            # SSL rulebook §8.4.2 "non-stopping foul": the foul-counter/card
            # side effect above is the entire response — the state machine
            # must NOT touch command/command_counter/next_command, or "the
            # game continues normally" would be violated (a spurious
            # command_counter bump reads as a real transition to every
            # other piece of code that watches it, e.g. rule.reset()).
            logger.info("Non-stopping foul detected: %s", violation.rule_name)
            return

        self.command = violation.suggested_command
        self.command_counter += 1
        self.command_timestamp = current_time
        if self.command == RefereeCommand.STOP:
            # Unlike _handle_goal and force_command, this path never recorded
            # when STOP was entered -- _stop_entered_time was left stale from
            # whatever STOP (or none) came before, which would have made the
            # new STOP-clear timeout above fire immediately (or never track
            # correctly) for every foul-triggered STOP, e.g. an out-of-bounds
            # restart. Found while adding that timeout, 2026-09-01.
            self._stop_entered_time = current_time
        placement_command = (
            self._ball_placement_command_for(violation.next_command)
            if violation.designated_position is not None
            else None
        )
        # A violation with no next_command/designated_position of its own
        # (e.g. DefenseAreaStoppageRule's HALT escalation) is not saying "the
        # restart in flight is cancelled" -- on a real pitch a HALT here just
        # means a human ref decides what happens next, and the previously
        # queued restart (e.g. a goal's PREPARE_KICKOFF_*/ball_placement_
        # target) is still the right thing to resume once play continues.
        # Overwriting these with None here previously erased that restart
        # entirely: a goal's queued kickoff got discarded the instant a
        # HALT-escalating foul fired before the restart's own STOP could
        # auto-advance, and since sim mode has no human to resume a HALT
        # (see StrategyRunner's _SIM_HALT_AUTO_RESUME_SECONDS), the auto-
        # resume had nothing left to restore into -- it jumped straight to
        # NORMAL_START with the ball still sitting wherever the interrupted
        # restart left it (e.g. still in the goal mouth right after a goal),
        # which let GoalRule immediately re-fire and repeat the whole cycle
        # every ~9s for the rest of the match (found live, tournament
        # replay counter_flow_vs_zone_fluid_LK.pkl, 2026-09-01: one real
        # goal at t=43.6s, then 23 more "goals" every ~9s from t=394s on,
        # via this exact STOP(goal)->HALT(2nd defense-area-stoppage foul)->
        # NORMAL_START(force_command, no teleport)->STOP(goal) loop).
        # Preserving instead of clobbering only changes behaviour for a
        # violation that provides nothing new -- every rule that has a real
        # next_command/designated_position to give still overrides freely.
        if violation.next_command is not None:
            self.next_command = placement_command or violation.next_command
            self._post_ball_placement_command = violation.next_command if placement_command is not None else None
        if violation.designated_position is not None:
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
