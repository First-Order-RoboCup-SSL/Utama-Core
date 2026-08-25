"""CustomReferee: orchestrates rule checking and state management."""

from __future__ import annotations

from typing import List, Optional

from utama_core.config.field_params import STANDARD_FIELD_DIMS
from utama_core.custom_referee.geometry import RefereeGeometry
from utama_core.custom_referee.profiles.profile_loader import (
    RefereeProfile,
    load_profile,
)
from utama_core.custom_referee.rules.ball_placement_interference_rule import (
    BallPlacementInterferenceRule,
)
from utama_core.custom_referee.rules.ball_speed_rule import BallSpeedRule
from utama_core.custom_referee.rules.base_rule import BaseRule, RuleViolation
from utama_core.custom_referee.rules.crashing_rule import CrashingRule
from utama_core.custom_referee.rules.defense_area_rule import DefenseAreaRule
from utama_core.custom_referee.rules.defense_area_stoppage_rule import (
    DefenseAreaStoppageRule,
)
from utama_core.custom_referee.rules.double_touch_rule import DoubleTouchRule
from utama_core.custom_referee.rules.excessive_dribbling_rule import (
    ExcessiveDribblingRule,
)
from utama_core.custom_referee.rules.goal_rule import GoalRule
from utama_core.custom_referee.rules.keep_out_rule import KeepOutRule
from utama_core.custom_referee.rules.keeper_held_ball_rule import KeeperHeldBallRule
from utama_core.custom_referee.rules.out_of_bounds_rule import OutOfBoundsRule
from utama_core.custom_referee.rules.pushing_rule import PushingRule
from utama_core.custom_referee.rules.robot_stop_speed_rule import RobotStopSpeedRule
from utama_core.custom_referee.state_machine import GameStateMachine
from utama_core.entities.data.referee import RefereeData
from utama_core.entities.game.game_frame import GameFrame
from utama_core.entities.referee.referee_command import RefereeCommand


def _build_active_rules(rules_cfg) -> List[BaseRule]:
    """Construct the ordered list of active rules from a RulesConfig."""
    active: List[BaseRule] = []

    # Priority order: GoalRule → OutOfBoundsRule → BallSpeedRule → DoubleTouchRule
    #                 → DefenseAreaRule → KeepOutRule
    if rules_cfg.goal_detection.enabled:
        active.append(GoalRule(cooldown_seconds=rules_cfg.goal_detection.cooldown_seconds))

    if rules_cfg.out_of_bounds.enabled:
        active.append(OutOfBoundsRule())

    if rules_cfg.ball_speed.enabled:
        active.append(BallSpeedRule(max_speed_mps=rules_cfg.ball_speed.max_speed_mps))

    if rules_cfg.double_touch.enabled:
        active.append(DoubleTouchRule())

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

    # PushingRule is a stopping foul (§8.4.1) and so DOES compete for the
    # "first stopping violation wins" slot above — placed last among the
    # stopping-foul group since a sustained, force-asymmetric push is a
    # slower-forming condition (persistence_frames-gated) than any of the
    # single-tick checks above it, so it should never pre-empt one of them
    # firing on the same tick a push also happens to be building up.
    if rules_cfg.pushing.enabled:
        active.append(
            PushingRule(
                min_closing_speed_mps=rules_cfg.pushing.min_closing_speed_mps,
                similar_force_margin_mps=rules_cfg.pushing.similar_force_margin_mps,
                persistence_frames=rules_cfg.pushing.persistence_frames,
            )
        )

    # KeeperHeldBallRule / ExcessiveDribblingRule / RobotStopSpeedRule added
    # after the existing stopping-foul rules above: none of the three can
    # ever fire during the same tick as GoalRule/OutOfBoundsRule/BallSpeed/
    # DoubleTouch/DefenseArea/KeepOut's actual trigger conditions (different
    # command-gating and geometry), so their relative priority position
    # among themselves and the existing list doesn't change behaviour either
    # way — appended here to keep this diff additive.
    if rules_cfg.keeper_held_ball.enabled:
        active.append(KeeperHeldBallRule(max_hold_seconds=rules_cfg.keeper_held_ball.max_hold_seconds))

    if rules_cfg.excessive_dribbling.enabled:
        active.append(ExcessiveDribblingRule(max_dribble_meters=rules_cfg.excessive_dribbling.max_dribble_meters))

    if rules_cfg.robot_stop_speed.enabled:
        active.append(
            RobotStopSpeedRule(
                max_speed_mps=rules_cfg.robot_stop_speed.max_speed_mps,
                grace_seconds=rules_cfg.robot_stop_speed.grace_seconds,
            )
        )

    # CrashingRule is a non-stopping foul (`is_stopping=False`) — it never
    # competes for the "first stopping violation wins" slot (see
    # CustomReferee.step()), so its position in this list only matters
    # relative to other non-stopping rules, of which there currently are
    # none. Appended here to keep this diff additive, same rationale as the
    # three rules directly above.
    if rules_cfg.crashing.enabled:
        active.append(
            CrashingRule(
                fault_speed_threshold_mps=rules_cfg.crashing.fault_speed_threshold_mps,
                both_fault_threshold_mps=rules_cfg.crashing.both_fault_threshold_mps,
                retrigger_cooldown_seconds=rules_cfg.crashing.retrigger_cooldown_seconds,
            )
        )

    # DefenseAreaStoppageRule is a stoppage-time (STOP/free-kick) rule with
    # no active-play overlap with DefenseAreaRule above, and
    # BallPlacementInterferenceRule only ever fires during BALL_PLACEMENT_*
    # (which no other rule in this list checks at all) — appended here to
    # keep this diff additive, same rationale as the rules directly above.
    if rules_cfg.defense_area_stoppage.enabled:
        active.append(
            DefenseAreaStoppageRule(
                min_distance_meters=rules_cfg.defense_area_stoppage.min_distance_meters,
                grace_seconds=rules_cfg.defense_area_stoppage.grace_seconds,
            )
        )

    if rules_cfg.ball_placement_interference.enabled:
        active.append(
            BallPlacementInterferenceRule(
                stadium_radius_meters=rules_cfg.ball_placement_interference.stadium_radius_meters,
                grace_seconds=rules_cfg.ball_placement_interference.grace_seconds,
            )
        )

    return active


class CustomReferee:
    """Stateful referee that operates on ``GameFrame`` objects.

    Works uniformly across Real, grSim, and RSim modes because it does not
    depend on any network receiver or simulator-specific data.

    Usage::

        referee = CustomReferee.from_profile_name("simulation")
        ref_data = referee.step(game_frame, time.time())

    To surface state in the browser dashboard, attach it after construction::

        from utama_core.dashboard import attach_dashboard
        from utama_core.dashboard.views import referee as referee_view
        server = attach_dashboard()
        referee_view.attach(server, referee, profile)
    """

    def __init__(
        self,
        profile: RefereeProfile,
        n_robots_yellow: int = 3,
        n_robots_blue: int = 3,
    ) -> None:
        self._profile_name = profile.profile_name
        self._geometry: RefereeGeometry = RefereeGeometry.from_field_dims(
            STANDARD_FIELD_DIMS
        )  # this is overrriden by StrategyRunner
        self._rules: List[BaseRule] = _build_active_rules(profile.rules)
        self._state = GameStateMachine(
            half_duration_seconds=profile.game.half_duration_seconds,
            kickoff_team=profile.game.kickoff_team,
            n_robots_yellow=n_robots_yellow,
            n_robots_blue=n_robots_blue,
            force_start_after_goal=profile.game.force_start_after_goal,
            stop_duration_seconds=profile.game.stop_duration_seconds,
            prepare_duration_seconds=profile.game.prepare_duration_seconds,
            kickoff_timeout_seconds=profile.game.kickoff_timeout_seconds,
            geometry=self._geometry,
            auto_advance=profile.game.auto_advance,
        )
        self._dashboard_notifier: Optional[callable] = None
        self._bt_nodes_per_robot: dict[int, list[str]] = {}
        self._robot_feedback_data: list[dict] = []
        self._match_log = None
        self._match_log_tick = 0
        self._last_logged_ref_data: Optional[RefereeData] = None
        # The `RuleViolation` (if any) detected on the most recent `step()`
        # call — independent of whether the state machine actually applied
        # it (it may be suppressed by a transition cooldown). Exposed so
        # callers like `StrategyRunner`'s stats accumulator can tally
        # detected events (goals, etc.) without `RuleViolation` needing to
        # round-trip through `RefereeData`, which doesn't carry it.
        self.last_violation: Optional[RuleViolation] = None

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

    def attach_dashboard_notifier(self, notifier: callable) -> None:
        """Set the callback invoked with (ref_data, game_frame, bt_nodes, robot_feedback)
        on every `step()`. Set by `dashboard.views.referee.attach()`; not meant
        to be called directly by strategy code.
        """
        self._dashboard_notifier = notifier

    def attach_match_log(self, match_log) -> None:
        """Set a `MatchLog` to record referee-state changes (score/command/
        stage/designated position) into, sparsely — one row per change, not
        per tick. Set by `StrategyRunner`; mirrors `attach_dashboard_notifier`.
        """
        self._match_log = match_log

    # ------------------------------------------------------------------
    # Main loop interface
    # ------------------------------------------------------------------

    def step(self, game_frame: GameFrame, current_time: float) -> RefereeData:
        """Evaluate all rules and advance the state machine by one tick.

        First *stopping* violation (in priority order) wins and stops the
        scan early, same as before non-stopping fouls existed. A
        non-stopping foul (`RuleViolation.is_stopping=False` — SSL rulebook
        §8.4.2, "the game continues normally") does not compete for that
        slot: it never suppresses a later stopping rule's check, and a
        stopping violation found later in priority order still overrides it
        as this tick's `last_violation`/applied violation. If nothing
        stopping is found, the first non-stopping violation (if any) is
        still applied, since its only effect is the foul-counter/card side
        effect in `GameStateMachine._handle_foul` — it never touches
        `command`.
        """
        violation: Optional[RuleViolation] = None
        non_stopping_violation: Optional[RuleViolation] = None
        for rule in self._rules:
            result = rule.check(game_frame, self._geometry, self._state.command, self._state.ball_placement_target)
            if result is None:
                continue
            if result.is_stopping:
                violation = result
                break
            if non_stopping_violation is None:
                non_stopping_violation = result
        violation = violation or non_stopping_violation
        self.last_violation = violation

        previous_command = self._state.command
        result = self._state.step(current_time, violation, game_frame)

        # Notify rules only when the command actually changed (not when the
        # state machine ignored the violation due to the transition cooldown).
        if self._state.command != previous_command:
            for rule in self._rules:
                rule.reset()
        if self._dashboard_notifier is not None:
            self._dashboard_notifier(result, game_frame, self._bt_nodes_per_robot, self._robot_feedback_data)

        self._match_log_tick += 1
        if self._match_log is not None and result != self._last_logged_ref_data:
            self._last_logged_ref_data = result
            self._match_log.referee(
                tick=self._match_log_tick,
                sim_time=game_frame.ts,
                command=result.referee_command.name,
                stage=result.stage.name,
                yellow_score=result.yellow_team.score,
                blue_score=result.blue_team.score,
                designated=result.designated_position,
            )
        return result

    def set_debug_status(self, bt_nodes_per_robot: dict[int, list[str]]) -> None:
        """Set per-robot tactic debug status for GUI display.

        Called by StrategyRunner every tick with `AbstractStrategy.debug_status()`.
        """
        self._bt_nodes_per_robot = bt_nodes_per_robot

    def set_robot_feedback_data(self, robot_feedback_data: list[dict]) -> None:
        """Set raw robot-controller feedback rows for GUI display.

        Called by StrategyRunner in real mode after polling the controller port.
        """
        self._robot_feedback_data = [dict(row) for row in robot_feedback_data]

    def seed_clock(self, timestamp: float, initial_command: RefereeCommand = RefereeCommand.HALT) -> None:
        """Align all internal state-machine timers to *timestamp* and apply
        *initial_command*.

        Called by StrategyRunner after the first valid game frame is available,
        so the referee's timebase matches game_frame.ts from the very first tick.
        This handles both rsim (sim-time starting near 0) and grsim/real reuse
        (wall-clock timestamps that may be far from 0).
        """
        self._state.seed_clock(timestamp)
        if initial_command != RefereeCommand.HALT:
            self._state.set_command(initial_command, timestamp)

    def set_command(
        self,
        command: RefereeCommand,
        timestamp: float,
        designated_position: Optional[tuple[float, float]] = None,
        next_command: Optional[RefereeCommand] = None,
        status_message: Optional[str] = None,
    ) -> None:
        """Manual override for operator use or test scripting.

        Args:
            command: Referee command to apply.
            timestamp: Timestamp to associate with the command transition.
            designated_position: Optional ball placement/free-kick target to
                expose through ``RefereeData.designated_position``.
            next_command: Optional queued command to expose through
                ``RefereeData.next_command``.
            status_message: Optional human-readable referee status.

        The optional fields are intentionally thin wrappers around the custom
        referee state machine so integration tests and scenario runners do not
        need to reach into private ``_state`` attributes.
        """
        self._state.set_command(command, timestamp)
        if designated_position is not None:
            self._state.ball_placement_target = designated_position
        if next_command is not None:
            self._state.next_command = next_command
        if status_message is not None:
            self._state.status_message = status_message

    def force_command(
        self,
        command: "RefereeCommand",
        timestamp: float,
        ball_placement_target=None,
    ) -> None:
        """God-mode override — bypasses the STOP-first guard."""
        self._state.force_command(command, timestamp, ball_placement_target)

    def reset(self) -> None:
        """Restore this referee to its just-constructed state (score, command,
        stage, timers, and every rule's internal counters), for reuse across
        RL episodes without constructing a new `CustomReferee`.

        Does not reset geometry (see `override_geometry`) or the profile
        config the referee was built with — only per-episode state.
        `seed_clock()` still needs to be called again afterward, same as
        after construction, once the new episode's first game frame is
        available.
        """
        self._state.reset()
        for rule in self._rules:
            rule.reset_for_new_episode()

    # ------------------------------------------------------------------
    # Properties (read-only access for callers that need to inspect state)
    # ------------------------------------------------------------------

    def override_geometry(self, geometry: RefereeGeometry) -> None:
        """Replace the active field geometry on both the referee and the state machine.

        Called by StrategyRunner to ensure the referee's geometry always matches
        the actual field dims/bounds in use, regardless of what the YAML profile
        specifies.
        """
        self._geometry = geometry
        self._state._geometry = geometry

    @property
    def geometry(self) -> RefereeGeometry:
        return self._geometry

    @property
    def profile_name(self) -> str:
        return self._profile_name
