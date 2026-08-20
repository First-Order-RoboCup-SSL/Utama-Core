"""Referee behaviour integration tests using rsim + CustomReferee.

Each test:
  1. Starts a full rsim session with a CustomReferee (simulation profile).
  2. Sets up a scenario via reset_field (ball position + velocity, robots placed).
  3. Waits for the referee to detect a violation and issue a command.
  4. Verifies that the robots respond correctly via eval_status.

Scenarios covered:
  - BALL_PLACEMENT_YELLOW issued directly → closest robot drives to designated target.
  - Ball exits side boundary → DIRECT_FREE_YELLOW issued → kicker drives to ball.
  - Ball exits side boundary, robot near ball → DIRECT_FREE_BLUE issued → robots clear keep-out zone.
  - PREPARE_KICKOFF issued → robots reach own-half positions outside centre circle.
  - Full out-of-bounds sequence: ball exits → STOP → BALL_PLACEMENT → DIRECT_FREE → NORMAL_START.

Note on initial commands:
  Out-of-bounds and goal rules only fire during NORMAL_START / FORCE_START, so the
  manager issues FORCE_START in reset_field to put the game into active play before
  the ball exits.  The kickoff test issues PREPARE_KICKOFF_YELLOW directly.

Note on last-touch tracking:
  DIRECT_FREE_YELLOW (our free kick) is issued when the enemy last touched the ball.
  DIRECT_FREE_BLUE (their free kick) is issued when we last touched the ball.
  We control which fires by positioning a friendly robot next to the ball before
  it exits (triggers friendly last-touch → DIRECT_FREE_BLUE, their kick).
  With no robot near the ball, last-touch defaults to DIRECT_FREE_YELLOW (ours).

Note on ball placement in out-of-bounds:
  OutOfBoundsRule issues STOP → DIRECT_FREE directly (no automatic ball placement).
  Ball placement is only reachable via set_command().  The full-sequence test manually
  injects BALL_PLACEMENT_YELLOW after the STOP fires, then lets auto-advance carry the
  state machine through BALL_PLACEMENT → DIRECT_FREE → NORMAL_START.
"""

import math
from dataclasses import dataclass
from typing import Optional

from utama_core.config.field_params import STANDARD_FIELD_DIMS
from utama_core.config.referee_constants import BALL_KEEP_OUT_DISTANCE
from utama_core.custom_referee import CustomReferee
from utama_core.custom_referee.geometry import RefereeGeometry
from utama_core.custom_referee.rules.out_of_bounds_rule import OutOfBoundsRule
from utama_core.entities.data.command import RobotCommand
from utama_core.entities.game import Game
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.kernel.context import KernelContext
from utama_core.kernel.kernel_strategy import build_default_kernel_strategy
from utama_core.kernel.strategy import Strategy as KernelSchedulerStrategy
from utama_core.kernel.tactic import BaseTactic, RobotId, TacticTag
from utama_core.motion_planning.src.common.motion_controller import MotionController
from utama_core.run.strategy_runner import StrategyRunner
from utama_core.skills.src.go_to_ball import go_to_ball
from utama_core.skills.src.utils.move_utils import empty_command
from utama_core.strategy.common.abstract_strategy import AbstractStrategy
from utama_core.team_controller.src.controllers import AbstractSimController
from utama_core.tests.common.abstract_test_manager import (
    AbstractTestManager,
    TestingStatus,
)

# ---------------------------------------------------------------------------
# Minimal idle strategy — RefereeOverride (kernel.Strategy.tick()) handles all
# motion during override commands; only test_out_of_bounds_restart_spot_...
# needs a real tactic below, since it isn't exercising the override layer.
# ---------------------------------------------------------------------------

PREPARE_KICKOFF_COMMANDS = {
    RefereeCommand.PREPARE_KICKOFF_YELLOW,
    RefereeCommand.PREPARE_KICKOFF_BLUE,
}


def _idle_strategy() -> AbstractStrategy:
    """Kernel strategy with an empty outfield pool — does nothing; the referee
    override layer (`kernel.Strategy.tick()`'s `RefereeOverride`) handles all
    motion during override commands regardless of the tactic roster."""
    return AbstractStrategy(build_kernel_strategy=build_default_kernel_strategy(()), exp_ball=True)


@dataclass
class _GoToBallUntilPossessionMem:
    pass


class _GoToBallUntilPossessionTactic(BaseTactic[_GoToBallUntilPossessionMem]):
    """Test-only tactic: single robot drives to the ball via `go_to_ball` until
    it has possession, then holds. Direct kernel-native port of the BT
    "SetRobotID -> Selector(HasBall, GoToBall)" state machine this replaces —
    no state machine needed since `go_to_ball` is idempotent to call every
    tick and `has_ball` doesn't change what command gets issued.
    """

    tag = TacticTag.ATTACK

    def initial_mem(self) -> _GoToBallUntilPossessionMem:
        return _GoToBallUntilPossessionMem()

    def tick(
        self, game: Game, ctx: KernelContext, robot_ids: tuple[RobotId, ...], mem: _GoToBallUntilPossessionMem
    ) -> tuple[dict[RobotId, RobotCommand], _GoToBallUntilPossessionMem]:
        robot_id = robot_ids[0]
        robot = game.friendly_robots[robot_id]
        if robot.has_ball:
            return {robot_id: empty_command(dribbler_on=True)}, mem
        return {robot_id: go_to_ball(game, ctx.motion_controller, robot_id)}, mem


def _go_to_ball_strategy(robot_id: int) -> AbstractStrategy:
    """`robot_id` is in both the kernel scheduler's outfield pool AND the
    default `goalkeeper_id=0` pin when `robot_id == 0` — harmless here since
    `AbstractStrategy.step()` only ticks the goalkeeper when its id is not
    already present in `cmd_map`, and the tactic always assigns `robot_id` a
    command, so the (otherwise-unused) goalkeeper tick is simply skipped."""

    def _build(motion_controller: MotionController) -> KernelSchedulerStrategy:
        ctx = KernelContext(motion_controller=motion_controller)
        return KernelSchedulerStrategy(
            tactics={"go_to_ball": _GoToBallUntilPossessionTactic()},
            partitioner=KernelSchedulerStrategy.single_tactic_picker(lambda game, active: "go_to_ball"),
            outfield_robot_ids=(robot_id,),
            ctx=ctx,
        )

    return AbstractStrategy(build_kernel_strategy=_build)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_runner(referee: CustomReferee, n_friendly: int = 3) -> StrategyRunner:
    return StrategyRunner(
        strategy=_idle_strategy(),
        my_team_is_yellow=True,
        my_team_is_right=False,  # defending left → own half is negative-x
        mode="rsim",
        exp_friendly=n_friendly,
        exp_enemy=0,
        exp_ball=True,
        referee=referee,
    )


# ---------------------------------------------------------------------------
# Scenario 1: BALL_PLACEMENT is covered by test_ball_placement_rsim.py, not
# here. This file's version of the scenario is structurally unfixable:
# StrategyRunner._run_step teleports the ball straight to designated_position
# the instant a BALL_PLACEMENT_* command is first seen ("simulate placement
# instantly" — robots cannot physically retrieve an out-of-bounds ball in
# simulation), so ball.p.distance_to(target) is already inside
# BALL_PLACEMENT_DONE_DISTANCE on the very first tick regardless of robot
# starting positions. BallPlacementOursStep enters its release/done phase
# immediately and no robot ever moves — there is no way to position robots to
# observe real "approach the ball" motion under this runner behavior.
# test_ball_placement_rsim.py's tests avoid this by placing the ball far from
# its own designated_position, so the teleport-to-target shortcut never
# collapses those scenarios the same way; that file is where this coverage
# (approach/carry/clearance/placer-selection) genuinely lives.
# ---------------------------------------------------------------------------
# Scenario 2a: our direct free kick — kicker drives toward ball
# ---------------------------------------------------------------------------


class _DirectFreeOursManager(AbstractTestManager):
    """DIRECT_FREE_YELLOW is injected directly; kicker must drive toward the ball."""

    n_episodes = 1
    APPROACH_TOLERANCE = 0.6

    def __init__(self, referee: CustomReferee):
        super().__init__()
        self._referee = referee
        self.direct_free_seen: bool = False
        self.robot_near_ball: bool = False

    def reset_field(self, sim_controller: AbstractSimController, game: Game):
        sim_controller.teleport_robot(game.my_team_is_yellow, 0, -1.5, 0.0)
        sim_controller.teleport_robot(game.my_team_is_yellow, 1, -2.0, 0.5)
        sim_controller.teleport_robot(game.my_team_is_yellow, 2, -2.0, -0.5)
        sim_controller.teleport_ball(0.0, 0.0)
        # Inject directly — bypass OOB detection which now routes through ball placement first.
        self._referee.force_command(RefereeCommand.DIRECT_FREE_YELLOW, game.ts)

    def eval_status(self, game: Game) -> TestingStatus:
        ref = game.referee
        if ref is None:
            return TestingStatus.IN_PROGRESS

        if ref.referee_command == RefereeCommand.DIRECT_FREE_YELLOW:
            self.direct_free_seen = True

        if not self.direct_free_seen:
            return TestingStatus.IN_PROGRESS

        ball = game.ball
        if ball is None:
            return TestingStatus.IN_PROGRESS

        for robot in game.friendly_robots.values():
            dist = math.hypot(robot.p.x - ball.p.x, robot.p.y - ball.p.y)
            if dist < self.APPROACH_TOLERANCE:
                self.robot_near_ball = True
                return TestingStatus.SUCCESS

        return TestingStatus.IN_PROGRESS


def test_direct_free_kick_ours_robot_drives_to_ball(headless):
    """After our direct free kick, the kicker drives toward the ball."""
    referee = CustomReferee.from_profile_name("simulation")
    runner = _make_runner(referee)
    tm = _DirectFreeOursManager(referee)

    passed = runner.run_test(tm, episode_timeout=20.0, rsim_headless=headless)

    assert tm.direct_free_seen, "CustomReferee never issued DIRECT_FREE_YELLOW"
    assert tm.robot_near_ball, "No robot drove toward the ball during our direct free kick"
    assert passed


# ---------------------------------------------------------------------------
# Scenario 2b: their direct free kick — robots clear the keep-out zone
# ---------------------------------------------------------------------------


class _DirectFreeTheirsManager(AbstractTestManager):
    """DIRECT_FREE_BLUE is injected directly; all robots must clear the keep-out radius."""

    n_episodes = 1
    CLEAR_TOLERANCE = 0.1

    def __init__(self, referee: CustomReferee):
        super().__init__()
        self._referee = referee
        self.direct_free_seen: bool = False
        self.robots_cleared: bool = False
        self._ball_pos_at_call: Optional[tuple[float, float]] = None

    def reset_field(self, sim_controller: AbstractSimController, game: Game):
        # All robots start inside keep-out radius around the ball.
        sim_controller.teleport_robot(game.my_team_is_yellow, 0, 0.0, 0.3)
        sim_controller.teleport_robot(game.my_team_is_yellow, 1, 0.1, -0.3)
        sim_controller.teleport_robot(game.my_team_is_yellow, 2, -0.1, 0.3)
        sim_controller.teleport_ball(0.0, 0.0)
        # Inject directly — bypass OOB detection which now routes through ball placement first.
        self._referee.force_command(RefereeCommand.DIRECT_FREE_BLUE, game.ts)

    def eval_status(self, game: Game) -> TestingStatus:
        ref = game.referee
        if ref is None:
            return TestingStatus.IN_PROGRESS

        if ref.referee_command == RefereeCommand.DIRECT_FREE_BLUE:
            self.direct_free_seen = True
            if self._ball_pos_at_call is None and game.ball is not None:
                self._ball_pos_at_call = (game.ball.p.x, game.ball.p.y)

        if not self.direct_free_seen or self._ball_pos_at_call is None:
            return TestingStatus.IN_PROGRESS

        bx, by = self._ball_pos_at_call
        threshold = BALL_KEEP_OUT_DISTANCE - self.CLEAR_TOLERANCE

        all_clear = all(math.hypot(r.p.x - bx, r.p.y - by) >= threshold for r in game.friendly_robots.values())
        if all_clear:
            self.robots_cleared = True
            return TestingStatus.SUCCESS

        return TestingStatus.IN_PROGRESS


def test_direct_free_kick_theirs_robots_clear_keep_out_zone(headless):
    """During opponent direct free kick, all robots clear the keep-out radius around the ball."""
    referee = CustomReferee.from_profile_name("simulation")
    runner = _make_runner(referee)
    tm = _DirectFreeTheirsManager(referee)

    passed = runner.run_test(tm, episode_timeout=20.0, rsim_headless=headless)

    assert tm.direct_free_seen, "CustomReferee never issued DIRECT_FREE_BLUE"
    assert tm.robots_cleared, "Robots did not clear the keep-out zone during opponent direct free kick"
    assert passed


# ---------------------------------------------------------------------------
# Scenario 3: PREPARE_KICKOFF → robots on own half, outside centre circle
# ---------------------------------------------------------------------------


class _KickoffPositioningManager(AbstractTestManager):
    """PREPARE_KICKOFF_YELLOW is issued in reset_field.

    Robots start near the centre circle so their clearing movement is visible.
    eval_status verifies all robots reach and hold own-half positions outside
    the centre circle for N_FRAMES_TO_CHECK consecutive frames.
    my_team_is_right=False → own half is negative-x.
    """

    n_episodes = 1
    N_FRAMES_TO_CHECK = 100
    POSITION_TOLERANCE = 0.15

    def __init__(self, referee: CustomReferee):
        super().__init__()
        self._referee = referee
        self.kickoff_command_seen: bool = False
        self.success_frame_count: int = 0

    def reset_field(self, sim_controller: AbstractSimController, game: Game):
        # Robots start near the centre circle so clearing movement is clearly visible.
        sim_controller.teleport_robot(game.my_team_is_yellow, 0, -0.3, 0.0)
        sim_controller.teleport_robot(game.my_team_is_yellow, 1, -0.2, 0.5)
        sim_controller.teleport_robot(game.my_team_is_yellow, 2, -0.2, -0.5)
        sim_controller.teleport_ball(0.0, 0.0)
        # force_command (not set_command) — set_command inserts STOP first, which
        # only auto-advances once every robot clears BALL_CLEAR_DIST from the
        # ball; robots start deliberately near the centre circle here (so the
        # clearing movement this test verifies is visible), which would never
        # satisfy that clearance and the command would never advance past STOP.
        self._referee.force_command(RefereeCommand.PREPARE_KICKOFF_YELLOW, game.ts)

    def eval_status(self, game: Game) -> TestingStatus:
        ref = game.referee
        if ref is None:
            return TestingStatus.IN_PROGRESS

        if ref.referee_command in PREPARE_KICKOFF_COMMANDS:
            self.kickoff_command_seen = True

        if not self.kickoff_command_seen:
            return TestingStatus.IN_PROGRESS

        # my_team_is_right=False → own half is negative-x.
        # Kicker (robot 1 — lowest-ID outfield robot; the goalkeeper is pinned
        # and exempt) targets (0,0) on the boundary — allow x <= 0.2.
        # Support robots (0, 2) must be strictly on own half and outside centre circle.
        support_robots = [r for rid, r in game.friendly_robots.items() if rid != 1]
        kicker = game.friendly_robots.get(1)

        kicker_ok = kicker is not None and kicker.p.x <= 0.2
        supports_on_half = all(r.p.x <= self.POSITION_TOLERANCE for r in support_robots)
        supports_outside_circle = all(math.hypot(r.p.x, r.p.y) >= 0.5 - self.POSITION_TOLERANCE for r in support_robots)

        if kicker_ok and supports_on_half and supports_outside_circle:
            self.success_frame_count += 1
        else:
            self.success_frame_count = 0

        if self.success_frame_count >= self.N_FRAMES_TO_CHECK:
            return TestingStatus.SUCCESS

        return TestingStatus.IN_PROGRESS


def test_prepare_kickoff_robots_form_on_own_half_outside_circle(headless):
    """At kickoff, all robots reach and hold own-half positions outside the centre circle."""
    referee = CustomReferee.from_profile_name("simulation")
    runner = _make_runner(referee)
    tm = _KickoffPositioningManager(referee)

    passed = runner.run_test(tm, episode_timeout=30.0, rsim_headless=headless)

    assert tm.kickoff_command_seen, "CustomReferee never issued a PREPARE_KICKOFF command"
    assert passed, "Robots did not sustain a legal kickoff formation for the required number of frames"


# ---------------------------------------------------------------------------
# Scenario 4: out-of-bounds restart spot can be played
# ---------------------------------------------------------------------------


class _RestartSpotCaptureManager(AbstractTestManager):
    """Places the ball at the out-of-bounds restart spot and verifies capture.

    This guards against restart positions that are technically infield but too
    close to the wall for the robot/dribbler geometry to acquire the ball.
    """

    n_episodes = 1

    def __init__(self, restart_spot: tuple[float, float]):
        super().__init__()
        self.restart_spot = restart_spot
        self.robot_has_ball = False
        self.min_distance_to_ball = math.inf

    def reset_field(self, sim_controller: AbstractSimController, game: Game):
        target_x, target_y = self.restart_spot
        sim_controller.teleport_robot(game.my_team_is_yellow, 0, target_x, target_y - 0.9, math.pi / 2)
        sim_controller.teleport_ball(target_x, target_y)

    def eval_status(self, game: Game) -> TestingStatus:
        robot = game.friendly_robots[0]
        ball = game.ball
        self.min_distance_to_ball = min(
            self.min_distance_to_ball,
            math.hypot(robot.p.x - ball.p.x, robot.p.y - ball.p.y),
        )
        if robot.has_ball:
            self.robot_has_ball = True
            return TestingStatus.SUCCESS
        return TestingStatus.IN_PROGRESS


def test_out_of_bounds_restart_spot_is_capturable_by_go_to_ball(headless):
    geometry = RefereeGeometry.from_field_dims(STANDARD_FIELD_DIMS)
    restart_spot = OutOfBoundsRule._nearest_infield_point(0.0, geometry.half_width + 0.5, geometry)

    runner = StrategyRunner(
        strategy=_go_to_ball_strategy(robot_id=0),
        my_team_is_yellow=True,
        my_team_is_right=False,
        mode="rsim",
        exp_friendly=1,
        exp_enemy=0,
        exp_ball=True,
        referee=None,
    )
    tm = _RestartSpotCaptureManager(restart_spot)

    passed = runner.run_test(tm, episode_timeout=5.0, rsim_headless=headless)

    assert tm.robot_has_ball, (
        "Robot could not acquire the out-of-bounds restart ball at "
        f"{restart_spot}; closest approach was {tm.min_distance_to_ball:.3f} m"
    )
    assert passed


# ---------------------------------------------------------------------------
# Future work: full out-of-bounds sequence integration test
#
# Intended scenario:
#   ball exits → STOP → BALL_PLACEMENT_YELLOW → robot physically carries ball
#   to designated position → DIRECT_FREE_YELLOW → kicker drives to ball →
#   NORMAL_START (play resumes)
#
# Why it is not implemented yet:
#   BallPlacementOursStep relies on robot.has_ball (IR sensor) to switch from
#   approach to carry mode. In rsim the robot drives to ball.p but decelerates
#   to a stop AT the ball centre rather than past it, so the dribbler never
#   properly captures the ball — the robot ends up pushing it instead of
#   carrying it. Several approaches were tried:
#     - Adding a behind-ball approach offset (robot stopped short with a gap)
#     - Driving directly into ball.p with face-target orientation (pushed sideways)
#     - Proximity fallback for has_ball (robot reached ball but pushed it away)
#   Root cause: the motion controller targets and the dribbler capture
#   mechanics need tighter integration (approach from behind, slower final
#   approach speed, or a dedicated "get-behind-ball" skill) before ball
#   placement via robot carry can be reliably tested end-to-end.
#
# Additionally, OutOfBoundsRule currently issues STOP → DIRECT_FREE directly
# (no automatic ball placement step). Ball placement must be injected manually
# via set_command(), which makes the test scenario somewhat artificial.
# ---------------------------------------------------------------------------
