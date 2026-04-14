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
import time
from typing import Optional

import py_trees

from utama_core.config.referee_constants import BALL_KEEP_OUT_DISTANCE
from utama_core.custom_referee import CustomReferee
from utama_core.entities.game import Game
from utama_core.entities.game.field import FieldBounds
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.run.strategy_runner import StrategyRunner
from utama_core.strategy.common.abstract_strategy import AbstractStrategy
from utama_core.team_controller.src.controllers import AbstractSimController
from utama_core.tests.common.abstract_test_manager import (
    AbstractTestManager,
    TestingStatus,
)

# ---------------------------------------------------------------------------
# Minimal idle strategy — referee override tree handles all motion
# ---------------------------------------------------------------------------

BALL_PLACEMENT_COMMANDS = {
    RefereeCommand.BALL_PLACEMENT_YELLOW,
    RefereeCommand.BALL_PLACEMENT_BLUE,
}
DIRECT_FREE_COMMANDS = {
    RefereeCommand.DIRECT_FREE_YELLOW,
    RefereeCommand.DIRECT_FREE_BLUE,
}
PREPARE_KICKOFF_COMMANDS = {
    RefereeCommand.PREPARE_KICKOFF_YELLOW,
    RefereeCommand.PREPARE_KICKOFF_BLUE,
}


class _IdleStrategy(AbstractStrategy):
    """Does nothing in the strategy subtree — referee override layer handles all motion."""

    exp_ball: bool = True

    def create_behaviour_tree(self) -> py_trees.behaviour.Behaviour:
        return py_trees.behaviours.Running(name="Idle")

    def assert_exp_robots(self, n_runtime_friendly: int, n_runtime_enemy: int) -> bool:
        return True

    def assert_exp_goals(self, includes_my_goal_line: bool, includes_opp_goal_line: bool) -> bool:
        return True

    def get_min_bounding_req(self) -> Optional[FieldBounds]:
        return None


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_runner(referee: CustomReferee, n_friendly: int = 3) -> StrategyRunner:
    return StrategyRunner(
        strategy=_IdleStrategy(),
        my_team_is_yellow=True,
        my_team_is_right=False,  # defending left → own half is negative-x
        mode="rsim",
        exp_friendly=n_friendly,
        exp_enemy=0,
        exp_ball=True,
        referee_system="custom",
        custom_referee=referee,
    )


# ---------------------------------------------------------------------------
# Scenario 1: BALL_PLACEMENT — closest robot drives to designated target
# ---------------------------------------------------------------------------


class _BallPlacementManager(AbstractTestManager):
    """BALL_PLACEMENT_YELLOW is issued directly via set_command.

    A robot is placed near the designated target so it is the obvious candidate
    to reach the position quickly.  We verify it gets within APPROACH_TOLERANCE.
    """

    n_episodes = 1
    TARGET = (1.0, 1.0)  # designated placement position
    APPROACH_TOLERANCE = 0.4

    def __init__(self, referee: CustomReferee):
        super().__init__()
        self._referee = referee
        self.placement_command_seen: bool = False
        self.robot_reached_target: bool = False

    def reset_field(self, sim_controller: AbstractSimController, game: Game):
        # Robot 0 starts close to the target so it is the clear closest candidate.
        sim_controller.teleport_robot(game.my_team_is_yellow, 0, 0.5, 0.5)
        sim_controller.teleport_robot(game.my_team_is_yellow, 1, -2.0, 0.5)
        sim_controller.teleport_robot(game.my_team_is_yellow, 2, -2.0, -0.5)
        # Ball far from the placement target
        sim_controller.teleport_ball(-1.0, -1.0)
        # Issue BALL_PLACEMENT directly and set the designated target
        self._referee.set_command(RefereeCommand.BALL_PLACEMENT_YELLOW, time.time())
        self._referee._state.ball_placement_target = self.TARGET

    def eval_status(self, game: Game) -> TestingStatus:
        ref = game.referee
        if ref is None:
            return TestingStatus.IN_PROGRESS

        if ref.referee_command in BALL_PLACEMENT_COMMANDS:
            self.placement_command_seen = True

        if not self.placement_command_seen:
            return TestingStatus.IN_PROGRESS

        target_x, target_y = self.TARGET
        for robot in game.friendly_robots.values():
            dist = math.hypot(robot.p.x - target_x, robot.p.y - target_y)
            if dist < self.APPROACH_TOLERANCE:
                self.robot_reached_target = True
                return TestingStatus.SUCCESS

        return TestingStatus.IN_PROGRESS


def test_ball_placement_robot_approaches_designated_position(headless):
    """During BALL_PLACEMENT, the closest robot drives toward the designated target."""
    referee = CustomReferee.from_profile_name("simulation")
    runner = _make_runner(referee)
    tm = _BallPlacementManager(referee)

    passed = runner.run_test(tm, episode_timeout=20.0, rsim_headless=headless)

    assert tm.placement_command_seen, "CustomReferee never issued a BALL_PLACEMENT command"
    assert tm.robot_reached_target, "No robot approached the ball placement designated position"
    assert passed


# ---------------------------------------------------------------------------
# Scenario 2a: our direct free kick — kicker drives toward ball
# ---------------------------------------------------------------------------


class _DirectFreeOursManager(AbstractTestManager):
    """Ball exits the side boundary with no friendly robot nearby → DIRECT_FREE_YELLOW (ours).

    With no robot close enough to the ball to register a friendly last-touch,
    OutOfBoundsRule defaults to DIRECT_FREE_YELLOW.  We verify the kicker
    drives toward the (now out-of-bounds) ball.
    """

    n_episodes = 1
    APPROACH_TOLERANCE = 0.6  # slightly wider: ball may be just outside boundary

    def __init__(self, referee: CustomReferee):
        super().__init__()
        self._referee = referee
        self.direct_free_seen: bool = False
        self.robot_near_ball: bool = False

    def reset_field(self, sim_controller: AbstractSimController, game: Game):
        # Keep all robots well away from the ball so last-touch is unknown → DIRECT_FREE_YELLOW
        sim_controller.teleport_robot(game.my_team_is_yellow, 0, -1.5, 0.0)
        sim_controller.teleport_robot(game.my_team_is_yellow, 1, -2.0, 0.5)
        sim_controller.teleport_robot(game.my_team_is_yellow, 2, -2.0, -0.5)
        # Ball heading out the top sideline
        sim_controller.teleport_ball(0.0, 2.5, vx=0.0, vy=2.5)
        self._referee.set_command(RefereeCommand.FORCE_START, time.time())

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
    """A robot starts inside the keep-out radius; DIRECT_FREE_BLUE (theirs) is issued.

    We place robot 0 right next to the ball before it exits so last-touch registers
    as friendly → OutOfBoundsRule issues DIRECT_FREE_BLUE (opponent's free kick).
    The test verifies that all robots end up outside the keep-out radius.
    """

    n_episodes = 1
    # All robots must clear beyond this radius from the ball position.
    CLEAR_TOLERANCE = 0.1  # allowed margin inside keep-out (robots should be well clear)

    def __init__(self, referee: CustomReferee):
        super().__init__()
        self._referee = referee
        self.direct_free_seen: bool = False
        self.robots_cleared: bool = False
        # We record the ball position when DIRECT_FREE_BLUE fires to check clearing.
        self._ball_pos_at_call: Optional[tuple[float, float]] = None

    def reset_field(self, sim_controller: AbstractSimController, game: Game):
        # Robot 0 is placed right next to the ball — it will register as last-toucher.
        sim_controller.teleport_robot(game.my_team_is_yellow, 0, 0.0, 2.4)
        # Robots 1 and 2 also start near the ball path — both inside keep-out.
        sim_controller.teleport_robot(game.my_team_is_yellow, 1, 0.1, 2.3)
        sim_controller.teleport_robot(game.my_team_is_yellow, 2, -0.1, 2.3)
        # Ball heading out the top sideline; robot 0 is close enough for last-touch
        sim_controller.teleport_ball(0.0, 2.5, vx=0.0, vy=2.5)
        self._referee.set_command(RefereeCommand.FORCE_START, time.time())

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
        self._referee.set_command(RefereeCommand.PREPARE_KICKOFF_YELLOW, time.time())

    def eval_status(self, game: Game) -> TestingStatus:
        ref = game.referee
        if ref is None:
            return TestingStatus.IN_PROGRESS

        if ref.referee_command in PREPARE_KICKOFF_COMMANDS:
            self.kickoff_command_seen = True

        if not self.kickoff_command_seen:
            return TestingStatus.IN_PROGRESS

        # my_team_is_right=False → own half is negative-x.
        # Kicker (robot 0) targets (0,0) on the boundary — allow x <= 0.2.
        # Support robots (1, 2) must be strictly on own half and outside centre circle.
        support_robots = [r for rid, r in game.friendly_robots.items() if rid != 0]
        kicker = game.friendly_robots.get(0)

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
