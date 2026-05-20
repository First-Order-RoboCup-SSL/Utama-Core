"""Integration tests for the ball placement feature.

These tests verify that ``BallPlacementStrategy`` (and the underlying
``BallPlacementStep`` behaviour) satisfies the three core requirements of
automatic ball placement:

  1. **Approach** — after BALL_PLACEMENT_YELLOW is issued, the placer robot
     drives toward the ball.
  2. **Carry** — once the robot captures the ball (``robot.has_ball`` or
     proximity), it moves toward ``designated_position`` with the dribbler on.
  3. **Clearance** — non-placer robots stay outside ``BALL_KEEP_OUT_DISTANCE``
     throughout the placement phase.

Setup
-----
- Exhibition Road field (``GREAT_EXHIBITION_FIELD_DIMS``, 4 m × 3 m).
- 2v2 — two yellow robots (our team) controlled by ``BallPlacementStrategy``;
  no enemy robots (they would interfere with deterministic positioning).
- CustomReferee with the "simulation" profile so auto-advance fires and the
  full BALL_PLACEMENT → DIRECT_FREE → NORMAL_START cycle can complete.

Running
-------
    pixi run pytest utama_core/tests/strategy_runner/test_ball_placement_rsim.py -v

All three tests are independent: they each start a fresh rsim episode, teleport
robots and ball to known positions, issue BALL_PLACEMENT_YELLOW directly, and
then observe the strategy's response over time.
"""

from __future__ import annotations

import math
from typing import Optional

import py_trees

from utama_core.config.field_params import GREAT_EXHIBITION_FIELD_DIMS
from utama_core.config.referee_constants import BALL_KEEP_OUT_DISTANCE
from utama_core.custom_referee import CustomReferee
from utama_core.entities.game import Game
from utama_core.entities.game.field import FieldBounds
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.run.strategy_runner import StrategyRunner
from utama_core.strategy.examples.ball_placement_strategy import BallPlacementStrategy
from utama_core.team_controller.src.controllers import AbstractSimController
from utama_core.tests.common.abstract_test_manager import (
    AbstractTestManager,
    TestingStatus,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Designated placement target used across all tests (inside exhibition field)
_TARGET = (0.8, 0.5)

# Tolerance within which a robot is considered to have "approached" the ball
_APPROACH_TOLERANCE = 0.5  # metres

# Tolerance within which the placer is considered to be "heading to target"
_TARGET_TOLERANCE = 0.6  # metres

# Clearance margin: robots must be at least this far from the ball
_CLEAR_MARGIN = 0.05  # metres — allowed slack inside keep-out boundary


# ---------------------------------------------------------------------------
# Shared runner factory
# ---------------------------------------------------------------------------


def _make_runner(referee: CustomReferee) -> StrategyRunner:
    """Build a 2v2 StrategyRunner on the Exhibition Road field."""
    return StrategyRunner(
        strategy=BallPlacementStrategy(),
        my_team_is_yellow=True,
        my_team_is_right=True,
        mode="rsim",
        exp_friendly=2,
        exp_enemy=0,
        exp_ball=True,
        full_field_dims=GREAT_EXHIBITION_FIELD_DIMS,
        referee=referee,
    )


# ---------------------------------------------------------------------------
# Test 1: placer robot drives toward the ball
# ---------------------------------------------------------------------------


class _ApproachBallManager(AbstractTestManager):
    """Verify that after BALL_PLACEMENT_YELLOW one robot moves toward the ball.

    Robot 0 starts near the placement target so it is the closest to the ball
    and therefore selected as the placer.  We wait until any robot closes within
    ``_APPROACH_TOLERANCE`` of the ball.
    """

    n_episodes = 1

    def __init__(self, referee: CustomReferee) -> None:
        super().__init__()
        self._referee = referee
        self.command_seen: bool = False
        self.robot_approached_ball: bool = False

    def reset_field(self, sim_controller: AbstractSimController, game: Game) -> None:
        # Ball near centre; robot 0 starts close to the ball so it becomes placer.
        sim_controller.teleport_ball(-0.5, 0.2)
        sim_controller.teleport_robot(game.my_team_is_yellow, 0, -0.3, 0.0)
        sim_controller.teleport_robot(game.my_team_is_yellow, 1, 0.6, -0.6)

        # Issue ball placement directly and set the designated target.
        self._referee.set_command(RefereeCommand.BALL_PLACEMENT_YELLOW, game.ts)
        self._referee._state.ball_placement_target = _TARGET

    def eval_status(self, game: Game) -> TestingStatus:
        ref = game.referee
        if ref is None:
            return TestingStatus.IN_PROGRESS

        if ref.referee_command == RefereeCommand.BALL_PLACEMENT_YELLOW:
            self.command_seen = True

        if not self.command_seen:
            return TestingStatus.IN_PROGRESS

        ball = game.ball
        if ball is None:
            return TestingStatus.IN_PROGRESS

        for robot in game.friendly_robots.values():
            dist = math.hypot(robot.p.x - ball.p.x, robot.p.y - ball.p.y)
            if dist < _APPROACH_TOLERANCE:
                self.robot_approached_ball = True
                return TestingStatus.SUCCESS

        return TestingStatus.IN_PROGRESS


def test_placer_approaches_ball_after_command(headless: bool) -> None:
    """After BALL_PLACEMENT_YELLOW, the closest robot drives toward the ball."""
    referee = CustomReferee.from_profile_name("simulation")
    runner = _make_runner(referee)
    tm = _ApproachBallManager(referee)

    passed = runner.run_test(tm, episode_timeout=20.0, rsim_headless=headless)

    assert tm.command_seen, "BALL_PLACEMENT_YELLOW was never seen in game.referee"
    assert tm.robot_approached_ball, "No robot moved within approach tolerance of the ball"
    assert passed


# ---------------------------------------------------------------------------
# Test 2: placer moves toward designated_position (carry phase)
# ---------------------------------------------------------------------------


class _CarryToTargetManager(AbstractTestManager):
    """Verify that the placer moves toward designated_position during the carry phase.

    Robot 0 starts on top of the ball so it immediately captures it (``has_ball``
    becomes True quickly in rsim).  We then check that robot 0 is moving toward
    the target rather than staying put.
    """

    n_episodes = 1

    # How much closer the placer must get to the target over the observation window
    _PROGRESS_THRESHOLD = 0.2  # metres

    def __init__(self, referee: CustomReferee) -> None:
        super().__init__()
        self._referee = referee
        self.command_seen: bool = False
        self.placer_made_progress: bool = False
        self._initial_dist_to_target: Optional[float] = None

    def reset_field(self, sim_controller: AbstractSimController, game: Game) -> None:
        # Place robot 0 right on the ball so it captures it immediately.
        sim_controller.teleport_ball(-0.6, 0.0)
        sim_controller.teleport_robot(game.my_team_is_yellow, 0, -0.6, 0.0)
        sim_controller.teleport_robot(game.my_team_is_yellow, 1, 0.6, -0.6)

        self._referee.set_command(RefereeCommand.BALL_PLACEMENT_YELLOW, game.ts)
        self._referee._state.ball_placement_target = _TARGET

    def eval_status(self, game: Game) -> TestingStatus:
        ref = game.referee
        if ref is None:
            return TestingStatus.IN_PROGRESS

        if ref.referee_command == RefereeCommand.BALL_PLACEMENT_YELLOW:
            self.command_seen = True

        if not self.command_seen:
            return TestingStatus.IN_PROGRESS

        robot0 = game.friendly_robots.get(0)
        if robot0 is None:
            return TestingStatus.IN_PROGRESS

        tx, ty = _TARGET
        dist = math.hypot(robot0.p.x - tx, robot0.p.y - ty)

        if self._initial_dist_to_target is None:
            self._initial_dist_to_target = dist
            return TestingStatus.IN_PROGRESS

        improvement = self._initial_dist_to_target - dist
        if improvement >= self._PROGRESS_THRESHOLD:
            self.placer_made_progress = True
            return TestingStatus.SUCCESS

        return TestingStatus.IN_PROGRESS


def test_placer_moves_toward_designated_position(headless: bool) -> None:
    """After capturing the ball, the placer robot moves toward the designated position."""
    referee = CustomReferee.from_profile_name("simulation")
    runner = _make_runner(referee)
    tm = _CarryToTargetManager(referee)

    passed = runner.run_test(tm, episode_timeout=25.0, rsim_headless=headless)

    assert tm.command_seen, "BALL_PLACEMENT_YELLOW was never seen in game.referee"
    assert tm.placer_made_progress, "Placer robot did not make sufficient progress toward the designated position"
    assert passed


# ---------------------------------------------------------------------------
# Test 3: non-placer robot stays outside keep-out distance
# ---------------------------------------------------------------------------


class _ClearanceManager(AbstractTestManager):
    """Verify that non-placer robots respect the ball keep-out distance.

    Robot 0 is the placer (closest to ball).  Robot 1 starts inside the
    keep-out radius.  We check that robot 1 clears the zone within the
    observation window and stays clear thereafter.
    """

    n_episodes = 1

    # Number of consecutive frames all non-placers must be clear
    _FRAMES_REQUIRED = 60

    def __init__(self, referee: CustomReferee) -> None:
        super().__init__()
        self._referee = referee
        self.command_seen: bool = False
        self.clearance_achieved: bool = False
        self._placer_id: Optional[int] = None
        self._clear_frame_count: int = 0

    def reset_field(self, sim_controller: AbstractSimController, game: Game) -> None:
        # Robot 0 closest to ball → becomes placer.
        # Robot 1 starts inside keep-out radius to test clearing behaviour.
        sim_controller.teleport_ball(-0.5, 0.3)
        sim_controller.teleport_robot(game.my_team_is_yellow, 0, -0.3, 0.2)
        sim_controller.teleport_robot(game.my_team_is_yellow, 1, -0.5, 0.5)  # within keep-out

        self._referee.set_command(RefereeCommand.BALL_PLACEMENT_YELLOW, game.ts)
        self._referee._state.ball_placement_target = _TARGET

    def eval_status(self, game: Game) -> TestingStatus:
        ref = game.referee
        if ref is None:
            return TestingStatus.IN_PROGRESS

        if ref.referee_command == RefereeCommand.BALL_PLACEMENT_YELLOW:
            self.command_seen = True

        if not self.command_seen:
            return TestingStatus.IN_PROGRESS

        ball = game.ball
        if ball is None:
            return TestingStatus.IN_PROGRESS

        # Identify the placer as the robot closest to the ball (matches BallPlacementStep logic)
        if self._placer_id is None and game.friendly_robots:
            self._placer_id = min(
                game.friendly_robots,
                key=lambda rid: math.hypot(
                    game.friendly_robots[rid].p.x - ball.p.x,
                    game.friendly_robots[rid].p.y - ball.p.y,
                ),
            )

        threshold = BALL_KEEP_OUT_DISTANCE - _CLEAR_MARGIN

        non_placers_clear = all(
            math.hypot(robot.p.x - ball.p.x, robot.p.y - ball.p.y) >= threshold
            for rid, robot in game.friendly_robots.items()
            if rid != self._placer_id
        )

        if non_placers_clear:
            self._clear_frame_count += 1
        else:
            self._clear_frame_count = 0

        if self._clear_frame_count >= self._FRAMES_REQUIRED:
            self.clearance_achieved = True
            return TestingStatus.SUCCESS

        return TestingStatus.IN_PROGRESS


def test_non_placer_clears_ball_keep_out_zone(headless: bool) -> None:
    """Non-placer robots clear and hold outside the ball keep-out radius."""
    referee = CustomReferee.from_profile_name("simulation")
    runner = _make_runner(referee)
    tm = _ClearanceManager(referee)

    passed = runner.run_test(tm, episode_timeout=20.0, rsim_headless=headless)

    assert tm.command_seen, "BALL_PLACEMENT_YELLOW was never seen in game.referee"
    assert tm.clearance_achieved, (
        f"Non-placer robots did not sustain clearance outside "
        f"BALL_KEEP_OUT_DISTANCE ({BALL_KEEP_OUT_DISTANCE} m) "
        f"for {_ClearanceManager._FRAMES_REQUIRED} consecutive frames"
    )
    assert passed
