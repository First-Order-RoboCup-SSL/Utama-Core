"""Tests for `utama_core.replay.scenario` (replay-to-repro reconstruction).

See `utama_core/replay/scenario.py`'s module docstring for why this module
exists and `docs/STRATEGY_DEVELOPMENT.md`'s Observability section for how it
fits alongside `MatchLog`/`render_window`/the stall watchdog.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from utama_core.config.field_params import STANDARD_FIELD_DIMS
from utama_core.engine.abstract_strategy import AbstractStrategy
from utama_core.entities.game import Game
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.replay.scenario import RobotState, Scenario, apply_scenario
from utama_core.strategy.kernel_strategy import build_default_kernel_strategy
from utama_core.team_controller.src.controllers import AbstractSimController
from utama_core.tests.common.abstract_test_manager import (
    AbstractTestManager,
    TestingStatus,
)

os.environ["SDL_VIDEO_WINDOW_POS"] = "100,100"


def _idle_strategy() -> AbstractStrategy:
    return AbstractStrategy(build_kernel_strategy=build_default_kernel_strategy(()))


class _ApplyScenarioTestManager(AbstractTestManager):
    """Drives a headless rsim `StrategyRunner` through `run_test`'s normal
    setup (formation placement, `GameGater`-driven `_load_game()`) far
    enough to have a real `sim_controller`/`current_game_frame`/refiners,
    then applies `scenario` exactly once and records what happened
    (`apply_scenario`'s own `verify=True` check, including its internal
    multi-tick settle - see `apply_scenario`'s docstring)."""

    n_episodes = 1

    def __init__(self, scenario):
        super().__init__()
        self.scenario = scenario
        self.applied = False
        self.post_apply_positions: dict = {}
        self.error: Exception | None = None

    def reset_field(self, sim_controller: AbstractSimController, game: Game):
        # Formation doesn't matter - apply_scenario overwrites it - but
        # robots/ball must be in-bounds for GameGater to consider the
        # initial frame valid.
        for rid in game.friendly_robots:
            sim_controller.teleport_robot(game.my_team_is_yellow, rid, -4.0, -2.0 + rid * 0.5, 0.0)
        sim_controller.teleport_ball(0.0, 0.0)

    def eval_status(self, game: Game) -> TestingStatus:
        if not self.applied:
            self.applied = True
            runner = self._runner
            try:
                apply_scenario(runner, self.scenario, verify=True)
            except Exception as e:  # noqa: BLE001 - captured for the test to assert on
                self.error = e
                return TestingStatus.FAILURE

            self.post_apply_positions = _snapshot(runner.my.current_game_frame)
            return TestingStatus.SUCCESS
        return TestingStatus.IN_PROGRESS


def _snapshot(frame) -> dict:
    out = {}
    if frame.ball is not None:
        out["ball"] = (frame.ball.p.x, frame.ball.p.y)
    for rid, r in frame.friendly_robots.items():
        out[("friendly", rid)] = (r.p.x, r.p.y)
    for rid, r in frame.enemy_robots.items():
        out[("enemy", rid)] = (r.p.x, r.p.y)
    return out


def _build_runner_and_scenario(scenario):
    from utama_core.run import StrategyRunner

    n_friendly = max((r.id for r in scenario.friendly_robots), default=-1) + 1
    n_enemy = max((r.id for r in scenario.enemy_robots), default=-1) + 1

    runner = StrategyRunner(
        strategy=_idle_strategy(),
        opp_strategy=_idle_strategy() if n_enemy > 0 else None,
        my_team_is_yellow=True,
        my_team_is_right=True,
        mode="rsim",
        exp_friendly=n_friendly,
        exp_enemy=n_enemy,
        enable_vision_stream=False,
    )
    return runner


def _robots(*positions) -> tuple[RobotState, ...]:
    return tuple(RobotState(id=i, x=x, y=y, orientation=0.0, vx=0.0, vy=0.0) for i, (x, y) in enumerate(positions))


# A mid-play position with both teams spread over the field and the ball away from
# every robot. Built by hand so the test doesn't need a replay from `replays/`, which
# isn't in the repo.
_SCENARIO = Scenario(
    sim_time=15.0,
    ball_x=0.8,
    ball_y=-0.6,
    ball_vx=0.0,
    ball_vy=0.0,
    friendly_robots=_robots((3.5, 0.0), (1.5, 1.2), (2.0, -1.5)),
    enemy_robots=_robots((-3.5, 0.0), (-1.0, 1.8), (-1.6, -0.9)),
    referee_command=RefereeCommand.NORMAL_START,
    source_replay=Path("hand-built"),
)


def test_apply_scenario_positions_match_within_tolerance():
    runner = _build_runner_and_scenario(_SCENARIO)
    manager = _ApplyScenarioTestManager(_SCENARIO)
    manager._runner = runner  # noqa: SLF001 - test-only wiring, see eval_status

    # run_test() closes the runner itself (rsim env, sim_controller, ...)
    # in its own finally block - closing it again here would double-close.
    passed = runner.run_test(test_manager=manager, episode_timeout=10.0, rsim_headless=True)

    assert manager.error is None, f"apply_scenario raised: {manager.error}"
    assert passed, "test episode did not complete"
    assert manager.applied

    _assert_close_to_scenario(manager.post_apply_positions, _SCENARIO, tol=0.15)


def _make_position_refiner():
    from utama_core.data_processing.refiners import PositionRefiner

    # exp_ball=False: this test only exercises the robot Kalman filter in
    # isolation, no ball is fed in.
    return PositionRefiner(STANDARD_FIELD_DIMS, filtering=True, exp_ball=False)


def _vision_frame(ts: float, x: float, y: float, orientation: float = 0.0):
    """A single-camera `RawVisionData` reporting robot 0 (yellow, friendly)
    at `(x, y)` with no ball - enough for `PositionRefiner.refine()` to run
    its Kalman update for that one robot in isolation."""
    from utama_core.entities.data.raw_vision import RawRobotData, RawVisionData

    return RawVisionData(
        ts=ts,
        yellow_robots=[RawRobotData(id=0, x=x, y=y, orientation=orientation, confidence=1.0)],
        blue_robots=[],
        balls=[],
        camera_id=0,
    )


def _make_frame(ts: float, robot):
    from utama_core.entities.data.vector import Vector2D
    from utama_core.entities.game import GameFrame, Robot

    return GameFrame(
        ts=ts,
        my_team_is_yellow=True,
        my_team_is_right=True,
        friendly_robots={0: robot},
        enemy_robots={},
        ball=None,
    )


def _robot_at(x: float, y: float, *, v=(0.0, 0.0)):
    from utama_core.entities.data.vector import Vector2D
    from utama_core.entities.game import Robot

    return Robot(
        id=0, is_friendly=True, has_ball=False, p=Vector2D(x, y), v=Vector2D(*v), a=Vector2D(0, 0), orientation=0.0
    )


def test_kalman_seed_without_reseed_lags_behind_teleport_target():
    """Pure unit-level reproduction of the bug `_overwrite_current_game_frame`
    fixes, isolated from rsim's own multi-tick physics/vision settling (see
    `test_apply_scenario_positions_match_within_tolerance`'s docstring note
    on why an rsim-level test is a noisy way to isolate this specific
    mechanism - field-boundary bounce and vision-buffer lag dominate there).

    Directly exercises `PositionRefiner`: settle it at an old position for
    several ticks (so its Kalman filter's covariance shrinks, mimicking a
    robot that's been sitting in its initial formation spot), then simulate
    a teleport by feeding it a vision measurement at a new, far-away
    position *without* first resetting the refiner or its seed frame - the
    same thing `apply_scenario` used to do before the fix. The filtered
    output should lag well behind the new measurement (a large gap remains
    between the two), because the filter is still blending in its old,
    tight-covariance state - not landing on the new position outright.
    """
    refiner = _make_position_refiner()
    refiner.start_filtering()  # matches StrategyRunner._load_game()'s real sequencing
    old_x, old_y = 0.0, 0.0
    new_x, new_y = 4.0, 0.0  # a large, single-tick "teleport" jump

    frame = _make_frame(0.0, _robot_at(old_x, old_y))
    for i in range(1, 31):
        frame = refiner.refine(frame, [_vision_frame(i * (1 / 60), old_x, old_y)])
    assert abs(frame.friendly_robots[0].p.x - old_x) < 0.01  # settled at the old position

    # Teleport: feed a new measurement WITHOUT resetting the refiner or
    # overwriting `frame` first - reproduces the bug (frame.friendly_robots[0]
    # is still the stale old-position Robot the filter was just seeded from).
    next_frame = refiner.refine(frame, [_vision_frame(31 * (1 / 60), new_x, new_y)])

    gap_to_target = abs(next_frame.friendly_robots[0].p.x - new_x)
    assert gap_to_target > 1.0, (
        f"expected the unseeded/un-reset refiner to lag well behind the teleport target "
        f"(large gap to new_x={new_x}), got x={next_frame.friendly_robots[0].p.x:.3f} "
        f"(gap={gap_to_target:.3f}) - if this fails, the bug this regression test pins may "
        f"no longer reproduce and the test's premise should be revisited"
    )


def test_kalman_seed_with_reseed_lands_on_teleport_target():
    """The fixed counterpart of the test above: after settling the same way,
    apply the same fix `_overwrite_current_game_frame` + `reset()` +
    `start_filtering()` perform (overwrite the seed frame to the new
    position, reset the Kalman filter, mark filtering active again) before
    feeding the same new-position measurement - the very next filtered
    output should land close to the target, not lag behind it like the
    unfixed case above.
    """
    refiner = _make_position_refiner()
    refiner.start_filtering()  # matches StrategyRunner._load_game()'s real sequencing
    old_x, old_y = 0.0, 0.0
    new_x, new_y = 4.0, 0.0

    frame = _make_frame(0.0, _robot_at(old_x, old_y))
    for i in range(1, 31):
        frame = refiner.refine(frame, [_vision_frame(i * (1 / 60), old_x, old_y)])
    assert abs(frame.friendly_robots[0].p.x - old_x) < 0.01

    # The fix: overwrite the seed frame to the teleport target, then reset
    # + restart the filter - mirrors `apply_scenario`'s
    # `_overwrite_current_game_frame` + `position_refiner.reset()` +
    # `start_filtering()` sequence exactly.
    frame = _make_frame(frame.ts, _robot_at(new_x, new_y))
    refiner.reset()
    refiner.start_filtering()

    next_frame = refiner.refine(frame, [_vision_frame(31 * (1 / 60), new_x, new_y)])

    gap_to_target = abs(next_frame.friendly_robots[0].p.x - new_x)
    assert gap_to_target < 0.05, (
        f"expected the reseeded refiner to land on the teleport target, got "
        f"x={next_frame.friendly_robots[0].p.x:.3f} (gap={gap_to_target:.3f})"
    )


def _assert_close_to_scenario(positions: dict, scenario, *, tol: float, context: str = "") -> None:
    label = f" ({context})" if context else ""
    if "ball" in positions:
        x, y = positions["ball"]
        dx, dy = abs(x - scenario.ball_x), abs(y - scenario.ball_y)
        assert dx <= tol and dy <= tol, (
            f"ball{label} at ({x:.3f}, {y:.3f}) deviates from scenario "
            f"({scenario.ball_x:.3f}, {scenario.ball_y:.3f}) by more than {tol}m"
        )
    for robot in scenario.friendly_robots:
        key = ("friendly", robot.id)
        if key not in positions:
            continue
        x, y = positions[key]
        dx, dy = abs(x - robot.x), abs(y - robot.y)
        assert dx <= tol and dy <= tol, (
            f"friendly robot {robot.id}{label} at ({x:.3f}, {y:.3f}) deviates from scenario "
            f"({robot.x:.3f}, {robot.y:.3f}) by more than {tol}m"
        )
    for robot in scenario.enemy_robots:
        key = ("enemy", robot.id)
        if key not in positions:
            continue
        x, y = positions[key]
        dx, dy = abs(x - robot.x), abs(y - robot.y)
        assert dx <= tol and dy <= tol, (
            f"enemy robot {robot.id}{label} at ({x:.3f}, {y:.3f}) deviates from scenario "
            f"({robot.x:.3f}, {robot.y:.3f}) by more than {tol}m"
        )
