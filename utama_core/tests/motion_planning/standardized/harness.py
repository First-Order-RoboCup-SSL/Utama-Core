"""Reusable black-box harness for comparing motion-control schemes in RSim.

The harness deliberately knows nothing about planner internals.  A scenario
describes only initial poses, destinations, and externally observable safety
requirements; every control scheme is driven through ``StrategyRunner``'s
public ``control_scheme`` option.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot
from typing import Callable

from utama_core.config.physical_constants import ROBOT_RADIUS
from utama_core.engine.abstract_strategy import AbstractStrategy
from utama_core.entities.game import Game
from utama_core.entities.game.robot import Robot
from utama_core.run import StrategyRunner
from utama_core.team_controller.src.controllers import AbstractSimController
from utama_core.tests.common.abstract_test_manager import (
    AbstractTestManager,
    TestingStatus,
)
from utama_core.tests.motion_planning._kernel_test_strategies import (
    go_to_point_strategy,
    go_to_trajectory_strategy,
)

Point = tuple[float, float]
RobotKey = tuple[str, int]

# A moving target's position as a function of elapsed scenario time (seconds).
TargetTrajectory = Callable[[float], Point]


@dataclass(frozen=True)
class Disturbance:
    """A scripted mid-scenario event: teleport a robot at a fixed elapsed time.

    Used for two scenario shapes that a fixed initial ``reset_field`` cannot
    express: an obstacle that appears suddenly (teleport a passive enemy from
    off-field into the corridor partway through), and stale-plan recovery
    (teleport a controlled friendly robot off its planned path and check it
    replans rather than continuing to execute a now-invalid trajectory).
    """

    at_time: float
    team: str  # "friendly" or "enemy", matching RobotKey's first element
    robot_id: int
    position: Point
    theta: float = 0.0


@dataclass(frozen=True)
class MotionScenario:
    """A planner-independent navigation problem.

    Enemy robots without ``enemy_targets`` are passive obstacles.  Supplying
    enemy targets makes the opponent use the same selected control scheme,
    which is useful for symmetric cross-team scenarios.

    ``friendly_target_trajectories``/``enemy_target_trajectories`` optionally
    override the corresponding fixed ``*_targets`` entry for a robot with a
    time-varying target (for interception-style scenarios); the fixed target
    in ``*_targets`` is still required and used as the final-time reference
    for ``final_errors``/direct-distance bookkeeping in callers that need it.
    ``disturbances`` are scripted mid-run teleports (see ``Disturbance``).
    """

    name: str
    friendly_starts: tuple[Point, ...]
    friendly_targets: tuple[Point, ...]
    enemy_starts: tuple[Point, ...] = ()
    enemy_targets: tuple[Point, ...] = ()
    endpoint_tolerance: float = 0.15
    collision_distance: float = 2.0 * ROBOT_RADIUS
    timeout: float = 8.0
    friendly_target_trajectories: dict[int, TargetTrajectory] = field(default_factory=dict)
    enemy_target_trajectories: dict[int, TargetTrajectory] = field(default_factory=dict)
    disturbances: tuple[Disturbance, ...] = ()

    def __post_init__(self) -> None:
        if not self.friendly_starts:
            raise ValueError("a scenario needs at least one friendly robot")
        if len(self.friendly_starts) != len(self.friendly_targets):
            raise ValueError("every friendly robot needs exactly one target")
        if self.enemy_targets and len(self.enemy_starts) != len(self.enemy_targets):
            raise ValueError("every controlled enemy robot needs exactly one target")

    @property
    def controlled_keys(self) -> tuple[RobotKey, ...]:
        friendly = tuple(("friendly", robot_id) for robot_id in range(len(self.friendly_targets)))
        enemy = tuple(("enemy", robot_id) for robot_id in range(len(self.enemy_targets)))
        return friendly + enemy


@dataclass
class ScenarioMetrics:
    """Measurements shared by all planner implementations."""

    samples: int = 0
    elapsed: float = 0.0
    minimum_separation: float = float("inf")
    peak_speed: float = 0.0
    path_lengths: dict[RobotKey, float] = field(default_factory=dict)
    final_errors: dict[RobotKey, float] = field(default_factory=dict)
    collision_pair: tuple[RobotKey, RobotKey] | None = None
    _first_timestamp: float | None = field(default=None, repr=False)
    _previous_positions: dict[RobotKey, Point] = field(default_factory=dict, repr=False)

    @property
    def total_path_length(self) -> float:
        return sum(self.path_lengths.values())

    def summary(self) -> str:
        separation = "n/a" if self.minimum_separation == float("inf") else f"{self.minimum_separation:.3f}m"
        errors = ", ".join(f"{team}[{rid}]={error:.3f}m" for (team, rid), error in self.final_errors.items())
        return (
            f"elapsed={self.elapsed:.2f}s, samples={self.samples}, path={self.total_path_length:.2f}m, "
            f"min_separation={separation}, peak_speed={self.peak_speed:.2f}m/s, final_errors=[{errors}]"
        )

    def as_dict(self) -> dict[str, float]:
        """Return scalar values suitable for pytest/JUnit properties."""

        return {
            "elapsed_seconds": self.elapsed,
            "samples": float(self.samples),
            "total_path_length_metres": self.total_path_length,
            "minimum_separation_metres": self.minimum_separation,
            "peak_speed_metres_per_second": self.peak_speed,
            "maximum_final_error_metres": max(self.final_errors.values(), default=0.0),
        }


class StandardizedScenarioManager(AbstractTestManager):
    """Collect common metrics and enforce common goal/collision contracts."""

    n_episodes = 1

    def __init__(self, scenario: MotionScenario):
        super().__init__()
        self.scenario = scenario
        self.metrics = ScenarioMetrics()
        self._sim_controller: AbstractSimController | None = None
        self._is_team_yellow: bool | None = None
        self._pending_disturbances: list[Disturbance] = list(scenario.disturbances)
        self._first_ts: float | None = None

    def reset_field(self, sim_controller: AbstractSimController, game: Game) -> None:
        self._sim_controller = sim_controller
        self._is_team_yellow = game.my_team_is_yellow
        self._pending_disturbances = list(self.scenario.disturbances)
        self._first_ts = None
        for robot_id, (x, y) in enumerate(self.scenario.friendly_starts):
            sim_controller.teleport_robot(game.my_team_is_yellow, robot_id, x, y, 0.0)
        for robot_id, (x, y) in enumerate(self.scenario.enemy_starts):
            sim_controller.teleport_robot(not game.my_team_is_yellow, robot_id, x, y, 0.0)
        self.metrics = ScenarioMetrics(
            path_lengths={key: 0.0 for key in self.scenario.controlled_keys},
            _previous_positions={
                **{("friendly", i): point for i, point in enumerate(self.scenario.friendly_starts)},
                **{
                    ("enemy", i): point
                    for i, point in enumerate(self.scenario.enemy_starts)
                    if self.scenario.enemy_targets
                },
            },
        )

    def _apply_due_disturbances(self, game: Game) -> None:
        """Fire any scripted teleport whose scheduled time has arrived.

        Applied at most once per `Disturbance` (removed from the pending list
        once fired), keyed off elapsed scenario time rather than `game.ts`
        directly, matching `_record_metrics`'s own elapsed-time convention.
        """
        if self._sim_controller is None or not self._pending_disturbances:
            return
        if self._first_ts is None:
            self._first_ts = game.ts
        elapsed = game.ts - self._first_ts

        still_pending = []
        for disturbance in self._pending_disturbances:
            if elapsed >= disturbance.at_time:
                is_team_yellow = self._is_team_yellow if disturbance.team == "friendly" else not self._is_team_yellow
                x, y = disturbance.position
                self._sim_controller.teleport_robot(is_team_yellow, disturbance.robot_id, x, y, disturbance.theta)
                key = (disturbance.team, disturbance.robot_id)
                self.metrics._previous_positions[key] = disturbance.position
            else:
                still_pending.append(disturbance)
        self._pending_disturbances = still_pending

    @staticmethod
    def _robots(game: Game) -> dict[RobotKey, Robot]:
        return {
            **{("friendly", robot_id): robot for robot_id, robot in game.friendly_robots.items()},
            **{("enemy", robot_id): robot for robot_id, robot in game.enemy_robots.items()},
        }

    def _record_metrics(self, game: Game, robots: dict[RobotKey, Robot]) -> None:
        metrics = self.metrics
        if metrics._first_timestamp is None:
            metrics._first_timestamp = game.ts
        metrics.elapsed = max(0.0, game.ts - metrics._first_timestamp)
        metrics.samples += 1

        for key in self.scenario.controlled_keys:
            robot = robots[key]
            position = (robot.p.x, robot.p.y)
            previous = metrics._previous_positions.get(key, position)
            metrics.path_lengths[key] += hypot(position[0] - previous[0], position[1] - previous[1])
            metrics._previous_positions[key] = position
            metrics.peak_speed = max(metrics.peak_speed, hypot(robot.v.x, robot.v.y))

        keys = tuple(robots)
        for index, first_key in enumerate(keys):
            first = robots[first_key]
            for second_key in keys[index + 1 :]:
                second = robots[second_key]
                separation = hypot(first.p.x - second.p.x, first.p.y - second.p.y)
                if separation < metrics.minimum_separation:
                    metrics.minimum_separation = separation
                if separation < self.scenario.collision_distance and metrics.collision_pair is None:
                    metrics.collision_pair = (first_key, second_key)

    def _current_target(self, key: RobotKey, elapsed: float) -> Point:
        """The live target for `key`: the scenario trajectory override if one
        exists for that robot (interception scenarios), else its fixed target.
        """
        team, robot_id = key
        trajectories = (
            self.scenario.friendly_target_trajectories
            if team == "friendly"
            else self.scenario.enemy_target_trajectories
        )
        trajectory = trajectories.get(robot_id)
        if trajectory is not None:
            return trajectory(elapsed)
        fixed = self.scenario.friendly_targets if team == "friendly" else self.scenario.enemy_targets
        return fixed[robot_id]

    def _record_goal_errors(self, robots: dict[RobotKey, Robot], elapsed: float) -> bool:
        targets = {
            **{("friendly", i): target for i, target in enumerate(self.scenario.friendly_targets)},
            **{("enemy", i): target for i, target in enumerate(self.scenario.enemy_targets)},
        }
        self.metrics.final_errors = {
            key: hypot(robots[key].p.x - target[0], robots[key].p.y - target[1]) for key, target in targets.items()
        }
        # A robot with a moving-target override is judged against its live
        # target (interception: reaching the moving point at any time counts
        # as success), not the scenario's fixed `*_targets` reference point.
        live_at_goal = True
        for key in targets:
            live_target = self._current_target(key, elapsed)
            live_error = hypot(robots[key].p.x - live_target[0], robots[key].p.y - live_target[1])
            if live_error > self.scenario.endpoint_tolerance:
                live_at_goal = False
        return live_at_goal

    def eval_status(self, game: Game) -> TestingStatus:
        self._apply_due_disturbances(game)
        robots = self._robots(game)
        self._record_metrics(game, robots)
        elapsed = self.metrics.elapsed
        all_at_goal = self._record_goal_errors(robots, elapsed)
        if self.metrics.collision_pair is not None:
            return TestingStatus.FAILURE
        if all_at_goal:
            return TestingStatus.SUCCESS
        if elapsed >= self.scenario.timeout:
            return TestingStatus.FAILURE
        return TestingStatus.IN_PROGRESS


def _build_side_strategy(
    fixed_targets: tuple[Point, ...],
    target_trajectories: dict[int, TargetTrajectory],
) -> AbstractStrategy:
    """`go_to_point_strategy` for a side with no moving targets; otherwise
    `go_to_trajectory_strategy` with every fixed target wrapped as a
    constant trajectory, so one team can mix moving and stationary targets.
    """
    if not target_trajectories:
        return go_to_point_strategy(dict(enumerate(fixed_targets)))

    def _as_trajectory(robot_id: int, fixed: Point) -> TargetTrajectory:
        override = target_trajectories.get(robot_id)
        return override if override is not None else (lambda _elapsed, _fixed=fixed: _fixed)

    robot_targets = {robot_id: _as_trajectory(robot_id, fixed) for robot_id, fixed in enumerate(fixed_targets)}
    return go_to_trajectory_strategy(robot_targets)


def run_scenario(scenario: MotionScenario, control_scheme: str, *, headless: bool) -> tuple[bool, ScenarioMetrics]:
    """Run one scenario through the same public runner path for every scheme."""

    friendly_strategy = _build_side_strategy(scenario.friendly_targets, scenario.friendly_target_trajectories)
    opponent_strategy = (
        _build_side_strategy(scenario.enemy_targets, scenario.enemy_target_trajectories)
        if scenario.enemy_targets
        else None
    )
    runner = StrategyRunner(
        strategy=friendly_strategy,
        opp_strategy=opponent_strategy,
        control_scheme=control_scheme,
        opp_control_scheme=control_scheme if opponent_strategy else None,
        my_team_is_yellow=True,
        my_team_is_right=False,
        mode="rsim",
        exp_friendly=len(scenario.friendly_starts),
        exp_enemy=len(scenario.enemy_starts),
        exp_ball=False,
        filtering=False,
        enable_vision_stream=False,
    )
    manager = StandardizedScenarioManager(scenario)
    # `run_test`'s `episode_timeout` is wall-clock (`time.time()`-based), not
    # simulated time, and headless rsim runs much faster than real-time --
    # passing `scenario.timeout` directly here let scenarios run for minutes
    # of simulated time before this wall-clock safety net finally fired.
    # `eval_status` above enforces `scenario.timeout` against simulated
    # elapsed time; this wall-clock bound only guards against the runner
    # itself stalling (e.g. an unresponsive sim process).
    wall_timeout = max(30.0, scenario.timeout * 10.0)
    passed = runner.run_test(manager, episode_timeout=wall_timeout, rsim_headless=headless)
    return passed, manager.metrics
