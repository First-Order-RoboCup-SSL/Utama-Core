#!/usr/bin/env python3
# ruff: noqa: E402
"""Repeatable, cross-controller motion-planning benchmark for headless rsim.

Run from the repository root, for example:

    pixi run python tools/motion_planning_benchmark.py
    pixi run python tools/motion_planning_benchmark.py --scenarios direct static_slalom

The tool deliberately exercises every controller through StrategyRunner and the
normal ``move()`` path.  It writes a raw JSON artifact and a Markdown comparison
report; it does not add pytest cases or assert implementation-specific behavior.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

# Direct execution puts ``tools/`` rather than the repository root on
# ``sys.path``.  Keep the documented ``python tools/...`` invocation working
# without requiring an editable package install.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utama_core.config.field_params import STANDARD_FIELD_DIMS
from utama_core.config.physical_constants import ROBOT_RADIUS
from utama_core.config.referee_constants import OPPONENT_DEFENSE_AREA_KEEP_DISTANCE
from utama_core.config.robot_params import RSIM_PARAMS
from utama_core.engine.abstract_strategy import AbstractStrategy
from utama_core.engine.context import TickContext
from utama_core.engine.strategy import Strategy as KernelStrategy
from utama_core.engine.tactic import BaseTactic, RobotId, TacticTag
from utama_core.entities.data.command import RobotCommand
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game
from utama_core.motion_planning.src.common.motion_controller import MotionController
from utama_core.run import StrategyRunner
from utama_core.skills.src.utils.move_utils import move
from utama_core.team_controller.src.controllers import AbstractSimController
from utama_core.tests.common.abstract_test_manager import (
    AbstractTestManager,
    TestingStatus,
)

Point = tuple[float, float]
RobotKey = tuple[str, int]
TargetTrajectory = Callable[[float], Point]
SCHEMES = ("fpp", "dwa", "trajsample")
COLLISION_DISTANCE = 2.0 * ROBOT_RADIUS
STOPPED_SPEED = 0.05
SPEED_LIMIT_TOLERANCE = 0.15
# Command-velocity jump between consecutive `calculate()` calls beyond which a
# tick counts as a "discontinuity" in the reported diagnostic. Chosen well
# above ordinary frame-to-frame acceleration-limited change (RSIM_PARAMS.
# MAX_ACCELERATION * one 60Hz tick ~= 0.067 m/s) so it flags genuine command
# jumps -- a fresh plan or an emergency brake -- not normal tracking.
COMMAND_JUMP_THRESHOLD_MPS = 0.5
SCHEMA_VERSION = 2


@dataclass(frozen=True)
class Disturbance:
    """A scripted mid-scenario teleport; see the standardized-suite harness's
    identically-named dataclass for the rationale (sudden obstacle
    appearance, stale-plan recovery)."""

    at_time_s: float
    team: str  # "friendly" or "enemy"
    robot_id: int
    position: Point
    theta: float = 0.0


# ~1mm, deliberately generous relative to real physics-sim position jitter
# (~1e-4m/tick) -- see `_make_jitter_trajectory`.
_JITTER_RADIUS_M = 0.001


def _make_jitter_trajectory(center: Point, seed: int) -> TargetTrajectory:
    """A deterministic, per-call ~1mm-jittered target around `center` -- the
    same shape `DirectFreeOursStep`'s kicker-approach point takes when it's
    recomputed every tick from the ball's live (noisy) position, see 5183ed1
    ("Fix trajsample planner target-jitter stall on DIRECT_FREE restarts").
    A fresh `random.Random(seed)` is captured per trajectory (not shared
    module state) so repeated benchmark runs are independently reproducible
    regardless of call order -- matches the standardized suite's identically
    named helper in `tests/motion_planning/standardized/test_scenarios.py`.
    """
    rng = random.Random(seed)

    def _target(_elapsed: float) -> Point:
        angle = rng.uniform(0.0, 2.0 * math.pi)
        radius = rng.uniform(0.0, _JITTER_RADIUS_M)
        return (center[0] + radius * math.cos(angle), center[1] + radius * math.sin(angle))

    return _target


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    friendly_starts: tuple[Point, ...]
    friendly_targets: tuple[Point, ...]
    enemy_starts: tuple[Point, ...] = ()
    enemy_targets: tuple[Point, ...] | None = None
    timeout_s: float = 20.0
    endpoint_tolerance_m: float = 0.2
    friendly_target_trajectories: dict[int, TargetTrajectory] = field(default_factory=dict)
    enemy_target_trajectories: dict[int, TargetTrajectory] = field(default_factory=dict)
    disturbances: tuple[Disturbance, ...] = ()
    # `None` uses the shared `COLLISION_DISTANCE` (2*ROBOT_RADIUS, real
    # physical contact) every other scenario is judged against. Only
    # `start_inside_obstacle` overrides this, to a value tighter than its own
    # deliberately-scripted starting overlap -- see that scenario's comment
    # for why a genuine physical start-overlap is the scenario itself, not a
    # scenario failure, while a real NEW collision picked up while escaping
    # must still fail it.
    collision_distance_m: float | None = None

    @property
    def controls_enemy(self) -> bool:
        return self.enemy_targets is not None

    @property
    def effective_collision_distance_m(self) -> float:
        return COLLISION_DISTANCE if self.collision_distance_m is None else self.collision_distance_m


SCENARIOS = {
    scenario.name: scenario
    for scenario in (
        Scenario(
            name="direct",
            description="One robot traverses six metres with no obstacles.",
            friendly_starts=((-3.0, 0.0),),
            friendly_targets=((3.0, 0.0),),
            timeout_s=12.0,
            endpoint_tolerance_m=0.15,
        ),
        Scenario(
            name="static_slalom",
            description="One robot crosses a staggered line of three stationary opponents.",
            friendly_starts=((-3.0, 0.0),),
            friendly_targets=((3.0, 0.0),),
            enemy_starts=((-1.0, 0.3), (0.0, -0.3), (1.0, 0.3)),
            timeout_s=20.0,
            endpoint_tolerance_m=0.15,
        ),
        Scenario(
            name="crossing",
            description="Two controlled robots cross at right angles through the origin.",
            friendly_starts=((-2.0, 0.0),),
            friendly_targets=((2.0, 0.0),),
            enemy_starts=((0.0, -2.0),),
            enemy_targets=((0.0, 2.0),),
            timeout_s=20.0,
            endpoint_tolerance_m=0.2,
        ),
        Scenario(
            name="crossing_oblique_45",
            description="Two controlled robots cross at 45 degrees, heading the same general way.",
            friendly_starts=((-2.0, 0.0),),
            friendly_targets=((2.0, 0.0),),
            enemy_starts=((-1.4, -1.4),),
            enemy_targets=((1.4, 1.4),),
            timeout_s=20.0,
            endpoint_tolerance_m=0.2,
        ),
        Scenario(
            name="crossing_oblique_135",
            description="Two controlled robots cross at 135 degrees, heading broadly towards each other.",
            friendly_starts=((-2.0, 0.0),),
            friendly_targets=((2.0, 0.0),),
            enemy_starts=((1.4, -1.4),),
            enemy_targets=((-1.4, 1.4),),
            timeout_s=20.0,
            endpoint_tolerance_m=0.2,
        ),
        Scenario(
            name="crossing_offset",
            description="Two controlled robots cross at right angles off-centre, so they reach the crossing at different times.",
            friendly_starts=((-2.0, 0.0),),
            friendly_targets=((2.0, 0.0),),
            enemy_starts=((-0.8, -2.0),),
            enemy_targets=((-0.8, 2.0),),
            timeout_s=20.0,
            endpoint_tolerance_m=0.2,
        ),
        Scenario(
            name="crossing_steady_runner",
            description="An opponent follows a point crossing our path at a steady 1 m/s.",
            friendly_starts=((-2.0, 0.0),),
            friendly_targets=((2.0, 0.0),),
            enemy_starts=((0.0, -2.0),),
            enemy_targets=((0.0, 2.0),),
            enemy_target_trajectories={0: lambda t: (0.0, min(2.0, -2.0 + 1.0 * t))},
            timeout_s=20.0,
            endpoint_tolerance_m=0.2,
        ),
        Scenario(
            name="overtaking",
            description="One robot overtakes an opponent following a point ahead on the same line at 0.5 m/s.",
            friendly_starts=((-2.5, 0.0),),
            friendly_targets=((2.5, 0.0),),
            enemy_starts=((-1.5, 0.0),),
            enemy_targets=((3.5, 0.0),),
            enemy_target_trajectories={0: lambda t: (min(3.5, -1.5 + 0.5 * t), 0.0)},
            timeout_s=20.0,
            endpoint_tolerance_m=0.2,
        ),
        Scenario(
            name="grid_intersection",
            description="Two horizontal lanes cross two vertical lanes at four intersections.",
            friendly_starts=((-2.0, 0.5), (-2.0, -0.5)),
            friendly_targets=((2.0, 0.5), (2.0, -0.5)),
            enemy_starts=((0.5, 2.0), (-0.5, 2.0)),
            enemy_targets=((0.5, -2.0), (-0.5, -2.0)),
            timeout_s=30.0,
            endpoint_tolerance_m=0.25,
        ),
        Scenario(
            name="mirror_swap",
            description="Six-versus-six mirrored swap with a deterministic 2 cm symmetry break.",
            friendly_starts=(
                (-2.5, -1.5),
                (-2.5, -0.5),
                (-2.5, 0.5),
                (-2.5, 1.5),
                (-3.5, -0.75),
                (-3.5, 0.75),
            ),
            friendly_targets=(
                (2.5, -1.5),
                (2.5, -0.5),
                (2.5, 0.5),
                (2.5, 1.5),
                (3.5, -0.75),
                (3.5, 0.75),
            ),
            enemy_starts=(
                (2.5, -1.48),
                (2.5, -0.48),
                (2.5, 0.52),
                (2.5, 1.52),
                (3.5, -0.73),
                (3.5, 0.77),
            ),
            enemy_targets=(
                (-2.5, -1.5),
                (-2.5, -0.5),
                (-2.5, 0.5),
                (-2.5, 1.5),
                (-3.5, -0.75),
                (-3.5, 0.75),
            ),
            timeout_s=45.0,
            endpoint_tolerance_m=0.3,
        ),
        Scenario(
            name="narrow_passage",
            description="One robot threads a 0.24m gap between two static obstacles.",
            friendly_starts=((-2.0, 0.0),),
            friendly_targets=((2.0, 0.0),),
            enemy_starts=((0.0, 0.21), (0.0, -0.21)),
            timeout_s=10.0,
            endpoint_tolerance_m=0.15,
        ),
        Scenario(
            name="head_on_swap",
            description="Two robots swap positions driving directly at each other.",
            friendly_starts=((-1.5, 0.0),),
            friendly_targets=((1.5, 0.0),),
            enemy_starts=((1.5, 0.0),),
            enemy_targets=((-1.5, 0.0),),
            timeout_s=10.0,
            endpoint_tolerance_m=0.2,
        ),
        Scenario(
            name="field_boundary_corner",
            description="Target sits just inside a real field corner.",
            friendly_starts=((0.0, 0.0),),
            friendly_targets=(
                (
                    STANDARD_FIELD_DIMS.full_field_half_length - 0.2,
                    STANDARD_FIELD_DIMS.full_field_half_width - 0.2,
                ),
            ),
            timeout_s=10.0,
            endpoint_tolerance_m=0.15,
        ),
        Scenario(
            name="defense_area_boundary",
            description="Target sits just outside the enforced defense-area keep-distance.",
            friendly_starts=((0.0, 0.0),),
            friendly_targets=(
                (
                    STANDARD_FIELD_DIMS.full_field_half_length
                    - 2 * STANDARD_FIELD_DIMS.half_defense_area_depth
                    - OPPONENT_DEFENSE_AREA_KEEP_DISTANCE
                    - 0.1,
                    0.0,
                ),
            ),
            timeout_s=10.0,
            # Looser than the other fixed-target scenarios: FPP's own
            # defense-area clamp settles ~0.2m short of this target
            # (verified directly against the standardized-suite equivalent).
            endpoint_tolerance_m=0.25,
        ),
        Scenario(
            name="interception",
            description="One robot intercepts an opponent-controlled point moving at constant velocity.",
            friendly_starts=((-2.0, -1.5),),
            friendly_targets=((0.0, 0.0),),
            enemy_starts=((-1.0, 1.0),),
            enemy_targets=((1.0, -1.0),),
            enemy_target_trajectories={0: lambda t: (-1.0 + 0.25 * t, 1.0 - 0.25 * t)},
            timeout_s=8.0,
            endpoint_tolerance_m=0.2,
        ),
        Scenario(
            name="sudden_obstacle",
            description="A corridor is clear for 2s, then an obstacle appears mid-path.",
            friendly_starts=((-2.0, 0.0),),
            friendly_targets=((2.0, 0.0),),
            enemy_starts=((3.5, 2.99),),
            disturbances=(Disturbance(at_time_s=2.0, team="enemy", robot_id=0, position=(0.0, 0.0)),),
            timeout_s=12.0,
            endpoint_tolerance_m=0.15,
        ),
        Scenario(
            name="disturbance_recovery",
            description="A robot is knocked off its planned path partway through and must replan.",
            friendly_starts=((-2.0, 0.0),),
            friendly_targets=((2.0, 0.0),),
            disturbances=(Disturbance(at_time_s=1.5, team="friendly", robot_id=0, position=(-1.0, 1.5)),),
            timeout_s=12.0,
            endpoint_tolerance_m=0.15,
        ),
        Scenario(
            name="start_inside_obstacle",
            description="A robot begins overlapping a stationary obstacle and must escape to a distant target.",
            # 0.10m separation is genuinely inside physical contact
            # (2*ROBOT_RADIUS = 0.18m); see 2e53f3e ("Fix trajsample planner
            # deadlock when a robot starts inside an obstacle") -- the
            # collision-avoidance margin at a stationary robot's own start
            # (speed 0) is 0, so only a true overlap makes every candidate
            # direction report an instant collision against the robot's own
            # starting position regardless of where it points. Uses a
            # scenario-local `collision_distance_m` tighter than this
            # deliberate starting overlap (see `Scenario.collision_distance_m`)
            # so the scripted start isn't itself flagged, while still
            # requiring the robot to separate well past the real 0.18m
            # contact radius to reach the 2m-distant target -- a genuine new
            # collision picked up while escaping is still caught.
            friendly_starts=((0.10, 0.0),),
            friendly_targets=((2.0, 0.0),),
            enemy_starts=((0.0, 0.0),),
            timeout_s=10.0,
            endpoint_tolerance_m=0.15,
            collision_distance_m=0.05,
        ),
        Scenario(
            name="jittering_target",
            description="The target is recomputed every tick with ~1mm deterministic jitter around a fixed point.",
            # The same pattern DirectFreeOursStep's kicker-approach point
            # exhibits when it's re-derived from the ball's live, noisy
            # position every tick -- see 5183ed1 ("Fix trajsample planner
            # target-jitter stall on DIRECT_FREE restarts"). Compare this
            # cell's `sim_time_mean_s` against the `direct` scenario's (same
            # geometry, fixed target) to see the relative slowdown a planner
            # that replans from t=0 on every sub-millimetre "change" incurs.
            friendly_starts=((-3.0, 0.0),),
            friendly_targets=((3.0, 0.0),),
            friendly_target_trajectories={0: _make_jitter_trajectory((3.0, 0.0), seed=0)},
            timeout_s=12.0,
            endpoint_tolerance_m=0.15,
        ),
    )
}


@dataclass
class _MoveMem:
    start_ts: float | None = None


class _MoveToTargetsTactic(BaseTactic[_MoveMem]):
    """Drives each robot to a per-robot, optionally time-varying target.

    `targets` entries are either a fixed `Point` or a `TargetTrajectory`
    (`elapsed_seconds -> Point`), evaluated against time since this tactic's
    own first tick so target motion always starts at the scenario's own t=0.
    """

    tag = TacticTag.MIXED

    def __init__(self, targets: dict[int, Point | TargetTrajectory]):
        self.targets = targets

    def initial_mem(self) -> _MoveMem:
        return _MoveMem()

    def tick(
        self,
        game: Game,
        ctx: TickContext,
        robot_ids: tuple[RobotId, ...],
        mem: _MoveMem,
    ) -> tuple[dict[RobotId, RobotCommand], _MoveMem]:
        if mem.start_ts is None:
            mem.start_ts = game.ts
        elapsed = game.ts - mem.start_ts

        commands = {}
        for robot_id in robot_ids:
            target = self.targets[robot_id]
            point = target(elapsed) if callable(target) else target
            commands[robot_id] = move(game, ctx.motion_controller, robot_id, Vector2D(*point), 0.0)
        return commands, mem


def _go_to_targets_strategy(
    fixed_targets: tuple[Point, ...],
    target_trajectories: dict[int, TargetTrajectory] | None = None,
) -> AbstractStrategy:
    target_trajectories = target_trajectories or {}
    target_map: dict[int, Point | TargetTrajectory] = {
        robot_id: target_trajectories.get(robot_id, fixed) for robot_id, fixed in enumerate(fixed_targets)
    }

    def build(motion_controller: MotionController) -> KernelStrategy:
        tactic = _MoveToTargetsTactic(target_map)
        return KernelStrategy(
            tactics={"benchmark_move": tactic},
            partitioner=KernelStrategy.single_tactic_picker(lambda game, active: "benchmark_move"),
            outfield_robot_ids=tuple(target_map),
            ctx=TickContext(motion_controller=motion_controller),
        )

    return AbstractStrategy(build_kernel_strategy=build, exp_ball=False)


class _ControllerTimer:
    """Wraps `MotionController.calculate()` to time it and, per-robot, track
    consecutive-command discontinuities and a black-box brake-event proxy.

    `MotionController.calculate()`'s actual contract is `tuple[Vector2D,
    float]` (global-frame commanded velocity, commanded angular velocity) --
    not the local-frame `RobotCommand` that `move()` builds one layer above
    it. Command discontinuity and brake/replan-proxy are measured from this
    directly *commanded* velocity, not the simulator's observed
    post-physics robot velocity that `BenchmarkManager` already tracks
    separately -- this is what the controller actually asked for, before
    acceleration limiting or simulator dynamics smooth it out.

    A uniform "replan count" isn't attempted here: FPP, DWA, and trajsample
    each have their own internal notion of committing/replanning a
    trajectory (trajsample's explicit commit, FPP's cached path, DWA's
    every-tick resample), and none of that is observable through the public
    `calculate()` interface this tool is restricted to. A large commanded
    velocity-direction change is used instead as a black-box proxy for "the
    planner just changed its mind" -- it is not equivalent to an internal
    replan count and is reported as `command_direction_changes`, not
    `replans`, to avoid overstating what is actually being measured.
    """

    def __init__(self, warmup_calls: int):
        self.warmup_calls = warmup_calls
        self.calls = 0
        self.samples_s: list[float] = []
        self.command_jumps = 0
        self.command_direction_changes = 0
        self.brake_events = 0
        self._last_command_velocity: Point | None = None
        self._was_braking = False

    def instrument(self, controller: MotionController) -> None:
        original = controller.calculate

        def timed_calculate(*args, **kwargs):
            started = time.perf_counter()
            try:
                command = original(*args, **kwargs)
                return command
            finally:
                elapsed = time.perf_counter() - started
                self.calls += 1
                if self.calls > self.warmup_calls:
                    self.samples_s.append(elapsed)
                    self._observe_command(command)

        controller.calculate = timed_calculate

    def _observe_command(self, command: tuple[Vector2D, float]) -> None:
        velocity_vector, _angular_velocity = command
        velocity = (velocity_vector.x, velocity_vector.y)
        speed = math.hypot(*velocity)
        previous = self._last_command_velocity
        if previous is not None:
            previous_speed = math.hypot(*previous)
            jump = math.dist(previous, velocity)
            if jump >= COMMAND_JUMP_THRESHOLD_MPS:
                self.command_jumps += 1
                # A jump that is mostly a direction change (both commands
                # have meaningful magnitude but point differently) versus one
                # that is mostly a magnitude collapse (a brake) are distinct
                # black-box signals worth telling apart.
                if previous_speed > STOPPED_SPEED and speed > STOPPED_SPEED:
                    cos_angle = (previous[0] * velocity[0] + previous[1] * velocity[1]) / (previous_speed * speed)
                    if cos_angle < 0.5:  # more than ~60 degrees of direction change
                        self.command_direction_changes += 1

            is_braking = speed < STOPPED_SPEED and previous_speed >= RSIM_PARAMS.MAX_VEL * 0.5
            if is_braking and not self._was_braking:
                self.brake_events += 1
            self._was_braking = is_braking
        self._last_command_velocity = velocity


class BenchmarkManager(AbstractTestManager):
    n_episodes = 1

    def __init__(self, scenario: Scenario):
        super().__init__()
        self.scenario = scenario
        self.start_ts: float | None = None
        self.last_ts: float | None = None
        self.last_positions: dict[RobotKey, Point] = {}
        self.last_velocities: dict[RobotKey, Point] = {}
        self.path_length_m = 0.0
        self.max_speed_mps = 0.0
        self.max_observed_acceleration_mps2 = 0.0
        self.stopped_robot_seconds = 0.0
        self.unreached_robot_seconds = 0.0
        self.min_center_distance_m: float | None = None
        self.collision_events = 0
        self.colliding_pairs: set[tuple[RobotKey, RobotKey]] = set()
        self.reached: set[RobotKey] = set()
        self.final_errors: dict[RobotKey, float] = {}
        self.ticks = 0
        self.termination_reason = "not_started"
        self._sim_controller: AbstractSimController | None = None
        self._is_team_yellow: bool | None = None
        self._pending_disturbances: list[Disturbance] = list(scenario.disturbances)
        # A teleport (initial placement or a scripted `Disturbance`) is an
        # instantaneous position jump; the simulator's velocity estimator can
        # report a huge transient speed/acceleration for a few ticks
        # afterwards purely from that discontinuity, not from anything the
        # controller commanded. Suppress `max_speed_mps`/
        # `max_observed_acceleration_mps2` (and therefore `speed_limit_exceeded`)
        # for a short settling window per teleported robot -- the same
        # estimator-transient caveat docs/motion_planning_comparison.md
        # already documents for acceleration applies here too.
        self._settling_until_tick: dict[RobotKey, int] = {}
        # Measured directly (see the comment above): the simulator's
        # velocity estimator takes roughly 6-7 ticks to decay back under
        # normal speeds after a teleport's position discontinuity. 15 ticks
        # (0.25s at 60Hz) leaves comfortable margin above that.
        self._teleport_settle_ticks = 15

    def _mark_settling(self, key: RobotKey) -> None:
        self._settling_until_tick[key] = self.ticks + self._teleport_settle_ticks

    def reset_field(self, sim_controller: AbstractSimController, game: Game) -> None:
        self._sim_controller = sim_controller
        self._is_team_yellow = game.my_team_is_yellow
        self._pending_disturbances = list(self.scenario.disturbances)
        self._settling_until_tick = {}
        for robot_id, (x, y) in enumerate(self.scenario.friendly_starts):
            sim_controller.teleport_robot(game.my_team_is_yellow, robot_id, x, y, 0.0)
            self._mark_settling(("friendly", robot_id))
        for robot_id, (x, y) in enumerate(self.scenario.enemy_starts):
            sim_controller.teleport_robot(not game.my_team_is_yellow, robot_id, x, y, 0.0)
            self._mark_settling(("enemy", robot_id))

    def _apply_due_disturbances(self, elapsed: float) -> None:
        if self._sim_controller is None or not self._pending_disturbances:
            return
        still_pending = []
        for disturbance in self._pending_disturbances:
            if elapsed >= disturbance.at_time_s:
                is_team_yellow = self._is_team_yellow if disturbance.team == "friendly" else not self._is_team_yellow
                x, y = disturbance.position
                self._sim_controller.teleport_robot(is_team_yellow, disturbance.robot_id, x, y, disturbance.theta)
                key = (disturbance.team, disturbance.robot_id)
                # Avoid crediting the teleport jump itself as travelled path.
                self.last_positions[key] = disturbance.position
                self._mark_settling(key)
            else:
                still_pending.append(disturbance)
        self._pending_disturbances = still_pending

    def _targets(self) -> dict[RobotKey, Point]:
        targets = {("friendly", robot_id): point for robot_id, point in enumerate(self.scenario.friendly_targets)}
        if self.scenario.enemy_targets is not None:
            targets.update({("enemy", robot_id): point for robot_id, point in enumerate(self.scenario.enemy_targets)})
        return targets

    def _live_target(self, key: RobotKey, elapsed: float) -> Point:
        """The fixed target, or the scenario's trajectory override for `key`
        if one exists (interception scenarios) -- see `Scenario`'s
        docstring-equivalent note in the standardized-suite harness."""
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

    @staticmethod
    def _robots(game: Game) -> dict[RobotKey, object]:
        robots = {("friendly", robot_id): robot for robot_id, robot in game.friendly_robots.items()}
        robots.update({("enemy", robot_id): robot for robot_id, robot in game.enemy_robots.items()})
        return robots

    def eval_status(self, game: Game) -> TestingStatus:
        robots = self._robots(game)
        targets = self._targets()
        now = game.ts
        dt = max(0.0, now - self.last_ts) if self.last_ts is not None else 0.0
        self.start_ts = now if self.start_ts is None else self.start_ts
        elapsed = now - self.start_ts
        self.ticks += 1
        self._apply_due_disturbances(elapsed)

        for key, robot in robots.items():
            position = (robot.p.x, robot.p.y)
            velocity = (robot.v.x, robot.v.y)
            speed = math.hypot(*velocity)
            settling = self.ticks <= self._settling_until_tick.get(key, -1)
            if not settling:
                self.max_speed_mps = max(self.max_speed_mps, speed)

            if key in targets:
                previous_position = self.last_positions.get(key)
                if previous_position is not None:
                    self.path_length_m += math.dist(previous_position, position)

                live_target = self._live_target(key, elapsed)
                error = math.dist(position, live_target)
                self.final_errors[key] = error
                if error <= self.scenario.endpoint_tolerance_m:
                    self.reached.add(key)
                elif dt > 0.0:
                    self.unreached_robot_seconds += dt
                    if speed < STOPPED_SPEED:
                        self.stopped_robot_seconds += dt

                previous_velocity = self.last_velocities.get(key)
                if previous_velocity is not None and dt > 1e-6 and not settling:
                    acceleration = math.dist(previous_velocity, velocity) / dt
                    self.max_observed_acceleration_mps2 = max(self.max_observed_acceleration_mps2, acceleration)

            self.last_positions[key] = position
            self.last_velocities[key] = velocity

        new_colliding_pairs: set[tuple[RobotKey, RobotKey]] = set()
        robot_items = sorted(robots.items())
        collision_distance = self.scenario.effective_collision_distance_m
        for index, (key_a, robot_a) in enumerate(robot_items):
            for key_b, robot_b in robot_items[index + 1 :]:
                distance = robot_a.p.distance_to(robot_b.p)
                if self.min_center_distance_m is None or distance < self.min_center_distance_m:
                    self.min_center_distance_m = distance
                if distance < collision_distance:
                    new_colliding_pairs.add((key_a, key_b))

        self.collision_events += len(new_colliding_pairs - self.colliding_pairs)
        self.colliding_pairs = new_colliding_pairs
        self.last_ts = now

        if new_colliding_pairs:
            self.termination_reason = "collision"
            return TestingStatus.FAILURE
        if len(self.reached) == len(targets):
            self.termination_reason = "completed"
            return TestingStatus.SUCCESS
        if now - self.start_ts >= self.scenario.timeout_s:
            self.termination_reason = "sim_timeout"
            return TestingStatus.FAILURE
        return TestingStatus.IN_PROGRESS

    def result(self) -> dict:
        targets = self._targets()
        direct_distance = 0.0
        for robot_id, (start, target) in enumerate(zip(self.scenario.friendly_starts, self.scenario.friendly_targets)):
            direct_distance += math.dist(start, target)
        if self.scenario.enemy_targets is not None:
            for start, target in zip(self.scenario.enemy_starts, self.scenario.enemy_targets):
                direct_distance += math.dist(start, target)

        sim_time = 0.0 if self.start_ts is None or self.last_ts is None else self.last_ts - self.start_ts
        min_clearance = None if self.min_center_distance_m is None else self.min_center_distance_m - COLLISION_DISTANCE
        stopped_ratio = (
            0.0 if self.unreached_robot_seconds <= 0.0 else self.stopped_robot_seconds / self.unreached_robot_seconds
        )
        max_error = max(self.final_errors.values(), default=None)
        # A real physical limit (RSIM_PARAMS.MAX_VEL, plus measurement
        # tolerance) is a hard pass/fail gate, unlike acceleration: observed
        # acceleration is reconstructed from filtered simulator velocity and
        # can include estimator transients (see docs/motion_planning_comparison.md),
        # so it stays diagnostic-only rather than gating `passed`.
        speed_limit_exceeded = self.max_speed_mps > RSIM_PARAMS.MAX_VEL + SPEED_LIMIT_TOLERANCE
        return {
            "passed": self.termination_reason == "completed"
            and self.collision_events == 0
            and not speed_limit_exceeded,
            "termination_reason": self.termination_reason,
            "sim_time_s": sim_time,
            "ticks": self.ticks,
            "robots_reached": len(self.reached),
            "robots_expected": len(targets),
            "final_error_max_m": max_error,
            "path_length_total_m": self.path_length_m,
            "path_length_ratio": self.path_length_m / direct_distance if direct_distance > 0.0 else None,
            "min_center_distance_m": self.min_center_distance_m,
            "min_clearance_m": min_clearance,
            "collision_events": self.collision_events,
            "max_speed_mps": self.max_speed_mps,
            "speed_limit_exceeded": speed_limit_exceeded,
            "max_observed_acceleration_mps2": self.max_observed_acceleration_mps2,
            "stopped_while_unreached_ratio": stopped_ratio,
        }


def _serializable_scenario(scenario: Scenario) -> dict:
    """`asdict(scenario)` directly would try to JSON-encode the trajectory
    callables in `*_target_trajectories`; describe them instead."""
    data = asdict(scenario)
    data["friendly_target_trajectories"] = sorted(scenario.friendly_target_trajectories)
    data["enemy_target_trajectories"] = sorted(scenario.enemy_target_trajectories)
    data["disturbances"] = [asdict(d) for d in scenario.disturbances]
    return data


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[math.ceil(fraction * len(ordered)) - 1]


def _git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def run_case(scheme: str, scenario: Scenario, repeat: int, timing_warmup_calls: int) -> dict:
    manager = BenchmarkManager(scenario)
    my_strategy = _go_to_targets_strategy(scenario.friendly_targets, scenario.friendly_target_trajectories)
    opp_strategy = (
        _go_to_targets_strategy(scenario.enemy_targets, scenario.enemy_target_trajectories)
        if scenario.enemy_targets is not None
        else None
    )
    runner: StrategyRunner | None = None
    wall_started = time.perf_counter()
    timer = _ControllerTimer(timing_warmup_calls)

    try:
        runner = StrategyRunner(
            strategy=my_strategy,
            opp_strategy=opp_strategy,
            my_team_is_yellow=True,
            my_team_is_right=False,
            mode="rsim",
            exp_friendly=len(scenario.friendly_starts),
            exp_enemy=len(scenario.enemy_starts),
            exp_ball=False,
            control_scheme=scheme,
            opp_control_scheme=scheme if opp_strategy is not None else None,
            enable_vision_stream=False,
        )
        timer.instrument(my_strategy.motion_controller)
        if opp_strategy is not None:
            timer.instrument(opp_strategy.motion_controller)
        wall_timeout = max(30.0, scenario.timeout_s * 5.0)
        runner.run_test(manager, episode_timeout=wall_timeout, rsim_headless=True)
        if manager.termination_reason == "not_started":
            manager.termination_reason = "wall_timeout"
        result = manager.result()
    except Exception as exc:  # keep the matrix running and make the failure reportable
        if runner is not None:
            try:
                runner.close()
            except Exception:
                pass
        result = manager.result()
        result.update(
            passed=False,
            termination_reason="exception",
            error=f"{type(exc).__name__}: {exc}",
        )

    timing_ms = [sample * 1000.0 for sample in timer.samples_s]
    result.update(
        scheme=scheme,
        scenario=scenario.name,
        repeat=repeat,
        wall_time_s=time.perf_counter() - wall_started,
        controller_calls=timer.calls,
        controller_timed_calls=len(timing_ms),
        controller_mean_ms=statistics.fmean(timing_ms) if timing_ms else None,
        controller_p95_ms=_percentile(timing_ms, 0.95),
        controller_max_ms=max(timing_ms, default=None),
        # Pooled across every instrumented controller in this cell (friendly,
        # plus enemy when the scenario controls it) -- see `_ControllerTimer`
        # for what each of these does and does not claim to measure.
        command_jumps=timer.command_jumps,
        command_direction_changes=timer.command_direction_changes,
        brake_events_proxy=timer.brake_events,
    )
    return result


def _mean(results: Iterable[dict], key: str) -> float | None:
    values = [result[key] for result in results if result.get(key) is not None]
    return statistics.fmean(values) if values else None


def aggregate_results(results: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for result in results:
        grouped.setdefault((result["scheme"], result["scenario"]), []).append(result)

    aggregates = []
    for (scheme, scenario), runs in grouped.items():
        aggregates.append(
            {
                "scheme": scheme,
                "scenario": scenario,
                "passed": all(run["passed"] for run in runs),
                "pass_rate": sum(run["passed"] for run in runs) / len(runs),
                "termination_reasons": sorted({run["termination_reason"] for run in runs}),
                "sim_time_mean_s": _mean(runs, "sim_time_s"),
                "path_length_ratio_mean": _mean(runs, "path_length_ratio"),
                "min_clearance_worst_m": min(
                    (run["min_clearance_m"] for run in runs if run["min_clearance_m"] is not None),
                    default=None,
                ),
                "collision_events_total": sum(run["collision_events"] for run in runs),
                "final_error_max_worst_m": max(
                    (run["final_error_max_m"] for run in runs if run["final_error_max_m"] is not None),
                    default=None,
                ),
                "stopped_while_unreached_ratio_mean": _mean(runs, "stopped_while_unreached_ratio"),
                "controller_mean_ms": _mean(runs, "controller_mean_ms"),
                "controller_p95_ms_mean": _mean(runs, "controller_p95_ms"),
                "max_speed_worst_mps": max((run["max_speed_mps"] for run in runs), default=None),
                "speed_limit_exceeded": any(run["speed_limit_exceeded"] for run in runs),
                "command_jumps_mean": _mean(runs, "command_jumps"),
                "command_direction_changes_mean": _mean(runs, "command_direction_changes"),
                "brake_events_proxy_mean": _mean(runs, "brake_events_proxy"),
            }
        )
    return aggregates


def _format(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def markdown_report(payload: dict) -> str:
    lines = [
        "# Motion-planning comparison report",
        "",
        f"Generated: {payload['generated_at_utc']}  ",
        f"Git revision: `{payload['git_revision'] or 'unknown'}`  ",
        f"Schemes: {', '.join(payload['schemes'])}  ",
        f"Repeats per cell: {payload['repeats']}",
        "",
        "A cell passes only if every repeat completed all moving robots within the scenario tolerance, "
        "without any observed robot overlap or exceeding the simulator's speed limit. Timing is the normal "
        '`MotionController.calculate()` call, excluding the configured warm-up calls. "Cmd jumps / dir '
        'changes / brakes" are black-box proxies computed from the commanded velocity itself (not observed '
        "post-physics robot state): see `_ControllerTimer` for exactly what each does and does not claim to "
        "measure -- none of them is a true replan count, since that is planner-internal state this tool "
        "cannot observe through the public controller interface.",
        "",
        "| Scenario | Scheme | Result | Pass rate | Time (s) | Path/direct | Worst clearance (m) | "
        "Stopped ratio | Mean / p95 controller (ms) | Cmd jumps / dir changes / brakes |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    order = {
        (scenario, scheme): index
        for index, (scenario, scheme) in enumerate(
            (scenario, scheme) for scenario in payload["scenarios"] for scheme in payload["schemes"]
        )
    }
    for row in sorted(payload["aggregates"], key=lambda item: order[(item["scenario"], item["scheme"])]):
        result = "PASS" if row["passed"] else "FAIL"
        controller = f"{_format(row['controller_mean_ms'])} / {_format(row['controller_p95_ms_mean'])}"
        black_box_events = (
            f"{_format(row['command_jumps_mean'], 1)} / "
            f"{_format(row['command_direction_changes_mean'], 1)} / "
            f"{_format(row['brake_events_proxy_mean'], 1)}"
        )
        lines.append(
            f"| {row['scenario']} | {row['scheme']} | {result} | {row['pass_rate']:.0%} | "
            f"{_format(row['sim_time_mean_s'])} | {_format(row['path_length_ratio_mean'])} | "
            f"{_format(row['min_clearance_worst_m'], 3)} | "
            f"{_format(row['stopped_while_unreached_ratio_mean'], 3)} | {controller} | {black_box_events} |"
        )

    lines.extend(["", "## Failures", ""])
    failures = [result for result in payload["results"] if not result["passed"]]
    if not failures:
        lines.append("None.")
    else:
        for result in failures:
            detail = result.get("error", result["termination_reason"])
            speed_note = " speed limit exceeded;" if result["speed_limit_exceeded"] else ""
            lines.append(
                f"- `{result['scenario']}` / `{result['scheme']}` repeat {result['repeat']}: {detail};{speed_note} "
                f"reached {result['robots_reached']}/{result['robots_expected']}, "
                f"collisions {result['collision_events']}."
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "PASS/FAIL is a universal safety-and-completion gate, not a ranking. Compare passing planners using "
            "completion time, path ratio, clearance, stopped ratio, and controller latency together; no single "
            "metric is an overall score.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--schemes", nargs="+", choices=SCHEMES, default=list(SCHEMES))
    parser.add_argument("--scenarios", nargs="+", choices=tuple(SCENARIOS), default=list(SCENARIOS))
    parser.add_argument("--repeats", type=int, default=1, help="Runs per scheme/scenario cell (default: 1).")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results"),
        help="Directory for timestamped JSON and Markdown reports.",
    )
    parser.add_argument(
        "--timing-warmup-calls",
        type=int,
        default=5,
        help="Controller calls excluded from latency metrics in each cell (default: 5).",
    )
    parser.add_argument("--list-scenarios", action="store_true", help="Print scenario descriptions and exit.")
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Write the report but exit zero even if a benchmark cell fails.",
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if args.timing_warmup_calls < 0:
        parser.error("--timing-warmup-calls cannot be negative")
    return args


def main() -> int:
    args = parse_args()
    if args.list_scenarios:
        for scenario in SCENARIOS.values():
            print(f"{scenario.name:<20} {scenario.description}")
        return 0

    results = []
    total = len(args.schemes) * len(args.scenarios) * args.repeats
    case_number = 0
    for scenario_name in args.scenarios:
        for scheme in args.schemes:
            for repeat in range(1, args.repeats + 1):
                case_number += 1
                print(f"[{case_number}/{total}] {scenario_name} / {scheme} / repeat {repeat}", flush=True)
                result = run_case(scheme, SCENARIOS[scenario_name], repeat, args.timing_warmup_calls)
                results.append(result)
                print(
                    f"  {'PASS' if result['passed'] else 'FAIL'} ({result['termination_reason']}), "
                    f"sim={result['sim_time_s']:.2f}s, controller={_format(result['controller_mean_ms'])}ms",
                    flush=True,
                )

    generated_at = datetime.now(timezone.utc)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": generated_at.isoformat(),
        "git_revision": _git_revision(),
        "mode": "rsim",
        "headless": True,
        "schemes": list(args.schemes),
        "scenarios": list(args.scenarios),
        "repeats": args.repeats,
        "timing_warmup_calls": args.timing_warmup_calls,
        "thresholds": {
            "collision_center_distance_m": COLLISION_DISTANCE,
            "stopped_speed_mps": STOPPED_SPEED,
            "speed_limit_mps": RSIM_PARAMS.MAX_VEL,
            "speed_limit_tolerance_mps": SPEED_LIMIT_TOLERANCE,
        },
        "scenario_definitions": [_serializable_scenario(SCENARIOS[name]) for name in args.scenarios],
        "results": results,
        "aggregates": aggregate_results(results),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = generated_at.strftime("motion_planning_%Y%m%d_%H%M%S")
    json_path = args.output_dir / f"{stem}.json"
    markdown_path = args.output_dir / f"{stem}.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(markdown_report(payload), encoding="utf-8")
    print(f"\nJSON: {json_path}")
    print(f"Markdown: {markdown_path}")

    any_failures = any(not row["passed"] for row in payload["aggregates"])
    return 0 if args.allow_failures or not any_failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
