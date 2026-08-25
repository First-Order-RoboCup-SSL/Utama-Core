"""Kernel-based test-only strategies for motion-planning tests.

Replaces `strategy/examples/motion_planning/{simple_navigation_strategy,
multi_robot_navigation_strategy,oscillating_obstacle_strategy}.py` (deleted —
BT-only, no longer exist). Both tactics here just call `move()`/`go_to_point`-
style motion every tick with a per-robot fixed or time-varying target; there's
no scheduling decision to make (every robot in the roster always gets a
command), so a single-tactic kernel strategy via `single_tactic_picker` is
enough — no need for `applicable()`/multi-tactic partitioning machinery.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from utama_core.engine.abstract_strategy import AbstractStrategy
from utama_core.engine.context import TickContext
from utama_core.engine.strategy import Strategy as KernelSchedulerStrategy
from utama_core.engine.tactic import BaseTactic, RobotId, TacticTag
from utama_core.entities.data.command import RobotCommand
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game
from utama_core.motion_planning.src.common.motion_controller import MotionController
from utama_core.skills.src.utils.move_utils import move


@dataclass
class _GoToPointMem:
    pass


class _GoToPointTactic(BaseTactic[_GoToPointMem]):
    """Drives each robot in `robot_targets` to its own fixed point every tick."""

    tag = TacticTag.MIXED

    def __init__(self, robot_targets: dict[int, tuple[float, float]], target_orientation: float = 0.0):
        self.robot_targets = robot_targets
        self.target_orientation = target_orientation

    def initial_mem(self) -> _GoToPointMem:
        return _GoToPointMem()

    def tick(
        self, game: Game, ctx: TickContext, robot_ids: tuple[RobotId, ...], mem: _GoToPointMem
    ) -> tuple[dict[RobotId, RobotCommand], _GoToPointMem]:
        commands: dict[RobotId, RobotCommand] = {}
        for robot_id in robot_ids:
            target = Vector2D(*self.robot_targets[robot_id])
            commands[robot_id] = move(game, ctx.motion_controller, robot_id, target, self.target_orientation)
        return commands, mem


def go_to_point_strategy(
    robot_targets: dict[int, tuple[float, float]], target_orientation: float = 0.0
) -> AbstractStrategy:
    """Single-robot or multi-robot fixed-point navigation, kernel-native.

    `goalkeeper_id` is pointed outside the roster (robot 0 usually IS one of
    `robot_targets`' keys here, and needs to actually run the tactic, not be
    pinned as goalkeeper) — safe because these tests have no ball/goalkeeper
    concept at all (`exp_ball=False` throughout), so an always-empty
    goalkeeper tick is harmless (see `AbstractStrategy.step()`: it only skips
    the goalkeeper tick when its id is already in `cmd_map`, which every
    robot here always is).
    """

    def _build(motion_controller: MotionController) -> KernelSchedulerStrategy:
        ctx = TickContext(motion_controller=motion_controller)
        tactic = _GoToPointTactic(robot_targets, target_orientation)
        return KernelSchedulerStrategy(
            tactics={"go_to_point": tactic},
            partitioner=KernelSchedulerStrategy.single_tactic_picker(lambda game, active: "go_to_point"),
            outfield_robot_ids=tuple(robot_targets.keys()),
            ctx=ctx,
        )

    return AbstractStrategy(build_kernel_strategy=_build, exp_ball=False)


def single_robot_go_to_point_strategy(
    robot_id: int, target_position: tuple[float, float], target_orientation: float = 0.0
) -> AbstractStrategy:
    """Convenience wrapper for the single-robot case."""
    return go_to_point_strategy({robot_id: target_position}, target_orientation)


@dataclass
class _OscillatingObstacleConfig:
    obstacle_id: int
    center_position: tuple[float, float]
    oscillation_axis: str
    amplitude: float
    direction_up_or_right: bool
    speed: float


@dataclass
class _OscillateMem:
    start_ts: float | None = None


class _OscillatingObstacleTactic(BaseTactic[_OscillateMem]):
    """Time-varying sinusoidal target per robot — same motion pattern as the
    deleted BT `OscillatingObstacleBehaviour`, ported directly."""

    tag = TacticTag.MIXED

    def __init__(self, configs: list[_OscillatingObstacleConfig]):
        self.configs_by_id = {c.obstacle_id: c for c in configs}

    def initial_mem(self) -> _OscillateMem:
        return _OscillateMem()

    def tick(
        self, game: Game, ctx: TickContext, robot_ids: tuple[RobotId, ...], mem: _OscillateMem
    ) -> tuple[dict[RobotId, RobotCommand], _OscillateMem]:
        if mem.start_ts is None:
            mem.start_ts = game.ts
        elapsed = game.ts - mem.start_ts

        commands: dict[RobotId, RobotCommand] = {}
        for robot_id in robot_ids:
            config = self.configs_by_id[robot_id]
            wave = (
                math.sin(config.speed * elapsed) if config.direction_up_or_right else math.cos(config.speed * elapsed)
            )
            offset = config.amplitude * wave
            cx, cy = config.center_position
            if config.oscillation_axis.lower() == "x":
                target = Vector2D(cx + offset, cy)
            else:
                target = Vector2D(cx, cy + offset)
            commands[robot_id] = move(game, ctx.motion_controller, robot_id, target, 0.0)
        return commands, mem


def oscillating_obstacle_strategy(obstacle_configs: list) -> AbstractStrategy:
    """`obstacle_configs`: list of objects with `center_position`,
    `oscillation_axis`, `amplitude`, `direction_up_or_right`, `speed`
    (matching `MovingObstacleConfig` in `single_robot_moving_obstacle_test.py`)
    — obstacle_id is assigned by list position (0, 1, 2, ...), matching the
    deleted BT strategy's convention."""
    configs = [
        _OscillatingObstacleConfig(
            obstacle_id=i,
            center_position=c.center_position,
            oscillation_axis=c.oscillation_axis,
            amplitude=c.amplitude,
            direction_up_or_right=c.direction_up_or_right,
            speed=c.speed,
        )
        for i, c in enumerate(obstacle_configs)
    ]

    def _build(motion_controller: MotionController) -> KernelSchedulerStrategy:
        ctx = TickContext(motion_controller=motion_controller)
        tactic = _OscillatingObstacleTactic(configs)
        return KernelSchedulerStrategy(
            tactics={"oscillate": tactic},
            partitioner=KernelSchedulerStrategy.single_tactic_picker(lambda game, active: "oscillate"),
            outfield_robot_ids=tuple(c.obstacle_id for c in configs),
            ctx=ctx,
        )

    return AbstractStrategy(build_kernel_strategy=_build, exp_ball=False)


__all__ = [
    "go_to_point_strategy",
    "single_robot_go_to_point_strategy",
    "oscillating_obstacle_strategy",
]
