"""Dynamic Window Approach based motion controllers.

This module provides drop-in replacements for the legacy PID translation and
rotation controllers. The new controllers share the same public interface so
that existing skills can remain unchanged while leveraging the DWA planner for
local motion decisions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

from utama_core.config.settings import (
    MAX_ANGULAR_VEL,
    MAX_ROBOTS,
    MAX_VEL,
    REAL_MAX_ANGULAR_VEL,
    REAL_MAX_VEL,
    TIMESTEP,
)
from utama_core.entities.game import Game
from utama_core.global_utils.math_utils import normalise_heading
from utama_core.motion_planning.src.pid.pid import (
    get_grsim_pids,
    get_real_pids,
    get_rsim_pids,
)
from utama_core.motion_planning.src.pid.pid_abstract import AbstractPID
from utama_core.motion_planning.src.planning.path_planner import DynamicWindowPlanner


@dataclass(slots=True)
class DWATranslationParams:
    """Tuning parameters for the translation DWA controller."""

    max_speed: float
    max_acceleration: float
    horizon: float = DynamicWindowPlanner.SIMULATED_TIMESTEP
    target_tolerance: float = 0.02


@dataclass(slots=True)
class DWAOrientationParams:
    """Tuning parameters for the rotational controller built on DWA ideas."""

    max_speed: float
    max_acceleration: float
    horizon: float = TIMESTEP
    samples: int = 11
    target_tolerance: float = 0.01
    heading_weight: float = 1.0
    velocity_weight: float = 0.1


class DWATranslationController(AbstractPID[Tuple[float, float]]):
    """Compute global linear velocities using a Dynamic Window Approach.

    The controller keeps the AbstractPID API so that existing call-sites can
    continue to call ``calculate`` and ``reset`` without modifications. Under
    the hood it relies on :class:`DynamicWindowPlanner` to generate a short
    horizon trajectory and converts that trajectory into a velocity command.
    """

    def __init__(self, params: DWATranslationParams, num_robots: int = MAX_ROBOTS):
        self._params = params
        self._game: Optional[Game] = None
        self._planner: Optional[DynamicWindowPlanner] = None
        self._previous_velocity: Dict[int, Tuple[float, float]] = {idx: (0.0, 0.0) for idx in range(num_robots)}

    # ------------------------------------------------------------------
    # Public interface shared with AbstractPID implementers
    # ------------------------------------------------------------------
    def calculate(
        self,
        target: Tuple[float, float],
        current: Tuple[float, float],
        robot_id: int,
    ) -> Tuple[float, float]:
        if self._game is None:
            # No context has been provided yet – treat as stationary.
            return 0.0, 0.0

        self._ensure_planner()
        assert self._planner is not None  # for type-checkers

        robot = self._game.friendly_robots.get(robot_id)
        if robot is None:
            return 0.0, 0.0

        start_x, start_y = robot.p.x, robot.p.y
        if target is None or None in target:
            return 0.0, 0.0

        # If we are sufficiently close to the goal, stop.
        if math.dist((start_x, start_y), target) <= self._params.target_tolerance:
            self._previous_velocity[robot_id] = (0.0, 0.0)
            return 0.0, 0.0

        best_move, _score = self._planner.path_to(
            robot_id,
            target,
            temporary_obstacles=[],
        )

        if best_move is None:
            return 0.0, 0.0

        vx = (best_move[0] - start_x) / self._params.horizon
        vy = (best_move[1] - start_y) / self._params.horizon
        vx, vy = self._apply_speed_limits(vx, vy)
        vx, vy = self._apply_acceleration_limits(robot_id, vx, vy)
        self._previous_velocity[robot_id] = (vx, vy)
        return vx, vy

    def reset(self, robot_id: int):
        self._previous_velocity[robot_id] = (0.0, 0.0)

    # ------------------------------------------------------------------
    # Context management
    # ------------------------------------------------------------------
    def update_game(self, game: Game):
        self._game = game
        if self._planner is not None:
            self._planner._game = game

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_planner(self):
        if self._planner is None and self._game is not None:
            self._planner = DynamicWindowPlanner(self._game)

    def _apply_speed_limits(self, vx: float, vy: float) -> Tuple[float, float]:
        speed = math.hypot(vx, vy)
        if speed <= self._params.max_speed:
            return vx, vy
        if speed == 0:
            return 0.0, 0.0
        scale = self._params.max_speed / speed
        return vx * scale, vy * scale

    def _apply_acceleration_limits(self, robot_id: int, vx: float, vy: float) -> Tuple[float, float]:
        prev_vx, prev_vy = self._previous_velocity.get(robot_id, (0.0, 0.0))
        delta_vx = vx - prev_vx
        delta_vy = vy - prev_vy
        delta_speed = math.hypot(delta_vx, delta_vy)
        allowed_delta = self._params.max_acceleration * self._params.horizon
        if delta_speed <= allowed_delta:
            return vx, vy
        if delta_speed == 0:
            return prev_vx, prev_vy
        scale = allowed_delta / delta_speed
        return prev_vx + delta_vx * scale, prev_vy + delta_vy * scale


class DWAOrientationController(AbstractPID[float]):
    """Angular velocity controller using a 1D dynamic window search."""

    def __init__(self, params: DWAOrientationParams, num_robots: int = MAX_ROBOTS):
        self._params = params
        self._previous_command: Dict[int, float] = {idx: 0.0 for idx in range(num_robots)}

    def calculate(self, target: float, current: float, robot_id: int) -> float:
        error = normalise_heading(target - current)
        if abs(error) <= self._params.target_tolerance:
            self._previous_command[robot_id] = 0.0
            return 0.0

        min_cmd, max_cmd = self._candidate_window(robot_id)
        best_cmd = 0.0
        best_score = float("-inf")
        for omega in self._linspace(min_cmd, max_cmd, self._params.samples):
            score = self._score_command(error, omega)
            if score > best_score:
                best_score = score
                best_cmd = omega

        self._previous_command[robot_id] = best_cmd
        return best_cmd

    def reset(self, robot_id: int):
        self._previous_command[robot_id] = 0.0

    # ------------------------------------------------------------------
    # Internal helper methods
    # ------------------------------------------------------------------
    def _candidate_window(self, robot_id: int) -> Tuple[float, float]:
        prev = self._previous_command.get(robot_id, 0.0)
        delta = self._params.max_acceleration * self._params.horizon
        low = max(-self._params.max_speed, prev - delta)
        high = min(self._params.max_speed, prev + delta)
        if low > high:
            low, high = high, low
        return low, high

    def _score_command(self, error: float, omega: float) -> float:
        predicted_error = normalise_heading(error - omega * self._params.horizon)
        heading_term = -abs(predicted_error)
        velocity_term = -abs(omega) / self._params.max_speed if self._params.max_speed else 0.0
        return self._params.heading_weight * heading_term + self._params.velocity_weight * velocity_term

    @staticmethod
    def _linspace(start: float, stop: float, samples: int) -> Iterable[float]:
        if samples <= 1:
            yield stop
            return
        step = (stop - start) / (samples - 1)
        for idx in range(samples):
            yield start + step * idx


# ----------------------------------------------------------------------
# Factory helpers matching the PID API
# ----------------------------------------------------------------------


def get_rsim_dwa_controllers() -> Tuple[AbstractPID, DWATranslationController]:
    pid_orientation, _ = get_rsim_pids()
    translation = DWATranslationController(
        DWATranslationParams(
            max_speed=MAX_VEL,
            max_acceleration=2.0,
            horizon=DynamicWindowPlanner.SIMULATED_TIMESTEP,
            target_tolerance=0.02,
        )
    )
    return pid_orientation, translation


def get_grsim_dwa_controllers() -> Tuple[AbstractPID, DWATranslationController]:
    pid_orientation, _ = get_grsim_pids()
    translation = DWATranslationController(
        DWATranslationParams(
            max_speed=MAX_VEL,
            max_acceleration=2.0,
            horizon=DynamicWindowPlanner.SIMULATED_TIMESTEP,
            target_tolerance=0.02,
        )
    )
    return pid_orientation, translation


def get_real_dwa_controllers() -> Tuple[AbstractPID, DWATranslationController]:
    pid_orientation, _ = get_real_pids()
    translation = DWATranslationController(
        DWATranslationParams(
            max_speed=REAL_MAX_VEL,
            max_acceleration=0.3,
            horizon=DynamicWindowPlanner.SIMULATED_TIMESTEP,
            target_tolerance=0.015,
        )
    )
    return pid_orientation, translation
