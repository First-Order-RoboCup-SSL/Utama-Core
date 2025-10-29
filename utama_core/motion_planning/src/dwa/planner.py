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
from utama_core.motion_planning.src.dwa.config import (
    MAX_ACCELERATION,
    MAX_SAFETY_RADIUS,
    MAX_SPEED_FOR_FULL_BUBBLE,
    SAFETY_PENALTY_DISTANCE_SQ,
    SIMULATED_TIMESTEP,
)
from utama_core.motion_planning.src.pid.pid_abstract import AbstractPID
from utama_core.motion_planning.src.planning.path_planner import DynamicWindowPlanner
from utama_core.rsoccer_simulator.src.ssl.envs import SSLStandardEnv


@dataclass(slots=True)
class DWATranslationParams:
    """Tuning parameters for the translation DWA controller."""

    max_speed: float
    max_acceleration: float
    horizon: float = SIMULATED_TIMESTEP
    target_tolerance: float = 0.1


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
    """Compute global linear velocities using a Dynamic Window Approach."""

    def __init__(self, params: DWATranslationParams, num_robots: int = MAX_ROBOTS):
        self._params = params
        self._control_period = TIMESTEP
        self._planner: Optional[DynamicWindowPlanner] = None
        self._previous_velocity: Dict[int, Tuple[float, float]] = {idx: (0.0, 0.0) for idx in range(num_robots)}
        self.env: SSLStandardEnv = None  # type: ignore

    # ------------------------------------------------------------------
    # Public interface shared with AbstractPID implementers
    # ------------------------------------------------------------------
    def calculate(
        self,
        game: Game,
        target: Tuple[float, float],
        robot_id: int,
    ) -> Tuple[float, float]:
        self._ensure_planner(game)
        assert self._planner is not None  # for type-checkers

        current = game.friendly_robots[robot_id].p

        if current is None:
            return 0.0, 0.0

        start_x, start_y = current[0], current[1]
        if target is None or None in target:
            return 0.0, 0.0

        # If we are sufficiently close to the target, stop.
        if math.dist((start_x, start_y), target) <= self._params.target_tolerance:
            self._previous_velocity[robot_id] = (0.0, 0.0)
            return 0.0, 0.0

        best_move, _ = self._planner.path_to(
            robot_id,
            target,
            temporary_obstacles=[],
        )

        if best_move is None:
            return 0.0, 0.0

        dt_plan = self._params.horizon
        dx_w = best_move[0] - start_x
        dy_w = best_move[1] - start_y
        vx_w = dx_w / dt_plan
        vy_w = dy_w / dt_plan

        dist_to_goal = math.dist((start_x, start_y), target)
        vmax_goal = min(self._params.max_speed, 2 * dist_to_goal)
        speed = math.hypot(vx_w, vy_w)
        if vmax_goal > 1e-6 and speed > vmax_goal:
            s = vmax_goal / speed
            vx_w *= s
            vy_w *= s

        # --- Respect speed/accel limits in the frame the actuators expect (usually body) ---
        vx_b, vy_b = self._apply_speed_limits(vx_w, vy_w)
        vx_b, vy_b = self._apply_acceleration_limits(robot_id, vx_b, vy_b)
        self._previous_velocity[robot_id] = (vx_b, vy_b)

        if self.env is not None:
            self.env.draw_point(best_move[0], best_move[1], color="blue", width=2)

        return vx_b, vy_b

    def reset(self, robot_id: int):
        self._previous_velocity[robot_id] = (0.0, 0.0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_planner(self, game: Game):
        if not isinstance(game, Game):
            raise TypeError(f"DWA planner requires a Game instance. {type(game)} given.")

        if self._planner is None:
            self._planner = DynamicWindowPlanner(
                game=game,
                simulated_timestep=SIMULATED_TIMESTEP,
                max_acceleration=MAX_ACCELERATION,
                max_safety_radius=MAX_SAFETY_RADIUS,
                safety_penalty_distance_sq=SAFETY_PENALTY_DISTANCE_SQ,
                max_speed_for_full_bubble=MAX_SPEED_FOR_FULL_BUBBLE,
            )

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

    def set_debug_env(self, env):
        self.env = env


### Using PID for Yaw control for now, as DWA is too unstable and not very useful ###


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
