from typing import Optional

from utama_core.config.settings import TIMESTEP
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game
from utama_core.motion_planning.src.common.acceleration_limiter import (
    AccelerationLimiter,
)
from utama_core.motion_planning.src.dwa.config import DynamicWindowConfig
from utama_core.rsoccer_simulator.src.ssl.envs import SSLStandardEnv

from .planner import DynamicWindowPlanner


class DWATranslationController:
    """Compute global linear velocities using a Dynamic Window Approach."""

    _TARGET_COLORS = (
        "RED",
        "ORANGE",
        "YELLOW",
        "GREEN",
        "BLUE",
        "PURPLE",
        "PINK",
        "WHITE",
    )

    def __init__(
        self,
        config: DynamicWindowConfig,
        env: SSLStandardEnv | None = None,
    ):
        self._planner_config = config
        self.env: SSLStandardEnv | None = env
        self._control_period = TIMESTEP
        self._planner = DynamicWindowPlanner(
            config=self._planner_config,
            env=self.env,
        )
        self._acceleration_limiter = AccelerationLimiter(
            max_acceleration=self._planner_config.max_acceleration,
            dt=self._control_period,
        )

    def calculate(
        self,
        game: Game,
        target: Vector2D,
        robot_id: int,
    ) -> Vector2D:
        current = game.friendly_robots[robot_id].p
        if current is None:
            return Vector2D(0.0, 0.0)

        if target is None:
            return Vector2D(0.0, 0.0)

        if self.env is not None:
            self.env.draw_point(target.x, target.y, color=self._target_color(robot_id), width=2)

        if current.distance_to(target) <= self._planner_config.target_tolerance:
            zero_velocity = Vector2D(0.0, 0.0)
            self._acceleration_limiter.reset(robot_id)
            return zero_velocity

        path = self._planner.path_to(
            game,
            robot_id,
            target,
            temporary_obstacles=[],
        )
        if path is None:
            return Vector2D(0.0, 0.0)

        best_move, best_score = path
        if best_score == float("-inf"):
            return Vector2D(0.0, 0.0)

        dt_plan = self._planner_config.simulate_frames * TIMESTEP
        dx_w = best_move.x - current.x
        dy_w = best_move.y - current.y
        velocity_global = Vector2D(dx_w / dt_plan, dy_w / dt_plan)

        velocity_limited = self._apply_speed_limits(velocity_global)
        velocity_limited = self._acceleration_limiter.limit(robot_id, velocity_limited)

        if self.env is not None:
            self.env.draw_point(best_move.x, best_move.y, color="blue", width=2)

        return velocity_limited

    def reset(self, robot_id: int):
        self._acceleration_limiter.reset(robot_id)

    def _target_color(self, robot_id: int) -> str:
        return self._TARGET_COLORS[robot_id % len(self._TARGET_COLORS)]

    def _apply_speed_limits(self, velocity: Vector2D) -> Vector2D:
        speed = velocity.mag()
        if speed <= self._planner_config.max_speed:
            return velocity
        if speed == 0:
            return Vector2D(0.0, 0.0)
        scale = self._planner_config.max_speed / speed
        return Vector2D(velocity.x * scale, velocity.y * scale)
