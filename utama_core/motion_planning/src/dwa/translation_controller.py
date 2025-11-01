from typing import Dict, Optional

from utama_core.config.physical_constants import MAX_ROBOTS
from utama_core.config.settings import TIMESTEP
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game
from utama_core.motion_planning.src.dwa.config import DynamicWindowConfig
from utama_core.rsoccer_simulator.src.ssl.envs import SSLStandardEnv

from .planner import DynamicWindowPlanner


class DWATranslationController:
    """Compute global linear velocities using a Dynamic Window Approach."""

    def __init__(
        self,
        config: DynamicWindowConfig,
        num_robots: int = MAX_ROBOTS,
        env: SSLStandardEnv | None = None,
    ):
        self._planner_config = config
        self.env: SSLStandardEnv | None = env
        self._control_period = TIMESTEP
        self._planner: Optional[DynamicWindowPlanner] = None
        self._previous_velocity: Dict[int, Vector2D] = {idx: Vector2D(0.0, 0.0) for idx in range(num_robots)}

    def _ensure_planner(self, game: Game) -> DynamicWindowPlanner:
        if not isinstance(game, Game):
            raise TypeError(f"DWA planner requires a Game instance. {type(game)} given.")

        if self._planner is None:
            self._planner = self._create_planner(game)
        return self._planner

    def calculate(
        self,
        game: Game,
        target: Vector2D,
        robot_id: int,
    ) -> Vector2D:
        planner = self._ensure_planner(game)

        current = game.friendly_robots[robot_id].p
        if current is None:
            return Vector2D(0.0, 0.0)

        if target is None:
            return Vector2D(0.0, 0.0)

        if current.distance_to(target) <= self._planner_config.target_tolerance:
            zero_velocity = Vector2D(0.0, 0.0)
            self._previous_velocity[robot_id] = zero_velocity
            return zero_velocity

        path = planner.path_to(
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
        velocity_limited = self._apply_acceleration_limits(robot_id, velocity_limited)
        self._previous_velocity[robot_id] = velocity_limited

        if self.env is not None:
            self.env.draw_point(best_move.x, best_move.y, color="blue", width=2)

        return velocity_limited

    def reset(self, robot_id: int):
        self._previous_velocity[robot_id] = Vector2D(0.0, 0.0)

    def _apply_speed_limits(self, velocity: Vector2D) -> Vector2D:
        speed = velocity.mag()
        if speed <= self._planner_config.max_speed:
            return velocity
        if speed == 0:
            return Vector2D(0.0, 0.0)
        scale = self._planner_config.max_speed / speed
        return Vector2D(velocity.x * scale, velocity.y * scale)

    def _apply_acceleration_limits(self, robot_id: int, velocity: Vector2D) -> Vector2D:
        prev_velocity = self._previous_velocity.get(robot_id, Vector2D(0.0, 0.0))
        delta_velocity = velocity - prev_velocity
        delta_speed = delta_velocity.mag()
        allowed_delta = self._planner_config.max_acceleration * self._control_period
        if delta_speed <= allowed_delta:
            return velocity
        if delta_speed == 0:
            return prev_velocity
        scale = allowed_delta / delta_speed
        return prev_velocity + delta_velocity * scale

    def _create_planner(self, game: Game) -> DynamicWindowPlanner:
        return DynamicWindowPlanner(
            game=game,
            config=self._planner_config,
            env=self.env,
        )
