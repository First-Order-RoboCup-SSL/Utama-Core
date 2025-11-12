"""Random Charging Strategy
============================

Purpose:
        Stress-test collision avoidance by commanding all six friendly robots
        to move at high speed with randomly changing local velocities.

Design:
        Implements `AbstractStrategy` with a single behaviour node that assigns
        far random targets and uses the current motion controller to charge
        towards them. For testing, it optionally overrides the DWA speed clamp
        and acceleration limits so robots can reach high speeds.

Notes:
                - Integrates with motion controller via move_utils.move, reducing yaw
                        twitching by requesting current orientation as target orientation.
                - Can override DWA max speed and acceleration at runtime (testing only).
                - Keeps a per-robot persistent random target; re-rolls when reached.
"""

import math
import random
from typing import Optional

import py_trees

from utama_core.config.enums import Role
from utama_core.entities.data.command import RobotCommand
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game
from utama_core.entities.game.field import Field, FieldBounds
from utama_core.skills.src.utils.move_utils import empty_command, move
from utama_core.strategy.common import AbstractBehaviour, AbstractStrategy

HIGH_TEST_SPEED = 6.0  # m/s desired high speed through DWA (override clamp)
ACCEL_TEST_LIMIT = 10.0  # m/s^2 higher accel for quicker speed-up
TARGET_REACH_TOL = 0.15  # m
MIN_TARGET_DISTANCE = 2.0  # m - ensure targets are far


class RandomChargeBehaviour(AbstractBehaviour):
    """Behaviour node that picks far random targets and charges towards them.

    Blackboard Requirements:
            - game (Game)
            - cmd_map (dict[int, RobotCommand | None])
            - motion_controller (MotionController)
            - random_targets (dict[int, Vector2D]) [created here]
            - speed_override_applied (bool) [created here]
    """

    def __init__(self, name: str = "RandomCharge"):
        super().__init__(name=name)

    def setup_(self):
        # custom keys
        self.blackboard.register_key(key="random_targets", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="speed_override_applied", access=py_trees.common.Access.WRITE)
        # init
        self.blackboard.random_targets = {}
        self.blackboard.speed_override_applied = False

    def update(self) -> py_trees.common.Status:
        game: Game = self.blackboard.game
        cmd_map = self.blackboard.cmd_map

        # One-time: try to override DWA clamps for testing
        if not self.blackboard.speed_override_applied:
            self._try_override_dwa_limits()
            self.blackboard.speed_override_applied = True

        # For each robot, pick/maintain a far target and move towards it via motion controller
        for robot_id, robot in game.friendly_robots.items():
            target = self._get_or_create_target(game, robot_id)

            # Reduce rotation twitching: keep current orientation as target orientation
            target_oren = robot.orientation

            cmd = move(
                game=game,
                motion_controller=self.blackboard.motion_controller,
                robot_id=robot_id,
                target_coords=target,
                target_oren=target_oren,
                dribbling=False,
            )
            cmd_map[robot_id] = cmd

            # If close enough to target, re-roll next tick
            if robot.p.distance_to(target) <= TARGET_REACH_TOL:
                self.blackboard.random_targets.pop(robot_id, None)

        return py_trees.common.Status.SUCCESS

    # --- Helpers ---
    def _get_or_create_target(self, game: Game, robot_id: int) -> Vector2D:
        tgt = self.blackboard.random_targets.get(robot_id)
        if tgt is None or game.friendly_robots[robot_id].p.distance_to(tgt) <= TARGET_REACH_TOL:
            tgt = self._random_far_point(game, robot_id)
            self.blackboard.random_targets[robot_id] = tgt
        return tgt

    def _random_far_point(self, game: Game, robot_id: int) -> Vector2D:
        # Choose a random point near field extremes, assuring it's far from current position
        field: Field = game.field
        x_min, y_max = field.field_bounds.top_left
        x_max, y_min = field.field_bounds.bottom_right

        rx = random.uniform(x_min + 0.2, x_max - 0.2)
        ry = random.uniform(y_min + 0.2, y_max - 0.2)

        current = game.friendly_robots[robot_id].p
        # If too close, bias towards far side by mirroring across center
        if current.distance_to(Vector2D(rx, ry)) < MIN_TARGET_DISTANCE:
            rx = x_max - (rx - x_min)
            ry = y_max - (ry - y_min)

        return Vector2D(rx, ry)

    def _try_override_dwa_limits(self) -> None:
        """Best-effort increase of DWA limits at runtime for testing.

        This pokes into the DWAController internals if present, to:
          - raise max_speed clamp
          - raise acceleration limiter
        Safe to no-op if a different controller is in use.
        """
        mc = getattr(self.blackboard, "motion_controller", None)
        if mc is None:
            return
        # Only if controller has DWA internals
        dwa_trans = getattr(mc, "_dwa_trans", None)
        if dwa_trans is None:
            return
        # Raise planner config limits
        planner_cfg = getattr(dwa_trans, "_planner_config", None)
        if planner_cfg is not None:
            try:
                planner_cfg.max_speed = max(planner_cfg.max_speed, HIGH_TEST_SPEED)
                planner_cfg.max_acceleration = max(planner_cfg.max_acceleration, ACCEL_TEST_LIMIT)
            except Exception:
                pass
        # Raise acceleration limiter
        accel_lim = getattr(dwa_trans, "_acceleration_limiter", None)
        if accel_lim is not None:
            try:
                # private attr, ok for testing
                accel_lim._max_acceleration = max(getattr(accel_lim, "_max_acceleration", 0.0), ACCEL_TEST_LIMIT)
            except Exception:
                pass


class RandomChargingStrategy(AbstractStrategy):
    """Strategy that drives six robots with random high-speed motions.

    Behaviour Tree:
            Root -> RandomChargeBehaviour
    """

    def create_behaviour_tree(self) -> py_trees.behaviour.Behaviour:
        return RandomChargeBehaviour()

    # --- Assertions & Requirements ---
    def assert_exp_robots(self, n_runtime_friendly: int, n_runtime_enemy: int):  # type: ignore[override]
        # Expect exactly six friendly robots; enemy count flexible (>=1)
        assert n_runtime_friendly == 6, "RandomChargingStrategy requires exactly 6 friendly robots"
        assert n_runtime_enemy >= 1, "Needs at least one opponent robot present"
        return True

    def assert_exp_goals(self, includes_my_goal_line: bool, includes_opp_goal_line: bool):  # type: ignore[override]
        # Require standard field with both goals available
        assert includes_my_goal_line and includes_opp_goal_line, "Full goals required for stress test"
        return True

    def get_min_bounding_zone(self) -> Optional[FieldBounds]:
        # Use entire field
        return Field.full_field_bounds

    # --- Optional overrides ---
    def execute_default_action(self, game: Game, role: Role, robot_id: int) -> RobotCommand:
        # If no command set (should not happen), stop robot safely
        return empty_command(False)


__all__ = ["RandomChargingStrategy"]
