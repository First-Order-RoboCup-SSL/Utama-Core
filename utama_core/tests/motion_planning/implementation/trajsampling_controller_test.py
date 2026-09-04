"""Tests for `TrajectorySamplingController` -- the `MotionController` wrapper
around `TrajectorySamplingPlanner` (see `controllers/trajsampling.py`).
"""

import math
from unittest.mock import patch

from utama_core.config.enums import Mode
from utama_core.config.field_params import STANDARD_FIELD_DIMS
from utama_core.entities.data.vector import Vector2D, Vector3D
from utama_core.entities.game import Ball, Field, Game, GameFrame, GameHistory, Robot
from utama_core.motion_planning.src.controllers.trajsampling import (
    TrajectorySamplingController,
)
from utama_core.motion_planning.src.trajsampling.bang_bang import Trajectory2D
from utama_core.motion_planning.src.trajsampling.planner import PlanResult


def _game_with_robot(p_xy, v_xy) -> Game:
    zero = Vector3D(0, 0, 0)
    robot = Robot(
        id=0,
        is_friendly=True,
        has_ball=False,
        p=Vector2D(*p_xy),
        v=Vector2D(*v_xy),
        a=Vector2D(0, 0),
        orientation=0.0,
    )
    frame = GameFrame(
        ts=0.0,
        my_team_is_yellow=True,
        my_team_is_right=False,
        friendly_robots={0: robot},
        enemy_robots={},
        ball=Ball(p=zero, v=zero, a=zero),
    )
    field = Field(
        my_team_is_right=False, field_dims=STANDARD_FIELD_DIMS, field_bounds=STANDARD_FIELD_DIMS.full_field_bounds
    )
    return Game(past=GameHistory(10), current=frame, field=field)


def test_brake_scales_planned_velocity_not_robot_raw_velocity():
    """Regression for the DIRECT_FREE congestion "retreat" bug traced live on
    `clear_danger_vs_clear_press_plus` (roadmap item 15): the emergency-brake
    layer used to return `robot.v * brake_scale` -- the robot's own raw,
    already-current velocity, scaled down -- instead of the PLANNED
    trajectory's velocity, scaled down. Whenever the robot's actual momentum
    pointed anywhere other than where `plan()` says it should go (e.g. after
    being nudged off-course near a crowded obstacle), a braking tick would
    command the robot further along its OLD, wrong heading rather than
    correcting it toward the target -- confirmed live: this fed a feedback
    loop where braking pushed the robot off its planned straight-line
    trajectory by just enough (~0.08-0.09m, right at
    `_TRAJECTORY_POSITION_TOLERANCE`) to force a fresh replan almost every
    cycle, and `Trajectory2D.compute` drops transverse velocity at the start
    of every fresh plan (a known, documented limitation -- item 13), so the
    robot visibly approached and retreated from the ball in a loop for the
    rest of the match even though `plan()` reported a clean, collision-free,
    converges-to-target trajectory on every single call.

    Set up: robot moving fast in +y (a direction with ZERO relation to the
    planned trajectory, which moves in +x) and close enough to a mocked
    obstacle to force the brake. The commanded output must be a scaled-down
    version of the PLANNED (+x) velocity, not the robot's own (+y) velocity.
    """
    controller = TrajectorySamplingController(Mode.RSIM, None)

    # Robot physically moving in +y — this has nothing to do with the plan.
    game = _game_with_robot(p_xy=(0.0, 0.0), v_xy=(0.0, 1.5))

    # Planned trajectory moves straight along +x.
    planned_trajectory = Trajectory2D.compute(
        (0.0, 0.0), (1.5, 0.0), (2.0, 0.0), controller.planner.v_max, controller.planner.a_max
    )
    # Force the brake to fire: a tiny nearest-obstacle distance and a large
    # closing speed guarantee `closing_speed > max_safe_closing_speed`.
    mocked_result = PlanResult(
        trajectory=planned_trajectory,
        has_collision=False,
        collision_time=None,
        elapsed=0.0,
        nearest_obstacle_distance=0.05,
        closing_speed=5.0,
    )

    with patch.object(controller.planner, "plan", return_value=mocked_result):
        v_out, _oren = controller.calculate(game, robot_id=0, target_pos=Vector2D(2.0, 0.0), target_oren=0.0)

    # The bug: v_out would equal robot.v (0, 1.5) scaled down -- i.e. still
    # pointing in +y, the direction the robot happened to already be moving,
    # with zero x-component. The fix: v_out must point along the PLANNED
    # direction (+x), not the robot's raw (+y) velocity.
    assert v_out.x > 0, f"braked velocity should retain the planned +x direction, got {v_out}"
    assert abs(v_out.y) < 1e-6, f"braked velocity must not carry the robot's unrelated +y motion, got {v_out}"

    # And it must actually be a genuine brake (slower than the unclamped plan).
    _, (plan_vx, plan_vy) = planned_trajectory.state_at(min(controller._dt, planned_trajectory.duration))
    assert math.hypot(v_out.x, v_out.y) < math.hypot(plan_vx, plan_vy)


def test_no_brake_when_closing_speed_is_safe():
    """Sanity check: with a generous obstacle margin, the controller must
    pass the planned velocity through unscaled (no braking artifact)."""
    controller = TrajectorySamplingController(Mode.RSIM, None)
    game = _game_with_robot(p_xy=(0.0, 0.0), v_xy=(0.5, 0.0))

    planned_trajectory = Trajectory2D.compute(
        (0.0, 0.0), (0.5, 0.0), (2.0, 0.0), controller.planner.v_max, controller.planner.a_max
    )
    mocked_result = PlanResult(
        trajectory=planned_trajectory,
        has_collision=False,
        collision_time=None,
        elapsed=0.0,
        nearest_obstacle_distance=5.0,
        closing_speed=0.1,
    )

    with patch.object(controller.planner, "plan", return_value=mocked_result):
        v_out, _oren = controller.calculate(game, robot_id=0, target_pos=Vector2D(2.0, 0.0), target_oren=0.0)

    _, (plan_vx, plan_vy) = planned_trajectory.state_at(min(controller._dt, planned_trajectory.duration))
    assert v_out.x == plan_vx
    assert v_out.y == plan_vy
