from typing import Optional

from utama_core.config.physical_constants import ROBOT_RADIUS
from utama_core.data_processing.predictors.position import predict_ball_pos_at_x
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game
from utama_core.motion_planning.src.common.motion_controller import MotionController
from utama_core.rsoccer_simulator.src.ssl.envs.standard_ssl import SSLStandardEnv
from utama_core.skills.src.go_to_point import go_to_point
from utama_core.skills.src.utils.defense_utils import _defense_geometry

STATIC_DEFENDER_OFFSET_X = 0.5
STATIC_DEFENDER_MARGIN_Y = 0.2
GOAL_FRAME_OFFSET = 0.2


def defend_parameter(
    game: Game,
    motion_controller: MotionController,
    robot_id: int,
    env: Optional[SSLStandardEnv] = None,
    goal_frame_y: Optional[float] = None,
):
    robot_pos = game.friendly_robots[robot_id].p
    # Ball position is stored in 3D, but all defending geometry here is planar.
    ball_pos = game.ball.p.to_2d()
    ball_vel = game.ball.v.to_2d()

    geo = _defense_geometry(game)
    goal_x = geo.goal_x
    goal_half_width = geo.goal_half_width
    defense_front_x = geo.defense_front_x
    defense_depth = geo.defense_depth
    defense_half_width = geo.defense_half_width
    sign = 1.0 if game.my_team_is_right else -1.0

    # Multi-robot: send specific defenders to static positions
    if len(game.friendly_robots) > 2:
        if ball_pos.y >= -goal_half_width and robot_id == 1:
            return go_to_point(
                game,
                motion_controller,
                robot_id,
                Vector2D(
                    defense_front_x - sign * STATIC_DEFENDER_OFFSET_X,
                    -(defense_half_width + STATIC_DEFENDER_MARGIN_Y),
                ),
            )
        elif ball_pos.y < -goal_half_width and robot_id == 2:
            return go_to_point(
                game,
                motion_controller,
                robot_id,
                Vector2D(
                    defense_front_x - sign * STATIC_DEFENDER_OFFSET_X,
                    defense_half_width + STATIC_DEFENDER_MARGIN_Y,
                ),
            )

    if goal_frame_y is None:
        goal_frame_y = goal_half_width if robot_id == 1 else -goal_half_width

    def project_and_clamp(goal_point: Vector2D) -> Vector2D:
        """Project onto the blocking line, anchored near the front of our box."""
        line = goal_point - ball_pos
        denom = line.dot(line)
        if denom < 1e-12:
            return robot_pos

        t = (robot_pos - ball_pos).dot(line) / denom

        if abs(line.x) > 1e-12:
            t_box = (defense_front_x - ball_pos.x) / line.x
            t = max(t, max(0.0, min(1.0, t_box)))

        t = max(0.0, min(1.0, t))
        projected = ball_pos + line * t

        # Clamp when projected point is near goal line and within defense area
        if abs(goal_point.x - projected.x) < defense_depth and -defense_half_width < projected.y < defense_half_width:
            ball_to_goal = ball_pos - goal_point

            # Horizontal clamp: at defense front
            if abs(ball_to_goal.x) > 1e-12:
                clamp_x = defense_front_x
                t_h = (clamp_x - goal_point.x) / ball_to_goal.x
                hori = Vector2D(clamp_x, goal_point.y + t_h * ball_to_goal.y)
            else:
                hori = projected

            # Vertical clamp: defense half-width span from goal point y
            if abs(ball_to_goal.y) > 1e-12:
                clamp_y = goal_point.y - 2 * defense_half_width
                t_v = (clamp_y - goal_point.y) / ball_to_goal.y
                ver = Vector2D(goal_point.x + t_v * ball_to_goal.x, clamp_y)
            else:
                ver = projected

            if projected.distance_to(hori) < projected.distance_to(ver):
                return hori
            else:
                return ver

        return projected

    # Check ball trajectory
    ball_at_baseline = predict_ball_pos_at_x(game, goal_x)
    ball_at_robot = predict_ball_pos_at_x(game, robot_pos.x)

    if (
        ball_vel.dot(ball_vel) > 0.05
        and ball_at_baseline is not None
        and abs(ball_at_baseline.y) < goal_half_width
        and ball_at_robot is not None
        and abs(ball_at_robot.y - robot_pos.y) > 0.1
    ):
        offset = GOAL_FRAME_OFFSET if goal_frame_y >= 0 else -GOAL_FRAME_OFFSET
        goal_point = Vector2D(goal_x, goal_frame_y + offset)
        target = project_and_clamp(goal_point)
    else:
        goal_point = Vector2D(goal_x, -goal_frame_y)
        target = project_and_clamp(goal_point)
        # Offset target by robot radius
        vec_to_target = target - robot_pos
        dist = vec_to_target.mag()
        if dist > 0:
            target = target - vec_to_target.norm() * ROBOT_RADIUS

    return go_to_point(game, motion_controller, robot_id, target, dribbling=True)
