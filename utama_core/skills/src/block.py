import math

from utama_core.entities.data.command import RobotCommand
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game.game import Game
from utama_core.motion_planning.src.common.motion_controller import MotionController
from utama_core.skills.src.utils.move_utils import move


def block_attacker(
    game: Game,
    motion_controller: MotionController,
    friendly_robot_id: int,
    enemy_robot_id: int,
    attacker_has_ball: bool,
    block_ratio: float = 0.1,
    max_ball_follow_dist: float = 1.0,
) -> RobotCommand:
    """
    Intelligent defense strategy:
    1) If the attacker has the ball, block on the attacker-goal line.
    2) Otherwise, stay closer to the ball while still considering the attacker's possible shot.
    :return: The command dict for the defender robot
    """
    defender = game.friendly_robots[friendly_robot_id]
    attacker = game.enemy_robots[enemy_robot_id]
    ball = game.ball

    if attacker_has_ball:
        # ========== Prioritize blocking the shot line ==========
        ax, ay = attacker.p.x, attacker.p.y

        gx, gy = -game.field.enemy_goal_line[0][0], 0

        agx, agy = (gx - ax), (gy - ay)
        dist_ag = math.hypot(agx, agy)

        if dist_ag < 1e-6:
            # Extreme edge case if attacker and goal are basically the same
            target_x, target_y = gx, gy
        else:
            target_x = ax + block_ratio * agx
            target_y = ay + block_ratio * agy

        # Face the attacker
        face_theta = math.atan2((ay - defender.p.y), (ax - defender.p.x))
    else:
        # ========== Attacker doesn't have ball; defend ball more closely ==========
        ax, ay = attacker.p.x, attacker.p.y
        bx, by = ball.p.x, ball.p.y

        # Move ~70% of the way toward the ball from the attacker
        abx, aby = (bx - ax), (by - ay)
        target_x = ax + 0.7 * abx
        target_y = ay + 0.7 * aby

        # If we are too far from the ball, move closer
        dist_def_to_ball = math.hypot(defender.p.x - bx, defender.p.y - by)
        if dist_def_to_ball > max_ball_follow_dist:
            ratio = max_ball_follow_dist / dist_def_to_ball
            target_x = defender.p.x + (bx - defender.p.x) * ratio
            target_y = defender.p.y + (by - defender.p.y) * ratio

        # Face the ball
        face_theta = math.atan2((by - defender.p.y), (bx - defender.p.x))

    cmd = move(
        game=game,
        motion_controller=motion_controller,
        robot_id=friendly_robot_id,
        target_coords=Vector2D(target_x, target_y),
        target_oren=face_theta,
    )
    return cmd
