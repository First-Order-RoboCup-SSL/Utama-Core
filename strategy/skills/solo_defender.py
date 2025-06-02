from entities.data.command import RobotCommand
from entities.game.game import Game
from strategy.abstract_strategy import AbstractStrategy
from entities.game.present_future_game import PresentFutureGame
from robot_control.src.skills import go_to_point
import math


def improved_block_goal_and_attacker(
    robot,
    attacker,
    ball,
    game: Game,
    pid_oren,
    pid_trans,
    attacker_has_ball: bool,
    block_ratio: float = 0.1,
    max_ball_follow_dist: float = 1.0,
) -> RobotCommand:
    """
    Intelligent defense strategy:
    1) If the attacker has the ball, block on the attacker-goal line.
    2) Otherwise, stay closer to the ball while still considering the attacker's possible shot.

    :param robot: Defender's Robot object (robot.x, robot.y, robot.orientation)
    :param attacker: Attacker's Robot object
    :param ball: Ball object
    :param goal_pos: The (x, y) location of the goal to defend
    :param pid_oren: PID controller for orientation
    :param pid_trans: PID controller for translation
    :param attacker_has_ball: Whether the attacker currently has ball possession
    :param block_ratio: 0~1 ratio to position ourselves between attacker & goal
    :param max_ball_follow_dist: if attacker doesn't have ball, how close we stay near the ball
    :return: The command dict for the defender robot
    """
    if attacker_has_ball:
        # ========== Prioritize blocking the shot line ==========
        ax, ay = attacker.p.x, attacker.p.y

        gx, gy = -game.field.enemy_goal_line.coords[0][0], 0

        agx, agy = (gx - ax), (gy - ay)
        dist_ag = math.hypot(agx, agy)

        if dist_ag < 1e-6:
            # Extreme edge case if attacker and goal are basically the same
            target_x, target_y = gx, gy
        else:
            target_x = ax + block_ratio * agx
            target_y = ay + block_ratio * agy

        # Face the attacker
        face_theta = math.atan2((ay - robot.p.y), (ax - robot.p.x))
    else:
        # ========== Attacker doesn't have ball; defend ball more closely ==========
        ax, ay = attacker.p.x, attacker.p.y
        bx, by = ball.p.x, ball.p.y

        # Move ~70% of the way toward the ball from the attacker
        abx, aby = (bx - ax), (by - ay)
        target_x = ax + 0.7 * abx
        target_y = ay + 0.7 * aby

        # If we are too far from the ball, move closer
        dist_def_to_ball = math.hypot(robot.p.x - bx, robot.p.y - by)
        if dist_def_to_ball > max_ball_follow_dist:
            ratio = max_ball_follow_dist / dist_def_to_ball
            target_x = robot.p.x + (bx - robot.p.x) * ratio
            target_y = robot.p.y + (by - robot.p.y) * ratio

        # Face the ball
        face_theta = math.atan2((by - robot.p.y), (bx - robot.p.x))

    cmd = go_to_point(
        game=game,
        pid_oren=pid_oren,
        pid_trans=pid_trans,
        robot_id=0,
        target_coords=(target_x, target_y),
        target_oren=face_theta,
    )
    return cmd


class SoloDefenderStrategy(AbstractStrategy):
    def __init__(
        self,
        target_id: int,
        block_ratio: float = 0.4,
        max_ball_follow_dist: float = 1.0,
    ):
        super().__init__()
        self.target_id = target_id
        self.block_ratio = block_ratio
        self.max_ball_follow_dist = max_ball_follow_dist

    def assert_exp_robots(self, n_runtime_friendly: int, n_runtime_enemy: int):
        return n_runtime_friendly >= 1

    def step(self, present_future_game: PresentFutureGame):
        game = present_future_game.current

        # Determine which goal we're defending
        our_goal_center = (-4.5, 0) if game.my_team_is_right else (4.5, 0)

        defender = game.friendly_robots[self.target_id]
        ball = game.ball

        # Find the most threatening enemy (closest to our goal)
        if not game.enemy_robots:
            # Default to center if no enemies
            return self.default_position(
                defender, our_goal_center, game.my_team_is_yellow
            )

        # Find enemy closest to our goal
        closest_enemy = min(
            game.enemy_robots.values(),
            key=lambda e: math.hypot(e.p.x - ball.p.x, e.p.y - ball.p.y),
        )

        cmd = improved_block_goal_and_attacker(
            defender,
            closest_enemy,
            ball,
            game,
            self.pid_oren,
            self.pid_trans,
            attacker_has_ball=True,
            block_ratio=0.4,
            max_ball_follow_dist=1.0,
        )
        self.robot_controller.add_robot_commands(cmd, self.target_id)
        self.robot_controller.send_robot_commands()
        return

        # Check if enemy has ball (using distance threshold)
        enemy_ball_dist = math.hypot(
            ball.p.x - closest_enemy.p.x, ball.p.y - closest_enemy.p.y
        )
        attacker_has_ball = enemy_ball_dist < 0.1  # 10cm threshold

        # Calculate defensive position
        ax, ay = closest_enemy.p.x, closest_enemy.p.y
        gx, gy = our_goal_center

        if attacker_has_ball:
            # Block attacker-goal line
            agx, agy = gx - ax, gy - ay
            dist_ag = math.hypot(agx, agy)

            if dist_ag < 1e-6:  # Edge case handling
                target_x, target_y = gx, gy
            else:
                target_x = ax + self.block_ratio * agx
                target_y = ay + self.block_ratio * agy

            # Face the attacker
            face_theta = math.atan2(ay - defender.p.y, ax - defender.p.x)
        else:
            # Position between ball and enemy
            bx, by = ball.p.x, ball.p.y
            abx, aby = bx - ax, by - ay
            target_x = ax + 0.7 * abx
            target_y = ay + 0.7 * aby

            # Don't stray too far from ball
            defender_ball_dist = math.hypot(defender.p.x - bx, defender.p.y - by)
            if defender_ball_dist > self.max_ball_follow_dist:
                ratio = self.max_ball_follow_dist / defender_ball_dist
                target_x = defender.p.x + (bx - defender.p.x) * ratio
                target_y = defender.p.y + (by - defender.p.y) * ratio

            # Face the ball
            face_theta = math.atan2(by - defender.p.y, bx - defender.p.x)

        # Generate movement command
        defender_pose = (defender.p.x, defender.p.y, defender.orientation)
        cmd = go_to_point(
            game,
            self.pid_oren,
            self.pid_trans,
            0,  # robot_id unused in go_to_point
            (target_x, target_y),
            face_theta,
        )

    def default_position(self, defender, goal_center, is_yellow):
        """Fallback position when no enemies are present"""
        # Position 30cm in front of goal
        offset = 0.3 if is_yellow else -0.3
        target_x = goal_center[0] + offset
        target_y = goal_center[1]

        # Face center of field
        face_theta = math.atan2(0, 1) if is_yellow else math.atan2(0, -1)

        defender_pose = (defender.p.x, defender.p.y, defender.p.theta)
        return go_to_point(
            self.pid_oren,
            self.pid_trans,
            defender_pose,
            0,
            (target_x, target_y),
            face_theta,
        )
