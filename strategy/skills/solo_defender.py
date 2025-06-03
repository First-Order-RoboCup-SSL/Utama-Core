from config.settings import ROBOT_RADIUS
from entities.data.command import RobotCommand
from entities.game.game import Game
from strategy.abstract_strategy import AbstractStrategy
from entities.game.present_future_game import PresentFutureGame
from robot_control.src.skills import go_to_point
import math


def improved_block_goal_and_attacker(
    robot,  # Robot object for the defender
    attacker,  # Robot object for the attacker
    ball,  # Ball object
    game: Game,
    pid_oren,  # PID controller for orientation
    pid_trans,  # PID controller for translation
    attacker_has_ball: bool,
    block_ratio: float = 0.1,
    max_ball_follow_dist: float = 1.0,
) -> RobotCommand:
    """
    Generates a command for a defender robot to intelligently block the goal and an attacker.

    The strategy is:
    1) If the attacker has the ball, block on the attacker-goal line.
    2) Otherwise, stay closer to the ball while still considering the attacker's possible shot.

    :param robot: The defender's Robot object (robot.p.x, robot.p.y, robot.orientation).
    :param attacker: The attacker's Robot object.
    :param ball: The Ball object.
    :param game: The current Game state.
    :param pid_oren: PID controller for orientation.
    :param pid_trans: PID controller for translation.
    :param attacker_has_ball: Boolean indicating if the attacker currently has ball possession.
    :param block_ratio: Ratio (0 to 1) to position the defender on the line between attacker and goal.
                        A smaller ratio means closer to the attacker.
    :param max_ball_follow_dist: If the attacker doesn't have the ball, this is the maximum distance
                                 the defender will allow itself to be from the ball.
    :return: A RobotCommand for the defender robot.
    """
    if attacker_has_ball:
        # ========== Prioritize blocking the shot line ==========
        ax, ay = attacker.p.x, attacker.p.y

        # Assuming our goal is on the negative x-axis side for this calculation
        # This should ideally use game.field.our_goal_center or similar
        gx, gy = (
            -game.field.enemy_goal_line.coords[0][0],
            0,
        )  # This implies enemy_goal is our goal if we use their definition

        agx, agy = (gx - ax), (gy - ay)
        dist_ag = math.hypot(agx, agy)

        if dist_ag < 1e-6:
            # Extreme edge case if attacker and goal are basically at the same point
            target_x, target_y = gx, gy
        else:
            # Position on the line from attacker to goal
            target_x = ax + block_ratio * agx
            target_y = ay + block_ratio * agy

        # Face the attacker
        face_theta = math.atan2((ay - robot.p.y), (ax - robot.p.x))
    else:
        # ========== Attacker doesn't have ball; defend ball more closely ==========
        ax, ay = attacker.p.x, attacker.p.y
        bx, by = ball.p.x, ball.p.y

        # Tentative target: move ~70% of the way toward the ball from the attacker's position
        # This keeps the defender between the attacker and the ball generally.
        abx, aby = (bx - ax), (by - ay)
        target_x = ax + 0.7 * abx
        target_y = ay + 0.7 * aby

        # If the defender is too far from the ball, adjust target to move closer to the ball
        dist_def_to_ball = math.hypot(robot.p.x - bx, robot.p.y - by)
        if dist_def_to_ball > max_ball_follow_dist:
            # Calculate ratio to move exactly to max_ball_follow_dist if current distance is greater
            # Note: this logic might make the robot move directly towards the ball,
            # potentially ignoring the attacker momentarily if it's pulled too far.
            # A better approach might be to project a point on the attacker-ball line.
            # For now, it moves along the line from robot to ball.
            ratio_to_move = max_ball_follow_dist / dist_def_to_ball
            target_x = robot.p.x + (bx - robot.p.x) * ratio_to_move
            target_y = robot.p.y + (by - robot.p.y) * ratio_to_move

        # Face the ball
        face_theta = math.atan2((by - robot.p.y), (bx - robot.p.x))

    # Note: robot_id is hardcoded to 0 here. Ideally, it should be `robot.id`.
    # This function is a utility, so it might be okay if the caller (strategy)
    # always intends this for robot 0, or if go_to_point ignores robot_id when
    # a full robot object isn't available (though 'game' is passed).
    # For now, keeping it as is, but this is a point of attention.

    cmd = go_to_point(
        game=game,
        pid_oren=pid_oren,
        pid_trans=pid_trans,
        robot_id=robot.id,
        target_coords=(target_x, target_y),
        target_oren=face_theta,
    )
    return cmd


class SoloDefenderStrategy(AbstractStrategy):
    """
    A strategy for a single designated defender robot.

    The defender uses the `improved_block_goal_and_attacker` logic to position
    itself defensively based on the ball's and an enemy attacker's position.
    It identifies the enemy closest to the ball as the primary attacker to defend against.
    """

    def __init__(
        self,
        target_id: int,
        block_ratio: float = 0.4,
        max_ball_follow_dist: float = 1.0,
    ):
        """
        Initializes the SoloDefenderStrategy.

        :param target_id: The ID of the friendly robot designated as the defender.
        :param block_ratio: The ratio used in `improved_block_goal_and_attacker`
                            for positioning between the attacker and the goal.
        :param max_ball_follow_dist: The maximum distance the defender will maintain
                                     from the ball when the attacker does not have possession.
        """
        super().__init__()
        self.target_id = target_id
        self.block_ratio = block_ratio
        self.max_ball_follow_dist = max_ball_follow_dist

    def assert_exp_robots(self, n_runtime_friendly: int, n_runtime_enemy: int) -> bool:
        """
        Checks if the expected number of robots are present for this strategy.

        Requires at least one friendly robot.

        :param n_runtime_friendly: Number of friendly robots currently in the game.
        :param n_runtime_enemy: Number of enemy robots currently in the game.
        :return: True if at least one friendly robot is present, False otherwise.
        """
        return n_runtime_friendly >= 1

    def step(self, present_future_game: PresentFutureGame):
        """
        Executes a step of the solo defender strategy.

        Identifies the defender, ball, and the most threatening enemy attacker.
        Then, commands the defender using the `improved_block_goal_and_attacker` logic.

        :param present_future_game: The current and predicted state of the game.
        :return: None
        """
        game = present_future_game.current

        # This line was present in the original `improved_block_goal_and_attacker`
        # but `our_goal_center` isn't used in this `step` method directly.
        # It is relevant for `improved_block_goal_and_attacker`'s goal calculation if adapted.
        # our_goal_center = (-4.5, 0) if game.my_team_is_right else (4.5, 0)

        if self.target_id not in game.friendly_robots:
            # Defender not available, cannot execute strategy
            return

        defender = game.friendly_robots[self.target_id]
        ball = game.ball

        if not game.enemy_robots:
            # No enemies, perhaps stay near goal or ball - for now, do nothing specific
            # Or, implement a default positioning, e.g., go to a defensive spot.
            # For simplicity, if no enemies, the concept of "attacker" is moot.
            # Let's make it stay near our goal center.
            our_goal_x = (
                -game.field.length / 2
                if game.my_team_is_right
                else game.field.length / 2
            )
            # A bit in front of the goal
            defensive_pos_x = our_goal_x * 0.8
            cmd_idle = go_to_point(
                game=game,
                pid_oren=self.pid_oren,
                pid_trans=self.pid_trans,
                robot_id=defender.id,
                target_coords=(defensive_pos_x, 0),
                target_oren=math.atan2(
                    ball.p.y - defender.p.y, ball.p.x - defender.p.x
                ),  # Face ball
            )
            self.robot_controller.add_robot_commands(cmd_idle, self.target_id)
            self.robot_controller.send_robot_commands()
            return

        # Find enemy closest to the ball (considered the primary attacker)
        closest_enemy_attacker = min(
            game.enemy_robots.values(),
            key=lambda e: math.hypot(e.p.x - ball.p.x, e.p.y - ball.p.y),
        )

        # Determine if the closest enemy attacker has the ball
        # This is a simple heuristic; a more robust check might involve ball sensor data from the enemy.
        # For now, we assume `attacker_has_ball` parameter in `improved_block_goal_and_attacker`
        # would be determined by more sophisticated logic if available.
        # The original call hardcoded `attacker_has_ball=True`. This should be determined dynamically.
        # A simple check: if enemy is very close to ball.
        dist_enemy_to_ball = math.hypot(
            closest_enemy_attacker.p.x - ball.p.x, closest_enemy_attacker.p.y - ball.p.y
        )
        # Assuming 'has_ball' if within, say, 0.15 meters (robot radius + ball radius roughly)
        enemy_has_ball_guess = dist_enemy_to_ball < ROBOT_RADIUS + 0.03

        cmd = improved_block_goal_and_attacker(
            robot=defender,
            attacker=closest_enemy_attacker,
            ball=ball,
            game=game,
            pid_oren=self.pid_oren,
            pid_trans=self.pid_trans,
            attacker_has_ball=enemy_has_ball_guess,  # Using the dynamic guess
            block_ratio=self.block_ratio,
            max_ball_follow_dist=self.max_ball_follow_dist,
        )
        self.robot_controller.add_robot_commands(cmd, self.target_id)
        self.robot_controller.send_robot_commands()
        return
