from robot_control.src.high_level_skills import DribbleToTarget
from robot_control.src.intent import score_goal_atomic
from strategy.abstract_strategy import AbstractStrategy
from entities.game.present_future_game import PresentFutureGame
from robot_control.src.skills import go_to_ball
import math


class SoloAttackerStrategy(AbstractStrategy):
    """
    A strategy for a single attacker robot.

    The robot will attempt to get the ball. Once it has the ball, it will
    either dribble towards the opponent's goal or attempt to shoot,
    based on its distance to the goal.
    """

    def __init__(self, target_id: int):
        """
        Initializes the SoloAttackerStrategy.

        :param target_id: The ID of the friendly robot designated as the attacker.
        """
        super().__init__()
        self.target_id = target_id
        # The DribbleToTarget target_coords will be updated dynamically in `step` if needed,
        # or it will use its default if target_coords are static (e.g. enemy goal).
        # For now, initializing with (0,0) as a placeholder.
        self.dribble_task = DribbleToTarget(
            robot_id=self.target_id,
            target_coords=(0, 0),  # Placeholder, will be updated in step
            cooldown_sec=0.16,
        )

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
        Executes a step of the solo attacker strategy.

        If the robot does not have the ball, it moves to intercept the ball.
        If the robot has the ball, it decides whether to dribble closer to the
        opponent's goal or to shoot, based on its current distance from the goal.

        :param present_future_game: The current and predicted state of the game.
        :return: None
        """
        game = present_future_game.current

        if self.target_id not in game.friendly_robots:
            # Attacker not available, cannot execute strategy
            return

        friendly_robot = game.friendly_robots[self.target_id]

        # Determine enemy goal position
        enemy_goal_x = game.field.enemy_goal_line.coords[0][0]
        enemy_goal_y = 0  # Assuming goal center is at y=0

        # Update dribble target to enemy goal
        self.dribble_task.target_coords = (enemy_goal_x, enemy_goal_y)

        friendly_dist_from_goal = math.hypot(
            enemy_goal_x - friendly_robot.p.x, enemy_goal_y - friendly_robot.p.y
        )

        cmd = None
        if not friendly_robot.has_ball:
            # If the robot does not have the ball, command it to go to the ball.
            cmd = go_to_ball(
                game=game,
                pid_oren=self.pid_oren,
                pid_trans=self.pid_trans,
                robot_id=self.target_id,
            )
        else:
            # If the robot has the ball, decide to shoot or dribble.
            # The criterion used here is a simple distance check.
            # This can be extended with more complex logic (e.g., checking for clear shot path).
            shoot_distance_threshold = (
                3.0  # Distance in meters within which to consider shooting
            )

            if friendly_dist_from_goal < shoot_distance_threshold:
                # Robot is close enough to the goal, attempt to score.
                cmd = score_goal_atomic(
                    game=game,
                    shooter_id=self.target_id,
                    pid_oren=self.pid_oren,
                    pid_trans=self.pid_trans,
                    force_shoot=False,  # Let skill decide if it should shoot or align first
                )
            else:
                # Robot is too far to shoot, dribble towards the enemy goal.
                cmd = self.dribble_task.enact(game, self.pid_oren, self.pid_trans)

        if cmd:
            self.robot_controller.add_robot_commands(cmd, self.target_id)
            self.robot_controller.send_robot_commands()
        return
