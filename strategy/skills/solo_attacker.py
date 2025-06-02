from robot_control.src.high_level_skills import DribbleToTarget
from robot_control.src.intent import score_goal_atomic
from strategy.abstract_strategy import AbstractStrategy
from entities.game.present_future_game import PresentFutureGame
from robot_control.src.skills import go_to_ball
import math


class SoloAttackerStrategy(AbstractStrategy):
    def __init__(self, target_id: int):
        super().__init__()
        self.target_id = target_id
        self.dribble_task = DribbleToTarget(
            robot_id=self.target_id,
            target_coords=(0, 0),
            cooldown_sec=0.1,
        )

    def assert_exp_robots(self, n_runtime_friendly: int, n_runtime_enemy: int):
        return n_runtime_friendly >= 1

    def step(self, present_future_game: PresentFutureGame):
        game = present_future_game.current

        friendly_robot = game.friendly_robots[self.target_id]
        ball = game.ball
        friendly_dist_from_ball = math.hypot(
            ball.p.x - friendly_robot.p.x, ball.p.y - friendly_robot.p.y
        )
        goal_x = game.field.enemy_goal_line.coords[0][0]
        friendly_dist_from_goal = math.hypot(
            goal_x - friendly_robot.p.x, 0 - friendly_robot.p.y
        )

        # enemy_dists_from_ball = [
        #     math.hypot(ball.p.x - enemy_robot.p.x, ball.p.y - enemy_robot.p.y)
        #     for enemy_robot in game.enemy_robots.values()
        # ]
        # enemy_dist_from_ball = min(enemy_dists_from_ball)

        # TODO: How to check has ball?
        has_ball = friendly_dist_from_ball < 0.115

        if not has_ball:
            # If the robot has no ball, go to ball
            cmd = go_to_ball(
                game=game,
                pid_oren=self.pid_oren,
                pid_trans=self.pid_trans,
                robot_id=self.target_id,
            )
        else:
            # If has ball, decide to shoot or dribble
            # The criteria used here is a simple distance check
            # This can be extended with more complex logic
            if friendly_dist_from_goal < 3:
                cmd = score_goal_atomic(
                    game=game,
                    shooter_id=self.target_id,
                    pid_oren=self.pid_oren,
                    pid_trans=self.pid_trans,
                    force_shoot=True,
                )
            else:
                cmd = self.dribble_task.enact(game, self.pid_oren, self.pid_trans)

        self.robot_controller.add_robot_commands(cmd, self.target_id)
        self.robot_controller.send_robot_commands()
