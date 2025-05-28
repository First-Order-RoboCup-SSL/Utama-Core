from strategy.abstract_strategy import AbstractStrategy
from entities.game.present_future_game import PresentFutureGame
from robot_control.src.skills import go_to_ball


class GoToBallStrategy(AbstractStrategy):
    def __init__(self, target_id: int):
        super().__init__()
        self.target_id = target_id

    def assert_exp_robots(self, n_runtime_friendly: int, n_runtime_enemy: int):
        if n_runtime_friendly <= 3 and n_runtime_enemy <= 3:
            return True
        return False

    def step(self, present_future_game: PresentFutureGame):
        game = present_future_game.current
        cmd = go_to_ball(
            game,
            self.pid_oren,
            self.pid_trans,
            self.target_id,
        )
        self.robot_controller.add_robot_commands(cmd, self.target_id)
        self.robot_controller.send_robot_commands()
        if self.env:
            v = game.friendly_robots[self.target_id].v
            p = game.friendly_robots[self.target_id].p
            self.env.draw_point(p.x + v.x * 0.2, p.y + v.y * 0.2, color="green")
