from entities.game import Game
from robot_control.src.skills import go_to_point

# from robot_control.src.tests.utils import one_robot_placement
from strategy.abstract_strategy import AbstractStrategy
import numpy as np
import math


class RobotPlacementStrategy(AbstractStrategy):
    def assert_exp_robots(self, n_runtime_friendly: int, n_runtime_enemy: int):
        return True

    def __init__(self, id: int, invert: bool = False):
        super().__init__()
        self.id = id

        self.ty = -1
        self.tx = -1 if invert else 1

    def step(self, game: Game):
        """Closure which advances the simulation by one step"""
        friendly_robots = game.friendly_robots

        if game.friendly_robots and game.ball is not None:
            friendly_robots = game.friendly_robots
            bx, by = game.ball.p.x, game.ball.p.y
            rp = friendly_robots[self.id].p
            cx, cy, _ = rp.x, rp.y, friendly_robots[self.id].orientation
            error = math.dist((self.tx, self.ty), (cx, cy))

            switch = error < 0.05
            if switch:
                if self.ty == -1:
                    self.ty = -2
                else:
                    self.ty = -1
                self.pid_trans.reset(self.id)
                self.pid_oren.reset(self.id)

            # changed so the robot tracks the ball while moving
            oren = np.atan2(by - cy, bx - cx)
            cmd = go_to_point(
                game,
                self.pid_oren,
                self.pid_trans,
                self.id,
                (self.tx, self.ty),
                oren,
            )
            if self.env:
                self.env.draw_point(self.tx, self.ty, color="red")
                v = game.friendly_robots[self.id].v
                p = game.friendly_robots[self.id].p
                self.env.draw_point(p.x + v.x * 0.2, p.y + v.y * 0.2, color="green")

            # # Rotate the local forward and left velocities to the global frame
            # lf_x, lf_y = rotate_vector(cmd.local_forward_vel, 0, -co)
            # ll_x, ll_y = rotate_vector(0, cmd.local_left_vel, -co)

            # # Draw the local forward vector
            # self.env.draw_line([(cx, cy), (cx + lf_x, cy + lf_y)], color="blue")

            # # Draw the local left vector
            # self.env.draw_line([(cx, cy), (cx + ll_x, cy + ll_y)], color="blue")

            # # Rotate the global velocity vector
            # gx, gy = rotate_vector(cmd.local_forward_vel, cmd.local_left_vel, -co)

            # # Draw the global velocity vector
            # self.env.draw_line([(cx, cy), (gx + cx, gy + cy)], color="black", width=2)
            self.robot_controller.add_robot_commands(cmd, self.id)
            self.robot_controller.send_robot_commands()
