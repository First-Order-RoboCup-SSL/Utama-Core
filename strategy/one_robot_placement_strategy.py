
from typing import Callable, Dict, Tuple
from config.starting_formation import LEFT_START_ONE, RIGHT_START_ONE
from entities.data.command import RobotCommand
from entities.game.present_future_game import PresentFutureGame
from motion_planning.src.pid.pid import PID, TwoDPID, get_grsim_pids
from robot_control.src.skills import face_ball, go_to_point
from robot_control.src.tests.utils import one_robot_placement
from global_utils.math_utils import rotate_vector
from strategy.behaviour_trees.behaviour_tree_strategy import BehaviourTreeStrategy
from rsoccer_simulator.src.ssl.envs.standard_ssl import SSLStandardEnv
from team_controller.src.controllers import RSimController
from strategy.strategy import Strategy
import numpy as np
import math
import logging

logger = logging.getLogger(__name__)

from team_controller.src.controllers.common.robot_controller_abstract import AbstractRobotController


class RobotPlacmentStrategy(Strategy):
    # TODO: Maybe we could bundle the pid into another layer on top of the robot controller to combine the 
    # PID for each sim with the corresponding robot controller
    def __init__(self, robot_controller: AbstractRobotController, pid_factory: Callable[[], Tuple[PID, TwoDPID]], id: int, invert: bool, is_yellow: bool):
        super().__init__(robot_controller)
        self.pid_oren, self.pid_trans = pid_factory()
        # self.env = env
        
        self.id = id
        
        # env_controller = RSimController(env)
        # env.reset()
        # # Move the other defender out of the way
        # for i in range(0, 6):
        #     if self.id != i:
        #         env_controller.set_robot_presence(i, is_yellow, False)
        #     env_controller.set_robot_presence(i, not is_yellow, False)
        
        # env.teleport_ball(1, 0)

        self.ty = 0 if invert else 1
        self.tx = -1
        

    def step(self, present_future_game: PresentFutureGame):
        """Closure which advances the simulation by one step"""
        game = present_future_game.current
        friendly_robots = game.friendly_robots
        
        if game.friendly_robots and game.ball is not None:
            friendly_robots = game.friendly_robots
            bx, by = game.ball.p.x, game.ball.p.y
            rp = friendly_robots[self.id].p
            cx, cy, co = rp.x, rp.y, friendly_robots[self.id].orientation
            error = math.dist((self.tx, self.ty), (cx, cy))

            switch = error < 0.05
            if switch:
                self.ty *= -1
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

