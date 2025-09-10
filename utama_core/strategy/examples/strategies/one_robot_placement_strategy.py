import math
from typing import Any

import numpy as np
import py_trees
from py_trees.composites import Selector, Sequence

from utama_core.entities.game import Game
from utama_core.global_utils.math_utils import Vector2D
from utama_core.skills.src.utils.move_utils import move
from utama_core.strategy.common.abstract_behaviour import AbstractBehaviour

# from robot_control.src.tests.utils import one_robot_placement
from utama_core.strategy.common.abstract_strategy import AbstractStrategy


class RobotPlacementStep(AbstractBehaviour):
    def __init__(self, invert: bool = False):
        super().__init__()
        self.ty = -1
        self.tx = -1 if invert else 1

    def setup_(self):
        self.blackboard.register_key(key="robot_id", access=py_trees.common.Access.READ)

    def update(self) -> py_trees.common.Status:
        """Closure which advances the simulation by one step."""
        game = self.blackboard.game.current
        rsim_env = self.blackboard.rsim_env
        id: int = self.blackboard.robot_id

        friendly_robots = game.friendly_robots

        if game.friendly_robots and game.ball is not None:
            friendly_robots = game.friendly_robots
            bx, by = game.ball.p.x, game.ball.p.y
            rp = friendly_robots[id].p
            cx, cy, _ = rp.x, rp.y, friendly_robots[id].orientation
            error = math.dist((self.tx, self.ty), (cx, cy))

            switch = error < 0.05
            if switch:
                if self.ty == -1:
                    self.ty = -2
                else:
                    self.ty = -1
                self.blackboard.motion_controller.pid_oren.reset(id)
                self.blackboard.motion_controller.pid_trans.reset(id)

            # changed so the robot tracks the ball while moving
            oren = np.atan2(by - cy, bx - cx)
            cmd = move(
                game,
                self.blackboard.motion_controller,
                id,
                Vector2D(self.tx, self.ty),
                oren,
            )
            if rsim_env:
                rsim_env.draw_point(self.tx, self.ty, color="red")
                v = game.friendly_robots[id].v
                p = game.friendly_robots[id].p
                rsim_env.draw_point(p.x + v.x * 0.2, p.y + v.y * 0.2, color="green")

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
            self.blackboard.cmd_map[id] = cmd
            return py_trees.common.Status.RUNNING


class SetBlackboardVariable(AbstractBehaviour):
    """A generic behaviour to set a variable on the blackboard."""

    def __init__(self, name: str, variable_name: str, value: Any, opp_strategy: bool = False):
        super().__init__(name=name)
        # Store the configuration, but DO NOT use the blackboard here.
        self.variable_name = variable_name
        self.value = value

    def setup_(self):
        self.blackboard.register_key(key=self.variable_name, access=py_trees.common.Access.WRITE)

    def update(self) -> py_trees.common.Status:
        # print(f"Setting {self.variable_name} to {self.value} on the blackboard.")
        self.blackboard.set(self.variable_name, self.value, overwrite=True)
        return py_trees.common.Status.SUCCESS


class RobotPlacementStrategy(AbstractStrategy):
    def __init__(self, robot_id: int):
        """Initializes the DemoStrategy with a specific robot ID.

        :param robot_id: The ID of the robot this strategy will control.
        """
        self.robot_id = robot_id
        super().__init__()

    def assert_exp_robots(self, n_runtime_friendly: int, n_runtime_enemy: int):
        return True

    def create_behaviour_tree(self) -> py_trees.behaviour.Behaviour:
        """Factory function to create a complete score_goal behaviour tree."""

        set_rbt_id = SetBlackboardVariable(
            name="SetTargetRobotID",
            variable_name="robot_id",
            value=self.robot_id,
        )

        # Create the root of the behaviour tree
        coach_root = Sequence(name="CoachRoot", memory=False)
        coach_root.add_children([set_rbt_id, RobotPlacementStep()])

        return coach_root
