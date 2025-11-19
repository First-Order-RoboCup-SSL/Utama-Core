from typing import Tuple

import numpy as np

from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game
from utama_core.motion_planning.src.controllers.pid_controller import PIDController
from utama_core.motion_planning.src.mpc.omni_mpc import (
    OmnidirectionalMPC,
    OmniMPCConfig,
)


class MPCController(PIDController):
    """
    Python-based MPC Controller.
    Uses OSQP/CVXPY via the OmnidirectionalMPC class.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize Python MPC
        config = OmniMPCConfig(
            T=5, DT=0.05, max_vel=2.0, max_accel=3.0, Q_pos=80.0, Q_vel=200.0, R_accel=0.05, Q_obstacle=50.0
        )
        self.mpc = OmnidirectionalMPC(config)
        print("[MPCController] Python MPC Initialized")

    def calculate(self, game: Game, robot_id: int, target_pos: Vector2D, target_oren: float) -> Tuple[Vector2D, float]:

        # 1. Run Base PID (Calculates Rotation + Default Translation)
        pid_vel, angular_vel = super().calculate(game, robot_id, target_pos, target_oren)

        # 2. Prepare MPC Data
        robot = game.friendly_robots[robot_id]
        current_state = np.array([robot.p.x, robot.p.y, robot.v.x, robot.v.y])
        goal_pos_arr = (target_pos.x, target_pos.y)

        obstacles = []
        for enemy in game.enemy_robots.values():
            obstacles.append((enemy.p.x, enemy.p.y, enemy.v.x, enemy.v.y, 0.09))
        for fid, friend in game.friendly_robots.items():
            if fid != robot_id:
                obstacles.append((friend.p.x, friend.p.y, friend.v.x, friend.v.y, 0.09))

        # 3. Solve
        vx, vy, info = self.mpc.get_control_velocities(current_state, goal_pos_arr, obstacles)

        if info["success"]:
            return Vector2D(vx, vy), angular_vel
        else:
            # Fallback to PID if MPC fails
            return pid_vel, angular_vel
