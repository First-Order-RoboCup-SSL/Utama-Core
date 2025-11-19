from typing import Tuple

import numpy as np

from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game
from utama_core.motion_planning.src.controllers.pid_controller import PIDController

# Try to import the compiled C++ extension we are about to build
try:
    import mpc_cpp_extension

    CPP_AVAILABLE = True
except ImportError:
    print("[MPCCppController] ⚠️ WARNING: C++ Extension not built/found. Falling back to PID.")
    CPP_AVAILABLE = False


class MPCCppController(PIDController):
    """
    High Performance C++ MPC Controller.
    Wrapper around the 'mpc_cpp_extension' module.
    """

    def __init__(self, mode, rsim_env=None):
        # Initialize PID (for rotation and fallback)
        super().__init__(mode, rsim_env)
        self.mpc = None

        if CPP_AVAILABLE:
            # Configure C++ MPC
            config = mpc_cpp_extension.MPCConfig()
            config.T = 5
            config.DT = 0.05
            config.max_vel = 2.0
            config.max_accel = 3.0
            config.Q_pos = 200.0
            config.Q_vel = 20.0
            config.R_accel = 0.5
            config.obstacle_buffer_ratio = 1.25
            config.safety_vel_coeff = 0.15

            self.mpc = mpc_cpp_extension.OmniMPC(config)
            print("[MPCCppController] ✅ C++ MPC Engine Loaded Successfully")

    def calculate(self, game: Game, robot_id: int, target_pos: Vector2D, target_oren: float) -> Tuple[Vector2D, float]:

        # 1. Run Base PID to get Angular Velocity (Rotation)
        # We still use Python PID for turning because it's simple and fast enough
        pid_vel, angular_vel = super().calculate(game, robot_id, target_pos, target_oren)

        if self.mpc is None:
            return pid_vel, angular_vel

        robot = game.friendly_robots[robot_id]

        # 2. Prepare Data for C++
        # Convert objects to raw numpy arrays for fast transfer
        current_state = np.array([robot.p.x, robot.p.y, robot.v.x, robot.v.y])
        goal_pos_arr = np.array([target_pos.x, target_pos.y])

        obstacles = []
        # Collect enemies
        for enemy in game.enemy_robots.values():
            obstacles.append([enemy.p.x, enemy.p.y, enemy.v.x, enemy.v.y, 0.09])
        # Collect friends (except self)
        for fid, friend in game.friendly_robots.items():
            if fid != robot_id:
                obstacles.append([friend.p.x, friend.p.y, friend.v.x, friend.v.y, 0.09])

        # 3. Call C++ (Microseconds)
        vx, vy, success = self.mpc.get_control_velocities(current_state, goal_pos_arr, obstacles)

        if success:
            return Vector2D(vx, vy), angular_vel

        # Fallback
        return pid_vel, angular_vel
