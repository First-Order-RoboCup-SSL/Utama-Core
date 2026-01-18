from typing import Tuple

import numpy as np

from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game
from utama_core.motion_planning.src.controllers.pid_controller import PIDController

# Try to import your new C++ module
try:
    # --- ADD THIS TRACKER ---
    import os

    import mpc_cpp_extension

    print(f"\n[DEBUG] 🕵️‍♂️ C++ MODULE LOADED FROM: {mpc_cpp_extension.__file__}")
    print(f"[DEBUG] 🕒 FILE TIMESTAMP: {os.path.getmtime(mpc_cpp_extension.__file__)}\n")
    CPP_AVAILABLE = True
except ImportError as e:
    print(f"[MPCCppController] ⚠️ IMPORT ERROR: {e}")
    CPP_AVAILABLE = False


class MPCCppController(PIDController):
    """
    High Performance C++ MPC Controller.
    Wrapper around the 'mpc_cpp_extension' module.
    """

    def __init__(self, *args, **kwargs):
        # Initialize PID (for rotation and fallback)
        super().__init__(*args, **kwargs)
        self.mpc = None

        if CPP_AVAILABLE:
            # Configure C++ MPC
            config = mpc_cpp_extension.MPCConfig()
            config.T = 20  # Prediction horizon (number of future timesteps to consider)
            config.DT = 0.05  # Time step duration in seconds (40Hz control rate)
            config.max_vel = 2.0  # Maximum velocity in m/s
            config.max_accel = 3.0  # Maximum acceleration in m/s²
            config.Q_pos = 200.0  # Position error weight (unused in heuristic solver)
            config.Q_vel = 20.0  # Velocity error weight (unused in heuristic solver)
            config.R_accel = 0.1  # Acceleration penalty weight (unused in heuristic solver)
            config.robot_radius = 0.09
            config.obstacle_buffer_ratio = 1.5  # Safety margin multiplier for obstacles (1.1 = 10% extra buffer)
            config.safety_vel_coeff = 0.5  # Velocity-dependent safety margin (higher speed = larger buffer)
            """
            config.T = 12
            config.DT = 0.05
            config.max_vel = 2.0
            config.max_accel = 3.0
            config.Q_pos = 200.0
            config.Q_vel = 20.0
            config.R_accel = 1.0
            config.obstacle_buffer_ratio = 1.1
            config.safety_vel_coeff = 0.3
            """

            self.mpc = mpc_cpp_extension.OmniMPC(config)
            print("[MPCCppController] 🚀 C++ MPC Engine Loaded")

    def calculate(self, game: Game, robot_id: int, target_pos: Vector2D, target_oren: float) -> Tuple[Vector2D, float]:

        # 1. Run Base PID to get Angular Velocity (Rotation)
        pid_vel, angular_vel = super().calculate(game, robot_id, target_pos, target_oren)

        if self.mpc is None:
            return pid_vel, angular_vel

        robot = game.friendly_robots[robot_id]

        # 2. Prepare Data for C++ (Numpy arrays are fast to transfer)
        # State: [x, y, vx, vy]
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

        # Fallback to PID if C++ solver fails (unlikely with current logic)
        return pid_vel, angular_vel
