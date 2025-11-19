from __future__ import annotations

import time
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

# --- IMPORTS FROM MAIN BRANCH (NEW STRUCTURE) ---
from utama_core.config.physical_constants import ROBOT_RADIUS
from utama_core.entities.data.command import RobotCommand
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game
from utama_core.global_utils.math_utils import rotate_vector
from utama_core.motion_planning.src.common.motion_controller import MotionController

# if TYPE_CHECKING:
# from utama_core.motion_planning.src.mpc.omni_mpc import OmnidirectionalMPC

# --- MPC SETUP (FROM YOUR WORK) ---
try:
    from utama_core.motion_planning.src.mpc.omni_mpc import (
        OmnidirectionalMPC,
        OmniMPCConfig,
    )

    MPC_AVAILABLE = True
    _global_mpc = None
except ImportError:
    MPC_AVAILABLE = False
    _global_mpc = None

# Control mode selection
USE_MPC = True  # Set to True to use MPC override
ENABLE_MPC_LOGGING = True

# Performance tracking
_mpc_call_count = 0
_mpc_total_solve_time = 0.0
_mpc_failures = 0
_mpc_robot_speeds = {}
_last_mpc_log_time = time.time()

_pid_call_count = 0
_pid_robot_speeds = {}
_last_pid_log_time = time.time()


def _get_mpc_instance() -> Optional["OmnidirectionalMPC"]:
    """Get or create global MPC instance"""
    global _global_mpc
    if not MPC_AVAILABLE:
        return None
    if _global_mpc is None:
        # --- AGGRESSIVE CONFIGURATION (Merged) ---
        config = OmniMPCConfig(
            T=5,  # Short horizon for speed (60Hz compliant)
            DT=0.05,  # 50ms steps
            max_vel=2.0,  # Limit to 2.0 m/s per your request
            max_accel=3.0,  # Realistic acceleration for 2.0 m/s
            Q_pos=80.0,
            Q_vel=100.0,
            R_accel=0.05,
            Q_obstacle=50.0,
        )
        _global_mpc = OmnidirectionalMPC(config)
        print("[move_utils] CVXPY-based MPC controller initialized (HYBRID MERGE)")
    return _global_mpc


def _collect_obstacles(game: Game, robot_id: int):
    """Collect obstacle information for MPC"""
    obstacles = []
    # Add enemy robots
    for enemy in game.enemy_robots.values():
        obstacles.append((enemy.p.x, enemy.p.y, enemy.v.x, enemy.v.y, 0.09))
    # Add friendly robots (except self)
    for fid, friendly in game.friendly_robots.items():
        if fid != robot_id:
            obstacles.append((friendly.p.x, friendly.p.y, friendly.v.x, friendly.v.y, 0.09))
    return obstacles


def move(
    game: Game,
    motion_controller: MotionController,
    robot_id: int,
    target_coords: Vector2D,
    target_oren: float,
    dribbling: bool = False,
) -> RobotCommand:
    """
    Unified move function.
    1. Calculates baseline PID (for rotation and fallback).
    2. Overwrites translation with MPC if enabled.
    """
    robot = game.friendly_robots[robot_id]

    # 1. Run Standard PID Controller (Main Branch Logic)
    # This ensures we always have angular_vel and a safe fallback
    pid_global_vel, angular_vel = motion_controller.calculate(
        game=game,
        robot_id=robot_id,
        target_pos=target_coords,
        target_oren=target_oren,
    )

    # Default velocities come from PID
    final_vx, final_vy = pid_global_vel.x, pid_global_vel.y

    # 2. Attempt MPC Override
    if USE_MPC and MPC_AVAILABLE:
        mpc = _get_mpc_instance()
        if mpc and target_coords is not None:
            current_state = np.array([robot.p.x, robot.p.y, robot.v.x, robot.v.y])
            goal_pos = (target_coords.x, target_coords.y)
            obstacles = _collect_obstacles(game, robot_id)

            try:
                mpc_vx, mpc_vy, info = mpc.get_control_velocities(current_state, goal_pos, obstacles)

                # Performance Tracking
                global _mpc_call_count, _mpc_total_solve_time, _mpc_failures, _mpc_robot_speeds, _last_mpc_log_time
                _mpc_call_count += 1

                if info["success"]:
                    _mpc_total_solve_time += info["solve_time"]

                    # OVERWRITE PID TRANSLATION WITH MPC
                    final_vx, final_vy = mpc_vx, mpc_vy

                    # Stats Update
                    speed = np.hypot(final_vx, final_vy)
                    _mpc_robot_speeds[robot_id] = (speed, 0.0)

                    # MPC Logging (with FPS)
                    if ENABLE_MPC_LOGGING and _mpc_call_count % 60 == 0:
                        current_time = time.time()
                        fps = 60.0 / max(0.001, current_time - _last_mpc_log_time)
                        _last_mpc_log_time = current_time
                        avg_time = (_mpc_total_solve_time / max(1, _mpc_call_count - _mpc_failures)) * 1000

                        robot_info = " | ".join(
                            [f"R{rid}: {spd:.2f}m/s" for rid, (spd, _) in sorted(_mpc_robot_speeds.items())]
                        )
                        print(f"[MPC Stats] FPS: {fps:.1f} | Avg Solve: {avg_time:.2f}ms | {robot_info}")
                else:
                    _mpc_failures += 1
                    # Fallback to PID (implicitly done by not overwriting final_vx/vy)

            except Exception as e:
                print(f"[MPC Error] {e}")
                _mpc_failures += 1
    else:
        # PID Stats Logging (Only if MPC is off)
        global _pid_call_count, _pid_robot_speeds, _last_pid_log_time
        _pid_call_count += 1
        speed = np.hypot(final_vx, final_vy)
        _pid_robot_speeds[robot_id] = (speed, 0.0)

        if ENABLE_MPC_LOGGING and _pid_call_count % 60 == 0:
            current_time = time.time()
            fps = 60.0 / max(0.001, current_time - _last_pid_log_time)
            _last_pid_log_time = current_time
            robot_info = " | ".join([f"R{rid}: {spd:.2f}m/s" for rid, (spd, _) in sorted(_pid_robot_speeds.items())])
            print(f"[PID Stats] FPS: {fps:.1f} | {robot_info}")

    # 3. Apply Rotation (Main Branch Logic)
    forward_vel, left_vel = rotate_vector(final_vx, final_vy, robot.orientation)

    return RobotCommand(
        local_forward_vel=forward_vel,
        local_left_vel=left_vel,
        angular_vel=angular_vel,
        kick=0,
        chip=0,
        dribble=1 if dribbling else 0,
    )


def face_ball(current: Vector2D, ball: Vector2D) -> float:
    """Calculate the angle to face the ball from the current position."""
    return current.angle_to(ball)


def turn_on_spot(
    game: Game,
    motion_controller: MotionController,
    robot_id: int,
    target_oren: float,
    dribbling: bool = False,
) -> RobotCommand:
    """Turns the robot on the spot to face the target orientation."""
    RADIUS_MODIFIER = 1.8

    turn = move(
        game=game,
        motion_controller=motion_controller,
        robot_id=robot_id,
        target_coords=game.friendly_robots[robot_id].p,
        target_oren=target_oren,
        dribbling=dribbling,
    )

    if game.friendly_robots[robot_id].has_ball:
        angular_vel = turn.angular_vel
        local_left_vel = -angular_vel * RADIUS_MODIFIER * ROBOT_RADIUS
        turn = turn._replace(local_left_vel=local_left_vel)

    return turn


def kick() -> RobotCommand:
    """Returns a command to kick the ball."""
    return RobotCommand(
        local_forward_vel=0,
        local_left_vel=0,
        angular_vel=0,
        kick=1,
        chip=0,
        dribble=0,
    )


def empty_command(dribbler_on: bool = False) -> RobotCommand:
    return RobotCommand(
        local_forward_vel=0,
        local_left_vel=0,
        angular_vel=0,
        kick=0,
        chip=0,
        dribble=dribbler_on,
    )
