from __future__ import annotations
from typing import Tuple, Optional, TYPE_CHECKING

import time
import numpy as np

if TYPE_CHECKING:
    from utama_core.motion_planning.src.mpc.omni_mpc import OmnidirectionalMPC

from utama_core.config.settings import ROBOT_RADIUS
from utama_core.entities.data.command import RobotCommand
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game
from utama_core.global_utils.math_utils import rotate_vector
from utama_core.motion_planning.src.motion_controller import MotionController

# MPC imports
try:
    from utama_core.motion_planning.src.mpc.omni_mpc import OmnidirectionalMPC, OmniMPCConfig
    MPC_AVAILABLE = True
    # Global MPC instance (reuse solver for speed)
    _global_mpc = None
except ImportError:
    MPC_AVAILABLE = False
    _global_mpc = None

# Control mode selection
USE_MPC = False  # Set to True to use MPC, False for PID
ENABLE_MPC_LOGGING = True  # Log MPC performance every N frames

# Performance tracking
_mpc_call_count = 0
_mpc_total_solve_time = 0.0
_mpc_failures = 0
_mpc_robot_speeds = {} 
_last_mpc_log_time = time.time() 

# PID tracking
_pid_call_count = 0
_pid_robot_speeds = {} 
_last_pid_log_time = time.time()


def _get_mpc_instance() -> Optional['OmnidirectionalMPC']:
    """Get or create global MPC instance"""
    global _global_mpc
    if not MPC_AVAILABLE:
        return None
    if _global_mpc is None:
        # --- AGGRESSIVE CONFIGURATION ---
        config = OmniMPCConfig(
            T=5,            # Short horizon for speed
            DT=0.05,        # 50ms steps
            max_vel=4.0,    
            max_accel=15.0, # Aggressive acceleration
            Q_pos=80.0,     
            Q_vel=200.0,    # Force velocity tracking
            R_accel=0.01,   
            Q_obstacle=50.0,
            safety_base=0.15,
            safety_vel_coeff=0.05,
        )
        _global_mpc = OmnidirectionalMPC(config)
        print("[move_utils] CVXPY-based MPC controller initialized (AGGRESSIVE MODE)")
    return _global_mpc


def _collect_obstacles(game: Game, robot_id: int):
    """Collect obstacle information for MPC (all other robots)"""
    obstacles = []

    # Add enemy robots
    for enemy in game.enemy_robots.values():
        obstacles.append((
            enemy.p.x,
            enemy.p.y,
            enemy.v.x,
            enemy.v.y,
            0.09  # Enemy robot radius
        ))

    # Add friendly robots (except self)
    for fid, friendly in game.friendly_robots.items():
        if fid != robot_id:
            obstacles.append((
                friendly.p.x,
                friendly.p.y,
                friendly.v.x,
                friendly.v.y,
                0.09  # Friendly robot radius
            ))

    return obstacles


def move(
    game: Game,
    motion_controller: MotionController,
    robot_id: int,
    target_coords: Vector2D,
    target_oren: float,
    dribbling: bool = False,
) -> RobotCommand:
    """Calculate the robot command to move towards a target point with a specified orientation."""
    robot = game.friendly_robots[robot_id]
    target_x, target_y = target_coords.x, target_coords.y

    # Choose control method
    if USE_MPC and MPC_AVAILABLE:
        # Use MPC for translation control
        mpc = _get_mpc_instance()

        if mpc is not None and target_x is not None and target_y is not None:
            # Current state: [x, y, vx, vy]
            current_state = np.array([
                robot.p.x,
                robot.p.y,
                robot.v.x,
                robot.v.y
            ])

            # Goal position
            goal_pos = (target_x, target_y)

            # Collect obstacles (other robots)
            obstacles = _collect_obstacles(game, robot_id)

            # Solve MPC
            try:
                global_x, global_y, info = mpc.get_control_velocities(
                    current_state, goal_pos, obstacles
                )

                # Track performance
                global _mpc_call_count, _mpc_total_solve_time, _mpc_failures, _mpc_robot_speeds, _last_mpc_log_time
                _mpc_call_count += 1
                if info['success']:
                    _mpc_total_solve_time += info['solve_time']
                else:
                    _mpc_failures += 1

                # Track this robot's speed
                speed = np.hypot(global_x, global_y)
                dist_to_goal = np.hypot(target_x - robot.p.x, target_y - robot.p.y)
                _mpc_robot_speeds[robot_id] = (speed, dist_to_goal)

                # Periodic logging (every 1 sec approx)
                if ENABLE_MPC_LOGGING and _mpc_call_count % 60 == 0:
                    # --- CALCULATE FPS ---
                    current_time = time.time()
                    time_diff = current_time - _last_mpc_log_time
                    # effective_fps = (Calls / Diff) / Num_Robots (Approx)
                    fps = 60.0 / time_diff if time_diff > 0 else 0.0
                    _last_mpc_log_time = current_time
                    
                    avg_time = (_mpc_total_solve_time / max(1, _mpc_call_count - _mpc_failures)) * 1000
                    robot_info = " | ".join([f"R{rid}: {spd:.2f}m/s" for rid, (spd, dist) in sorted(_mpc_robot_speeds.items())])
                    
                    # Print with FPS
                    print(f"[MPC Stats] FPS: {fps:.1f} | Avg Solve: {avg_time:.2f}ms | {robot_info}")

                # Check for close obstacles (collision warning)
                if obstacles:
                    min_dist = min(np.hypot(robot.p.x - obs[0], robot.p.y - obs[1]) for obs in obstacles)
                    if min_dist < 0.18 and _mpc_call_count % 60 == 0:  # Warn every second
                        # print(f"[MPC Warning] Robot {robot_id} very close to obstacle: {min_dist:.3f}m")
                        pass

            except Exception as e:
                # Fallback to PID if MPC fails
                print(f"[MPC] Failed, using PID fallback: {e}")
                _mpc_failures += 1
                global_x, global_y = motion_controller.pid_trans.calculate(
                    (target_x, target_y), (robot.p.x, robot.p.y), robot_id
                )
        else:
            # Fallback to PID
            global_x, global_y = motion_controller.pid_trans.calculate(
                (target_x, target_y), (robot.p.x, robot.p.y), robot_id
            )
    else:
        # Use PID for translation control
        pid_trans = motion_controller.pid_trans
        if target_x is not None and target_y is not None:
            global_x, global_y = pid_trans.calculate((target_x, target_y), (robot.p.x, robot.p.y), robot_id)

            # PID logging
            global _pid_call_count, _pid_robot_speeds, _last_pid_log_time
            _pid_call_count += 1

            # Track this robot's speed
            speed = np.hypot(global_x, global_y)
            dist_to_goal = np.hypot(target_x - robot.p.x, target_y - robot.p.y)
            _pid_robot_speeds[robot_id] = (speed, dist_to_goal)

            # Periodic logging (every 1 sec)
            if ENABLE_MPC_LOGGING and _pid_call_count % 60 == 0:
                # --- CALCULATE FPS ---
                current_time = time.time()
                time_diff = current_time - _last_pid_log_time
                fps = 60.0 / time_diff if time_diff > 0 else 0.0
                _last_pid_log_time = current_time

                robot_info = " | ".join([f"R{rid}: {spd:.2f}m/s" for rid, (spd, dist) in sorted(_pid_robot_speeds.items())])
                
                # Print with FPS
                print(f"[PID Stats] FPS: {fps:.1f} | {robot_info}")
        else:
            global_x = 0
            global_y = 0

    # Convert global velocities to robot-local frame
    forward_vel, left_vel = rotate_vector(global_x, global_y, robot.orientation)

    # Use PID for orientation control (keep this regardless of MPC)
    if target_oren is not None:
        angular_vel = motion_controller.pid_oren.calculate(target_oren, robot.orientation, robot_id)
    else:
        angular_vel = 0

    return RobotCommand(
        local_forward_vel=forward_vel,
        local_left_vel=left_vel,
        angular_vel=angular_vel,
        kick=0,
        chip=0,
        dribble=1 if dribbling else 0,
    )


def face_ball(current: Tuple[float, float], ball: Tuple[float, float]) -> float:
    """Calculate the angle to face the ball from the current position."""
    return np.arctan2(ball[1] - current[1], ball[0] - current[0])


def turn_on_spot(
    game: Game,
    motion_controller: MotionController,
    robot_id: int,
    target_oren: float,
    dribbling: bool = False,
) -> RobotCommand:
    """Turns the robot on the spot to face the target orientation.

    pivot_on_ball: If True, the robot will pivot on the ball, otherwise it will pivot on its own centre.
    """
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