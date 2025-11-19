from __future__ import annotations

import time
from typing import Tuple

import numpy as np

from utama_core.config.physical_constants import ROBOT_RADIUS
from utama_core.entities.data.command import RobotCommand
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game
from utama_core.global_utils.math_utils import rotate_vector
from utama_core.motion_planning.src.common.motion_controller import MotionController

# --- LOGGING CONFIGURATION ---
ENABLE_PERFORMANCE_LOGGING = True  # Set to False to silence console

# Global tracking variables
_call_count = 0
_last_log_time = time.time()
_robot_speeds = {}  # Stores {robot_id: speed}
_total_compute_time = 0.0


def move(
    game: Game,
    motion_controller: MotionController,
    robot_id: int,
    target_coords: Vector2D,
    target_oren: float,
    dribbling: bool = False,
) -> RobotCommand:
    """
    Calculate the robot command to move towards a target.
    Delegates logic to motion_controller but handles Performance Logging.
    """
    robot = game.friendly_robots[robot_id]

    # --- 1. RUN CONTROLLER & MEASURE TIME ---
    start_time = time.perf_counter()

    global_velocity, angular_vel = motion_controller.calculate(
        game=game,
        robot_id=robot_id,
        target_pos=target_coords,
        target_oren=target_oren,
    )

    compute_duration = time.perf_counter() - start_time

    # --- 2. PERFORMANCE LOGGING ---
    if ENABLE_PERFORMANCE_LOGGING:
        global _call_count, _last_log_time, _robot_speeds, _total_compute_time

        # Update stats
        _call_count += 1
        _total_compute_time += compute_duration

        # Track speed for this robot
        speed = np.hypot(global_velocity.x, global_velocity.y)
        _robot_speeds[robot_id] = speed

        # Log every ~60 calls (approx 1 second at 60Hz)
        if _call_count % 60 == 0:
            current_time = time.time()
            time_diff = current_time - _last_log_time

            # Avoid division by zero
            fps = 60.0 / time_diff if time_diff > 0 else 0.0
            avg_latency_ms = (_total_compute_time / 60) * 1000

            # Reset counters
            _last_log_time = current_time
            _total_compute_time = 0.0

            # Format robot info string
            robot_info = " | ".join([f"R{rid}: {spd:.2f}m/s" for rid, spd in sorted(_robot_speeds.items())])

            # Print Unified Stats
            # We try to guess the controller name for better logs
            ctrl_name = motion_controller.__class__.__name__.replace("Controller", "")
            print(f"[{ctrl_name} Stats] FPS: {fps:.1f} | Latency: {avg_latency_ms:.2f}ms | {robot_info}")

    # --- 3. CONVERT TO LOCAL FRAME ---
    forward_vel, left_vel = rotate_vector(global_velocity.x, global_velocity.y, robot.orientation)

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
