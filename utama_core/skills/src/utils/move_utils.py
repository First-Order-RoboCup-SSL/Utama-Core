from typing import Tuple

import numpy as np

from utama_core.config.physical_constants import BALL_RADIUS, ROBOT_RADIUS
from utama_core.entities.data.command import RobotCommand
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game
from utama_core.global_utils.math_utils import rotate_vector
from utama_core.motion_planning.src.common.motion_controller import MotionController

### TODO: IT IS CONTENTIOUS IF WE EVEN NEED THIS MOVE FUNCTION ANYMORE. POTENTIALLY JUST SEND DIRECTLY TO MOTION CONTROLLER ###


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

    global_velocity, angular_vel = motion_controller.calculate(
        game=game,
        robot_id=robot_id,
        target_pos=target_coords,
        target_oren=target_oren,
    )

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
    """
    Turns the robot on the spot to face the target orientation.
    When the robot is in dribbler contact with the ball, pivots around the ball
    rather than the robot's own center.
    """
    PIVOT_RADIUS = ROBOT_RADIUS + BALL_RADIUS  # distance from robot center to ball center at contact

    # Below this clearance to another robot's body, the pivot's lateral push is
    # treated as physically blocked rather than merely close. A full-length
    # tournament traced a match where a robot ended up body-to-body with an
    # enemy while pivoting on the ball: the pivot term always points the same
    # way regardless of what's there, so it kept commanding a lateral push
    # straight into the enemy every tick. Box2D's own contact resolution
    # cancelled the resulting motion each time, and since the geometry never
    # changed, the identical command (and identical freeze) recurred forever
    # -- a real deadlock, not a stale-state bug. See lead_and_support.py's
    # COMMITTED_FROZEN stall in split_shape_vs_switch_of_play_RK.
    #
    # `2 * ROBOT_RADIUS` alone (exact body-touching distance, zero margin) was
    # found live to still deadlock: a decoy_and_overload COMMITTED_FROZEN
    # traced an enemy parked at 0.1982 m -- 1.8 mm outside that exact cutoff
    # -- so the guard never fired, yet rsim's real rigid-body contact still
    # resisted the commanded push at that range (rotation crawled at ~3 deg/s
    # against a commanded ~190 deg/s), a genuine sustained physical deadlock,
    # not a one-tick fluke. Extra headroom above contact distance, same style
    # as `_pass_and_score.py`'s `_MIN_SETUP_CLEARANCE`/`block_shape.py`'s
    # `_AREA_CLEARANCE`, so a robot sitting just outside exact contact still
    # counts as blocking.
    _PIVOT_CLEARANCE_M = 2 * ROBOT_RADIUS + 0.05

    robot = game.friendly_robots[robot_id]
    ball = game.ball

    # Pivot around the ball when the robot is in dribbler contact (IR or visual proximity).
    in_contact = robot.has_ball or (ball is not None and robot.p.distance_to(ball.p.to_2d()) < PIVOT_RADIUS)

    # If the pivot's lateral push is blocked (checked below using the
    # in-place-hold turn's own angular_vel), steer `move()`'s target away
    # from the blocking enemy instead of holding `robot.p` -- see the
    # blocked branch below for why holding position doesn't actually yield a
    # working turn.
    move_target = robot.p
    if in_contact:
        held_turn = move(
            game=game,
            motion_controller=motion_controller,
            robot_id=robot_id,
            target_coords=robot.p,
            target_oren=target_oren,
            dribbling=dribbling,
        )
        local_left_vel = -held_turn.angular_vel * PIVOT_RADIUS
        pivot_push_global = rotate_vector(0.0, local_left_vel, -robot.orientation)
        for enemy in game.enemy_robots.values():
            if enemy is None:
                continue
            to_enemy = enemy.p - robot.p
            distance = to_enemy.mag()
            if distance >= _PIVOT_CLEARANCE_M or distance == 0:
                continue
            if to_enemy.x * pivot_push_global[0] + to_enemy.y * pivot_push_global[1] <= 0:
                continue
            # Merely zeroing `local_left_vel` here (an earlier fix attempt)
            # does NOT produce a working pure in-place spin: rotating around
            # the ball at `PIVOT_RADIUS` inherently requires the chassis to
            # sweep an arc through space, and dropping only the model's own
            # lateral estimate leaves that physical requirement unmet --
            # rsim's contact solver then has to supply/oppose the sweep
            # itself, which measured live as ZERO net motion (position and
            # orientation bit-identical tick over tick, not merely slow) for
            # the rest of the match, not a working degraded turn. Turning off
            # the dribbler to free the spin isn't safe either --
            # `_apply_dribbler_release_kicks` fires a release kick the moment
            # `dribbler` drops while moving, punting the ball away. Instead,
            # move `move()`'s own hold-position target away from the
            # blocking enemy (along the enemy->robot direction, out to
            # clearance) so the motion controller actually drives the
            # chassis clear before the pivot resumes, rather than holding
            # `robot.p` while commanding a sweep the enemy's body prevents.
            away = robot.p - enemy.p
            away_dist = away.mag()
            if away_dist > 1e-6:
                move_target = robot.p + away * (_PIVOT_CLEARANCE_M / away_dist)
            break

    turn = move(
        game=game,
        motion_controller=motion_controller,
        robot_id=robot_id,
        target_coords=move_target,
        target_oren=target_oren,
        dribbling=dribbling,
    )

    if in_contact:
        local_left_vel = -turn.angular_vel * PIVOT_RADIUS
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
