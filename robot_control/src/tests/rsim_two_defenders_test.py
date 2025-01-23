import sys
import os
import numpy as np
import pytest
from motion_planning.src.pid.pid import TwoDPID, get_rsim_pids
from robot_control.src.skills import (
    get_goal_centre,
    go_to_ball,
    go_to_point,
    align_defenders,
    to_defense_parametric,
    face_ball,
    velocity_to_orientation,
)
from team_controller.src.controllers import RSimRobotController
from rsoccer_simulator.src.ssl.envs.standard_ssl import SSLStandardEnv
from entities.game import Game
from robot_control.src.intent import find_likely_enemy_shooter, score_goal
from motion_planning.src.pid import PID
from team_controller.src.controllers.sim.rsim_robot_controller import PVPManager
from team_controller.src.config.settings import TIMESTEP
from robot_control.src.tests.utils import one_robot_placement, setup_pvp
import logging

logger = logging.getLogger(__name__)
from typing import List


def defend(
    pid_oren: PID,
    pid_2d: TwoDPID,
    game: Game,
    controller: RSimRobotController,
    is_yellow: bool,
    defender_ids: List[int],
    env,
):
    # Assume that is_yellow <-> not is_left here # TODO : FIX
    """
    Strategy for two defenders,

    if only one attacker:
        predict shot location, and extrapolate target parametric position tcenter
        need the two robots to span tcenter

        Let w = defender gap, assume w is small
        Suppose t1,t2 are desired defender positions

        approx arc length t1 -> tcenter as straight line length ROBOT_RADIUS / 2
        approx arc length tcenter -> t2 as straight line length ROBOT_RADIUS / 2

        find unit gradient vector at tcenter, (tx, ty) +- w/2 * deriv at(tx, ty) is t1,t2

    """

    friendly, enemy, balls = game.get_my_latest_frame(my_team_is_yellow=is_yellow)
    defender1_id, def2 = defender_ids
    shooters_data = find_likely_enemy_shooter(enemy, balls)
    orientation = None
    tracking_ball = False
    if not shooters_data:
        target_tracking_coord = balls[0].x, balls[0].y
        # TODO game.get_ball_velocity() can return (None, None)
        if (
            game.get_ball_velocity() is not None
            and None not in game.get_ball_velocity()
        ):
            orientation = velocity_to_orientation(game.get_ball_velocity())
            tracking_ball = True
    else:
        # TODO (deploy more defenders, or find closest shooter?)
        sd = shooters_data[0]
        target_tracking_coord = sd.x, sd.y
        orientation = sd.orientation

    real_def_pos = friendly[defender1_id].x, friendly[defender1_id].y
    current_def_parametric = to_defense_parametric(real_def_pos, is_left=not is_yellow)
    target = align_defenders(
        current_def_parametric, target_tracking_coord, orientation, not is_yellow, env
    )
    cmd = go_to_point(
        pid_oren,
        pid_2d,
        friendly[defender1_id],
        defender1_id,
        target,
        face_ball(real_def_pos, (balls[0].x, balls[0].y)),
    )

    controller.add_robot_commands(cmd, defender1_id)

    controller.send_robot_commands()

    gp = get_goal_centre(is_left=not is_yellow)
    env.draw_line(
        [gp, (target_tracking_coord[0], target_tracking_coord[1])],
        width=5,
        color="RED" if tracking_ball else "PINK",
    )


def attack(
    pid_oren: PID,
    pid_2d: TwoDPID,
    game: Game,
    controller: RSimRobotController,
    shooter_id: int,
    defender_is_yellow: bool,
) -> bool:
    cmd = score_goal(
        game,
        controller.robot_has_ball(shooter_id),
        shooter_id=shooter_id,
        pid_oren=pid_oren,
        pid_trans=pid_2d,
        is_yellow=not defender_is_yellow,
        shoot_in_left_goal=not defender_is_yellow,
    )
    controller.add_robot_commands(cmd, shooter_id)
    controller.send_robot_commands()

    if game.is_ball_in_goal(not defender_is_yellow):
        logger.info("Goal Scored at Position: ", game.get_ball_pos())
        return True
    return False


def test_two_defenders(
    defender_id: int, shooter_id: int, defender_is_yellow: bool, headless: bool
):
    game = Game()

    if defender_is_yellow:
        N_ROBOTS_YELLOW = 3
        N_ROBOTS_BLUE = 6
    else:
        N_ROBOTS_BLUE = 3
        N_ROBOTS_YELLOW = 6

    env = SSLStandardEnv(
        n_robots_blue=N_ROBOTS_BLUE,
        n_robots_yellow=N_ROBOTS_YELLOW,
        render_mode="ansi" if headless else "human",
    )
    env.reset()

    env.teleport_ball(2.25, -1)

    # Move the other defender out of the way
    # not_defender_id = 2 if defender_id == 1 else 1
    # env.teleport_robot(defender_is_yellow, not_defender_id, 0, 0, 0)

    pid_oren_y, pid_2d_y = get_rsim_pids(N_ROBOTS_YELLOW)
    pid_oren_b, pid_2d_b = get_rsim_pids(N_ROBOTS_BLUE)
    sim_robot_controller_yellow, sim_robot_controller_blue, pvp_manager = setup_pvp(
        env, game, N_ROBOTS_BLUE, N_ROBOTS_YELLOW
    )

    if defender_is_yellow:
        sim_robot_controller_attacker, sim_robot_controller_defender = (
            sim_robot_controller_blue,
            sim_robot_controller_yellow,
        )
        pid_oren_a, pid_2d_a, pid_oren_d, pid_2d_d = (
            pid_oren_b,
            pid_2d_b,
            pid_oren_y,
            pid_2d_y,
        )
    else:
        sim_robot_controller_attacker, sim_robot_controller_defender = (
            sim_robot_controller_yellow,
            sim_robot_controller_blue,
        )
        pid_oren_a, pid_2d_a, pid_oren_d, pid_2d_d = (
            pid_oren_y,
            pid_2d_y,
            pid_oren_b,
            pid_2d_b,
        )

    any_scored = False
    attacker_gets_ball = False
    for _ in range(900):
        scored = attack(
            pid_oren_a,
            pid_2d_a,
            game,
            sim_robot_controller_attacker,
            shooter_id,
            defender_is_yellow,
        )
        if scored:
            any_scored = True
            break
        defend(
            pid_oren_d,
            pid_2d_d,
            game,
            sim_robot_controller_defender,
            defender_is_yellow,
            [defender_id, -1],
            env,
        )

        if sim_robot_controller_defender.robot_has_ball(
            defender_id
        ):  # Sim ends when the defender gets the ball
            break
        attacker_gets_ball = (
            attacker_gets_ball
            or sim_robot_controller_attacker.robot_has_ball(shooter_id)
        )

    assert not any_scored
    assert attacker_gets_ball


if __name__ == "__main__":
    try:
        test_two_defenders(1, 2, True, False)
    except KeyboardInterrupt:
        print("Exiting...")
