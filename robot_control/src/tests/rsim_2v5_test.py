import time
import numpy as np
import logging

from typing import Tuple, List

from motion_planning.src.pid.pid import get_rsim_pids
from robot_control.src.skills import (
    go_to_ball,
    go_to_point,
    goalkeep,
    empty_command,
    man_mark,
)
from robot_control.src.tests.utils import setup_pvp
from robot_control.src.utils.pass_quality_utils import (
    find_best_receiver_position,
    find_pass_quality,
)
from robot_control.src.utils.shooting_utils import find_shot_quality
from rsoccer_simulator.src.ssl.envs.standard_ssl import SSLStandardEnv
from entities.game import Game
from robot_control.src.intent import PassBall, defend, score_goal
from global_utils.math_utils import distance


logger = logging.getLogger(__name__)

MAX_TIME = 20  # in seconds
N_ROBOTS = 7
DEFENDING_ROBOTS = 5
ATTACKING_ROBOTS = 2
# TARGET_COORDS = (-2, 3)
PASS_QUALITY_THRESHOLD = 1.15
SHOT_QUALITY_THRESHOLD = 0.5

BALL_V0_MAGNITUDE = 3
BALL_A_MAGNITUDE = -0.3


def intercept_ball(
    receiver_pos: Tuple[float, float],
    ball_pos: Tuple[float, float],
    ball_vel: Tuple[float, float],
    robot_speed: float,
) -> Tuple[float, float]:
    """
    Simple function to calculate intercept position for a robot.
    Assumes the ball is moving in a straight line, and we find the point where the receiver should go.
    """
    # Calculate the time it will take for the robot to reach the ball (simplified)
    distance_to_ball = np.linalg.norm(np.array(ball_pos) - np.array(receiver_pos))
    time_to_reach = (
        distance_to_ball / robot_speed if robot_speed != 0 else float("inf")
    )  # Assuming constant robot speed

    # Predict the future position of the ball
    intercept_pos = (
        (
            ball_pos[0] + ball_vel[0] * time_to_reach,
            ball_pos[1] + ball_vel[1] * time_to_reach,
        )
        if ball_vel != None
        else None
    )

    return intercept_pos


def test_2v5(friendly_robot_ids: List[int], attacker_is_yellow: bool, headless: bool):
    game = Game()

    N_ROBOTS_ATTACK = 2
    N_ROBOTS_DEFEND = 5

    N_ROBOTS_YELLOW = N_ROBOTS_ATTACK if attacker_is_yellow else N_ROBOTS_DEFEND
    N_ROBOTS_BLUE = N_ROBOTS_DEFEND if attacker_is_yellow else N_ROBOTS_ATTACK

    env = SSLStandardEnv(
        n_robots_blue=N_ROBOTS_BLUE,
        n_robots_yellow=N_ROBOTS_YELLOW,
        render_mode="ansi" if headless else "human",
    )
    env.reset()

    env.teleport_ball(1, 1)

    sim_robot_controller_yellow, sim_robot_controller_blue, pvp_manager = setup_pvp(
        env, game, N_ROBOTS_BLUE, N_ROBOTS_YELLOW
    )

    if attacker_is_yellow:
        sim_robot_controller_attacker = sim_robot_controller_yellow
        sim_robot_controller_defender = sim_robot_controller_blue
    else:
        sim_robot_controller_attacker = sim_robot_controller_blue
        sim_robot_controller_defender = sim_robot_controller_yellow

    pid_oren_attacker, pid_2d_attacker = get_rsim_pids(N_ROBOTS_ATTACK)
    pid_oren_defender, pid_2d_defender = get_rsim_pids(N_ROBOTS_DEFEND)

    player1_id = friendly_robot_ids[0]  # Start with robot 0
    player2_id = friendly_robot_ids[1]  # Start with robot 1

    pass_task = None
    goal_scored = False

    for iter in range(2000):
        if iter % 100 == 0:
            print(iter)

        sim_robot_controller_defender.add_robot_commands(
            defend(
                pid_oren_defender,
                pid_2d_defender,
                game,
                not attacker_is_yellow,
                1,
                env,
            ),
            1,
        )
        """
        sim_robot_controller_defender.add_robot_commands(
            defend(
                pid_oren_defender, pid_2d_defender, game, not attacker_is_yellow, 4, env
            ),
            4,
        )
        """

        sim_robot_controller_defender.add_robot_commands(
            goalkeep(
                attacker_is_yellow,
                game,
                0,
                pid_oren_defender,
                pid_2d_defender,
                not attacker_is_yellow,
                sim_robot_controller_defender.robot_has_ball(0),
            ),
            0,
        )

        """
        sim_robot_controller_defender.add_robot_commands(
            man_mark(
                not attacker_is_yellow,
                game,
                2,
                0,
                pid_oren_defender,
                pid_2d_defender,
            ),
            2,
        )
    
        sim_robot_controller_defender.add_robot_commands(
            man_mark(
                not attacker_is_yellow,
                game,
                3,
                0,
                pid_oren_defender,
                pid_2d_defender,
            ),
            3,
        )
        """

        sim_robot_controller_defender.send_robot_commands()

        if iter > 10:  # give them chance to spawn in the correct place
            goal_scored = goal_scored or game.is_ball_in_goal(not attacker_is_yellow)
            if game.is_ball_in_goal(not attacker_is_yellow):
                break

            commands = {}
            pass_task = None
            trying_to_pass = False
            sampled_positions = None
            target_pos = None

            target_goal_line = game.field.enemy_goal_line(attacker_is_yellow)
            latest_frame = game.get_my_latest_frame(attacker_is_yellow)

            # if not latest_frame:
            #    return

            friendly_robots, enemy_robots, balls = latest_frame

            enemy_velocities = game.get_robots_velocity(attacker_is_yellow) or [
                (0.0, 0.0)
            ] * len(enemy_robots)
            enemy_speeds = np.linalg.norm(enemy_velocities, axis=1)

            # TODO: Not sure if this is sufficient for both blue and yellow scoring
            # It won't be because note that in real life the blue team is not necessarily
            # on the left of the pitch
            goal_x = target_goal_line.coords[0][0]
            goal_y1 = target_goal_line.coords[1][1]
            goal_y2 = target_goal_line.coords[0][1]

            # TODO: For now we just look at the first ball, but this will eventually have to be smarter
            ball_data = balls[0]
            ball_vel = game.get_ball_velocity()

            player1_has_ball = sim_robot_controller_attacker.robot_has_ball(player1_id)
            player2_has_ball = sim_robot_controller_attacker.robot_has_ball(player2_id)
            ball_possessor_id = None

            if player1_has_ball:
                ball_possessor_id = player1_id
            elif player2_has_ball:
                ball_possessor_id = player2_id

            ### CASE 1: No one has the ball - Try to intercept it ###
            if ball_possessor_id is None and trying_to_pass == False:
                print("No one has the ball, trying to intercept")
                best_interceptor = None
                best_intercept_score = float(
                    "inf"
                )  # Lower is better (closer to ball path)
                for rid in friendly_robot_ids:
                    ball_pos = ball_data.x, ball_data.y
                    robot = friendly_robots[rid]
                    # this is a bit hacky for now. we need a better interception function
                    # Calculate intercept position using the intercept_ball function
                    intercept_pos = intercept_ball(
                        (robot.x, robot.y), ball_pos, ball_vel, robot_speed=4.0
                    )  # Use appropriate robot speed

                    # Calculate how close the robot is to the intercept position (lower score is better)
                    intercept_score = (
                        distance(robot, intercept_pos)
                        if intercept_pos != None
                        else float("inf")
                    )

                    if intercept_score < best_intercept_score:
                        best_interceptor = rid
                        best_intercept_score = intercept_score

                # Send the best robot to intercept
                if best_interceptor is not None:
                    ball_vel = game.get_ball_velocity()
                    intercept_pos = intercept_ball(
                        (
                            friendly_robots[best_interceptor].x,
                            friendly_robots[best_interceptor].y,
                        ),
                        ball_pos,
                        ball_vel,
                        robot_speed=4.0,
                    )
                    commands[best_interceptor] = (
                        go_to_point(
                            pid_oren_attacker,
                            pid_2d_attacker,
                            friendly_robots[best_interceptor],
                            best_interceptor,
                            intercept_pos,
                            friendly_robots[best_interceptor].orientation,
                        )
                        if intercept_pos != None
                        else empty_command(dribbler_on=True)
                    )

                for rid in friendly_robot_ids:
                    if rid == best_interceptor or best_interceptor == None:
                        continue
                    # If a pass is happening, don't override the receiver's movement
                    if trying_to_pass:
                        continue  # Let PassBall handle the receiver

                    potential_passer_id = (
                        rid + 1 if rid + 1 <= len(friendly_robots) - 1 else rid - 1
                    )
                    target_pos, sampled_positions, _ = find_best_receiver_position(
                        friendly_robots[rid],
                        friendly_robots[
                            potential_passer_id
                        ],  # PointOnField(ball_pos[0], ball_pos[1]),
                        enemy_robots,
                        enemy_speeds,
                        BALL_V0_MAGNITUDE,
                        BALL_A_MAGNITUDE,
                        goal_x,
                        goal_y1,
                        goal_y2,
                        attacker_is_yellow,
                    )

                    commands[rid] = go_to_ball(
                        pid_oren_attacker,
                        pid_2d_attacker,
                        friendly_robots[rid],
                        rid,
                        target_pos,
                        # friendly_robots[rid].orientation,
                    )

                # return commands, sampled_positions, target_pos

                sim_robot_controller_attacker.add_robot_commands(commands)
                print("intercepting")
                sim_robot_controller_attacker.send_robot_commands()

                if sampled_positions != None:
                    for sample in sampled_positions:
                        if sample == target_pos:
                            env.draw_point(sample.x, sample.y, "YELLOW", width=4)
                        else:
                            env.draw_point(sample.x, sample.y, width=2)
                continue

            ### CASE 2: Someone has the ball ###
            print("We have the ball", ball_possessor_id)
            possessor_data = friendly_robots[ball_possessor_id]

            # Check shot opportunity
            shot_quality = find_shot_quality(
                possessor_data,
                enemy_robots,
                goal_x,
                goal_y1,
                goal_y2,
                attacker_is_yellow,
            )
            if shot_quality > SHOT_QUALITY_THRESHOLD:
                print("shooting with chance", shot_quality, SHOT_QUALITY_THRESHOLD)
                commands[ball_possessor_id] = score_goal(
                    game,
                    True,
                    ball_possessor_id,
                    pid_oren_attacker,
                    pid_2d_attacker,
                    attacker_is_yellow,
                    attacker_is_yellow,
                )
                # return commands, None, None  # Just shoot, no need to pass
                sim_robot_controller_attacker.add_robot_commands(commands)
                sim_robot_controller_attacker.send_robot_commands()

                if sampled_positions != None:
                    for sample in sampled_positions:
                        if sample == target_pos:
                            env.draw_point(sample.x, sample.y, "BLUE", width=2)
                        else:
                            env.draw_point(sample.x, sample.y, width=2)
                continue

            # Check for best pass
            best_receiver_id = None
            best_pass_quality = 0

            for rid in friendly_robot_ids:
                if rid == ball_possessor_id:
                    continue
                pq = find_pass_quality(
                    friendly_robots[ball_possessor_id],
                    friendly_robots[rid],
                    enemy_robots,
                    enemy_speeds,
                    BALL_V0_MAGNITUDE,
                    BALL_A_MAGNITUDE,
                    goal_x,
                    goal_y1,
                    goal_y2,
                    attacker_is_yellow,
                )
                if pq > best_pass_quality:
                    best_pass_quality = pq
                    best_receiver_id = rid
            if (
                best_receiver_id is not None
                and best_pass_quality > PASS_QUALITY_THRESHOLD
            ):
                print(
                    "trying to execute a pass with quality ",
                    best_pass_quality,
                    PASS_QUALITY_THRESHOLD,
                )
                trying_to_pass = True
                pass_task = PassBall(
                    pid_oren_attacker,
                    pid_2d_attacker,
                    game,
                    ball_possessor_id,
                    best_receiver_id,
                    (
                        friendly_robots[best_receiver_id].x,
                        friendly_robots[best_receiver_id].y,
                    ),
                )
                pass_commands = pass_task.enact(passer_has_ball=True)
                commands[ball_possessor_id] = pass_commands[0]
                commands[best_receiver_id] = pass_commands[1]
            else:
                commands[ball_possessor_id] = empty_command(
                    dribbler_on=True
                )  # Wait for a better pass

            # Move non-possessing robots to good positions
            for rid in friendly_robot_ids:
                if rid == ball_possessor_id:
                    continue
                # If a pass is happening, don't override the receiver's movement
                if trying_to_pass:
                    continue  # Let PassBall handle the receiver
                target_pos, sampled_positions, _ = find_best_receiver_position(
                    friendly_robots[rid],
                    friendly_robots[ball_possessor_id],
                    enemy_robots,
                    enemy_speeds,
                    BALL_V0_MAGNITUDE,
                    BALL_A_MAGNITUDE,
                    goal_x,
                    goal_y1,
                    goal_y2,
                    attacker_is_yellow,
                )

                commands[rid] = go_to_point(
                    pid_oren_attacker,
                    pid_2d_attacker,
                    friendly_robots[rid],
                    rid,
                    (target_pos.x, target_pos.y),
                    friendly_robots[rid].orientation,
                )

            # sim_robot_controller_attacker.send_robot_commands()
            sim_robot_controller_attacker.add_robot_commands(commands)
            sim_robot_controller_attacker.send_robot_commands()

            if sampled_positions != None:
                for sample in sampled_positions:
                    if sample == target_pos:
                        env.draw_point(sample.x, sample.y, "YELLOW", width=4)
                    else:
                        env.draw_point(sample.x, sample.y, width=2)
    assert goal_scored


if __name__ == "__main__":
    try:
        test_2v5([0, 1], True, False)
    except KeyboardInterrupt:
        print("Exiting...")
