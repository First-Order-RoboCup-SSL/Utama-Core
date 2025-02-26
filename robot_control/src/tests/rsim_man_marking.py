from motion_planning.src.pid.pid import get_rsim_pids
from robot_control.src.skills import face_ball, find_likely_enemy_shooter, go_to_point
from robot_control.src.tests.utils import setup_pvp
from team_controller.src.controllers import RSimRobotController
from rsoccer_simulator.src.ssl.envs.standard_ssl import SSLStandardEnv
from entities.game import Game
from robot_control.src.intent import score_goal
from motion_planning.src.pid import PID
from team_controller.src.config.settings import TIMESTEP
import logging
import math
import random

logger = logging.getLogger(__name__)

ITERS = 500
N_ROBOTS = 6


def calculate_pass_viability_score(opponent, ball_pos, goal_pos):
    # Ignore robots close to the opponent defense area, the ball owner, and robots behind the ball
    if opponent.x > ball_pos[0] or abs(opponent.y - ball_pos[1]) < 0.5:
        return 0

    # Calculate distance to the ball and distance to our goal
    distance_to_ball = math.sqrt(
        (opponent.x - ball_pos[0]) ** 2 + (opponent.y - ball_pos[1]) ** 2
    )
    distance_to_goal = math.sqrt(
        (opponent.x - goal_pos[0]) ** 2 + (opponent.y - goal_pos[1]) ** 2
    )

    # Pass viability score
    score = (1 / distance_to_ball) + (1 / distance_to_goal)
    return score


def assign_piggies(friendly_robots, enemy_robots, ball_pos, goal_pos):
    piggies = []
    for i, enemy in enumerate(enemy_robots):
        score = calculate_pass_viability_score(enemy, ball_pos, goal_pos)
        if score > 0.5:  # Threshold value
            piggies.append((i, enemy))

    # Assign piggies to friendly robots
    assignments = {}
    for i, piggy in enumerate(piggies):
        if i < len(friendly_robots):
            assignments[friendly_robots[i].id] = piggy[1]
    return assignments


def man_mark(robot, target, ball_pos, pid_oren, pid_trans):
    # Position with a perpendicular offset to the line between target and ball
    dx = target.x - ball_pos[0]
    dy = target.y - ball_pos[1]
    norm = math.sqrt(dx**2 + dy**2)
    dx /= norm
    dy /= norm

    # Perpendicular offset
    offset_x = -dy * 0.2
    offset_y = dx * 0.2

    target_x = target.x + offset_x
    target_y = target.y + offset_y

    cmd = go_to_point(
        pid_oren,
        pid_trans,
        robot,
        0,
        (target_x, target_y),
        face_ball((robot.x, robot.y), ball_pos),
    )
    return cmd


def block_goal_and_attacker(robot, attacker, goal_pos, pid_oren, pid_trans):
    # Position between the goal and the attacker
    target_x = (attacker.x + goal_pos[0]) / 2
    target_y = (attacker.y + goal_pos[1]) / 2

    cmd = go_to_point(
        pid_oren,
        pid_trans,
        robot,
        0,
        (target_x, target_y),
        face_ball((robot.x, robot.y), (attacker.x, attacker.y)),
    )
    return cmd


def block_pass_between_attackers(
    robot, main_attacker, support_attacker, pid_oren, pid_trans
):
    # Position between the main attacker and the support attacker
    target_x = (main_attacker.x + support_attacker.x) / 2
    target_y = (main_attacker.y + support_attacker.y) / 2

    cmd = go_to_point(
        pid_oren,
        pid_trans,
        robot,
        0,
        (target_x, target_y),
        face_ball((robot.x, robot.y), (main_attacker.x, main_attacker.y)),
    )
    return cmd


def test_man_marking(shooter_id: int, defender_is_yellow: bool, headless: bool):
    game = Game()

    if defender_is_yellow:
        N_ROBOTS_YELLOW = 3  # Our team: 1 keeper, 2 defenders
        N_ROBOTS_BLUE = (
            3  # Opponent team: 1 keeper, 1 main attacker, 1 support attacker
        )
    else:
        N_ROBOTS_BLUE = 3  # Our team: 1 keeper, 2 defenders
        N_ROBOTS_YELLOW = (
            3  # Opponent team: 1 keeper, 1 main attacker, 1 support attacker
        )

    env = SSLStandardEnv(
        n_robots_blue=N_ROBOTS_BLUE,
        n_robots_yellow=N_ROBOTS_YELLOW,
        render_mode="ansi" if headless else "human",
    )
    env.reset()

    import random

    env.teleport_ball(random.random(), random.random())

    pid_oren_y, pid_2d_y = get_rsim_pids()
    pid_oren_b, pid_2d_b = get_rsim_pids()

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

    shoot_in_left_goal = random.random() > 0.5
    goal_scored = False

    for iter in range(ITERS):
        if not goal_scored:
            friendly, enemy, balls = game.get_my_latest_frame(
                my_team_is_yellow=defender_is_yellow
            )

            # Opponent team: Keeper stays near the goal
            keeper_target = (-4.5, 0) if defender_is_yellow else (4.5, 0)
            cmd = go_to_point(
                pid_oren_a,
                pid_2d_a,
                enemy[0],  # Opponent keeper (ID 0)
                0,
                keeper_target,
                face_ball((enemy[0].x, enemy[0].y), game.get_ball_pos()[0]),
            )
            sim_robot_controller_attacker.add_robot_commands(cmd, 0)
            # sim_robot_controller_attacker.send_robot_commands()

            # Visualize keeper target position and orientation
            env.draw_line(
                [(enemy[0].x, enemy[0].y), keeper_target], width=2, color="BLUE"
            )
            env.draw_point(keeper_target[0], keeper_target[1], color="BLUE")

            # Opponent team: Main attacker tries to score or pass
            if sim_robot_controller_attacker.robot_has_ball(
                1
            ):  # Main attacker (ID 1) has the ball
                # Randomly decide to shoot or pass
                if random.random() < 0.5:  # 50% chance to shoot
                    cmd = score_goal(
                        game,
                        True,
                        shooter_id=1,
                        pid_oren=pid_oren_a,
                        pid_trans=pid_2d_a,
                        is_yellow=not defender_is_yellow,
                        shoot_in_left_goal=shoot_in_left_goal,
                    )
                else:  # Pass to support attacker
                    cmd = go_to_point(
                        pid_oren_a,
                        pid_2d_a,
                        enemy[1],  # Main attacker (ID 1)
                        0,
                        (enemy[2].x, enemy[2].y),  # Pass to support attacker (ID 2)
                        face_ball((enemy[1].x, enemy[1].y), game.get_ball_pos()[0]),
                    )
            else:
                # Main attacker moves toward the ball to gain possession
                cmd = go_to_point(
                    pid_oren_a,
                    pid_2d_a,
                    enemy[1],  # Main attacker (ID 1)
                    0,
                    game.get_ball_pos()[0],
                    face_ball((enemy[1].x, enemy[1].y), game.get_ball_pos()[0]),
                )
            sim_robot_controller_attacker.add_robot_commands(cmd, 1)

            # Opponent team: Support attacker positions to receive a pass
            if not sim_robot_controller_attacker.robot_has_ball(
                2
            ):  # Support attacker (ID 2) does not have the ball
                # Move to a position ahead of the main attacker
                support_target = (
                    enemy[1].x + 0.5,  # Slightly ahead of the main attacker
                    enemy[1].y + 0.5,  # Offset to the side
                )
                cmd = go_to_point(
                    pid_oren_a,
                    pid_2d_a,
                    enemy[2],  # Support attacker (ID 2)
                    0,
                    support_target,
                    face_ball((enemy[2].x, enemy[2].y), game.get_ball_pos()[0]),
                )
                sim_robot_controller_attacker.add_robot_commands(cmd, 2)
            sim_robot_controller_attacker.send_robot_commands()

            # Check if a goal is scored
            if game.is_ball_in_goal(shoot_in_left_goal):
                logger.info("Goal Scored at Position: ", game.get_ball_pos())
                goal_scored = True

            # Our team: Keeper stays near the goal
            keeper_id = 0
            keeper_target = (4.5, 0) if defender_is_yellow else (-4.5, 0)
            cmd = go_to_point(
                pid_oren_d,
                pid_2d_d,
                friendly[keeper_id],
                0,
                keeper_target,
                face_ball(
                    (friendly[keeper_id].x, friendly[keeper_id].y),
                    game.get_ball_pos()[0],
                ),
            )
            if not find_likely_enemy_shooter(enemy, balls):
                cmd = go_to_point(
                    pid_oren_d,
                    pid_2d_d,
                    friendly[0],
                    0,
                    keeper_target,
                    face_ball((friendly[0].x, friendly[0].y), game.get_ball_pos()[0]),
                )
                sim_robot_controller_defender.add_robot_commands(cmd, keeper_id)

            # Our team: Main defender focuses on intercepting the main attacker
            main_defender_id = 1
            main_attacker = enemy[1]  # Opponent's main attacker (ID 1)
            goal_pos = (4.5, 0) if defender_is_yellow else (-4.5, 0)
            cmd = block_goal_and_attacker(
                friendly[main_defender_id],
                main_attacker,
                goal_pos,
                pid_oren_d,
                pid_2d_d,
            )
            sim_robot_controller_defender.add_robot_commands(cmd, main_defender_id)

            # Our team: Support defender marks the support attacker
            support_defender_id = 2
            support_attacker = enemy[2]  # Opponent's support attacker (ID 2)
            cmd = block_pass_between_attackers(
                friendly[support_defender_id],
                main_attacker,
                support_attacker,
                pid_oren_d,
                pid_2d_d,
            )
            sim_robot_controller_defender.add_robot_commands(cmd, support_defender_id)
            sim_robot_controller_defender.send_robot_commands()

            # Visualize main attacker target position and orientation
            env.draw_line(
                [(enemy[1].x, enemy[1].y), keeper_target],
                width=4,
                color="RED",
            )
            # Visualize support attacker target position and orientation
            env.draw_line(
                [(enemy[2].x, enemy[2].y), support_target], width=4, color="RED"
            )

            # Visualize main defender target position and orientation
            env.draw_line(
                [
                    (friendly[main_defender_id].x, friendly[main_defender_id].y),
                    (main_attacker.x, main_attacker.y),
                ],
                width=4,
                color="BLUE",
            )
            # Visualize support defender target position and orientation
            env.draw_line(
                [
                    (friendly[support_defender_id].x, friendly[support_defender_id].y),
                    (support_attacker.x, support_attacker.y),
                ],
                width=4,
                color="BLUE",
            )

    assert not goal_scored


if __name__ == "__main__":
    try:
        test_man_marking(5, True, False)
    except KeyboardInterrupt:
        print("Exiting...")
