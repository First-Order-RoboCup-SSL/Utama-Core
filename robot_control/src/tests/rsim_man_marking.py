import time
from entities.data.vision import BallData, FrameData, RobotData
from motion_planning.src.pid.pid import get_rsim_pids
from robot_control.src.skills import face_ball, find_likely_enemy_shooter, go_to_point
from robot_control.src.tests.utils import setup_pvp
from team_controller.src.controllers import RSimRobotController
from rsoccer_simulator.src.ssl.envs.standard_ssl import SSLStandardEnv
from entities.game import Game
from robot_control.src.intent import score_goal
from motion_planning.src.pid import PID
from config.settings import TIMESTEP
import logging
import math
import random

from team_controller.src.controllers.sim.rsim_robot_controller import (
    PVPManager,
    PVPManager2,
)

logger = logging.getLogger(__name__)

ITERS = 5000
N_ROBOTS = 6

blue_robot_data = [RobotData(i, i, 10 * i, 0) for i in range(6)]
yellow_robot_data = [RobotData(i, 100 * i, 1000 * i, 0) for i in range(6)]
basic_ball = BallData(0, 0, 0, 0)


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


def man_mark(game, robot, target, ball_pos, pid_oren, pid_trans):
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
        game,
        pid_oren,
        pid_trans,
        # robot,
        0,
        (target_x, target_y),
        face_ball((robot.x, robot.y), ball_pos),
    )
    return cmd


def block_goal_and_attacker(game, robot_id, attacker, goal_pos, pid_oren, pid_trans):
    # Position between the goal and the attacker
    target_x = (attacker.x + goal_pos[0]) / 2
    target_y = (attacker.y + goal_pos[1]) / 2

    robot = game.friendly_robots[robot_id]

    cmd = go_to_point(
        game,
        pid_oren,
        pid_trans,
        robot_id,
        (target_x, target_y),
        face_ball((robot.x, robot.y), (attacker.x, attacker.y)),
    )
    return cmd


def block_pass_between_attackers(
    game, robot_id, main_attacker_id, support_attacker_id, pid_oren, pid_trans
):
    # Position between the main attacker and the support attacker
    enemy = game.enemy_robots
    target_x = (enemy[main_attacker_id].x + enemy[support_attacker_id].x) / 2
    target_y = (enemy[main_attacker_id].y + enemy[support_attacker_id].y) / 2

    robot = game.friendly_robots[robot_id]

    cmd = go_to_point(
        game,
        pid_oren,
        pid_trans,
        robot_id,
        (target_x, target_y),
        face_ball(
            (robot.x, robot.y), (enemy[main_attacker_id].x, enemy[main_attacker_id].y)
        ),
    )
    return cmd


def test_man_marking(shooter_id: int, defender_is_yellow: bool, headless: bool):
    """
    Assumptions:
    1. Colour, side and role
        - Yellow team is on the right and attacking left goal.
        - Blue team is on the left and attacking right goal.
        - Yellow team is purely attacking while blue team is purely defending.
        - TODO: consider switching colour / side in real match

    2. Full Field Game
        - TODO: Half field implementation

    3. Role
        - Attacker team: 1 keeper, 1 main attacker, 1 support attacker
        - Defender team: 1 keeper, 1 main defender, 1 piggy (man marking on support attacker)
        - The role / messaging system will dynamically adjust the role.
        - TODO: integrate role system once available
    """

    yellow_game = Game(
        my_team_is_yellow=True,
        my_team_is_right=True,
        start_frame=FrameData(0, yellow_robot_data, blue_robot_data, [basic_ball]),
    )
    blue_game = Game(
        my_team_is_yellow=False,
        my_team_is_right=False,
        start_frame=FrameData(0, blue_robot_data, yellow_robot_data, [basic_ball]),
    )

    N_ROBOTS_YELLOW = 3
    N_ROBOTS_BLUE = 3
    VISUALIZE = True
    # yellow_goal = yellow_game.field.my_goal_target.xy
    # blue_goal = blue_game.field.my_goal_target

    yellow_goal, blue_goal = (-4.5, 0), (4.5, 0)
    # Goal protected by yellow team and blue team

    env = SSLStandardEnv(
        n_robots_blue=N_ROBOTS_BLUE,
        n_robots_yellow=N_ROBOTS_YELLOW,
        render_mode="ansi" if headless else "human",
    )
    env.reset()

    env.teleport_ball(random.random(), random.random())

    pid_oren_attacker, pid_2d_attacker = get_rsim_pids()
    pid_oren_defender, pid_2d_defender = get_rsim_pids()

    pvp_manager = PVPManager2(
        env, N_ROBOTS_BLUE, N_ROBOTS_YELLOW, yellow_game, blue_game
    )
    sim_robot_controller_attacker = RSimRobotController(
        is_team_yellow=True, env=env, game_obj=yellow_game, pvp_manager=pvp_manager
    )
    sim_robot_controller_defender = RSimRobotController(
        is_team_yellow=False, env=env, game_obj=blue_game, pvp_manager=pvp_manager
    )
    pvp_manager.set_yellow_controller(sim_robot_controller_attacker)
    pvp_manager.set_blue_controller(sim_robot_controller_defender)
    pvp_manager.reset_env()
    # sim_robot_controller_defender = RSimRobotController(
    #     is_team_yellow=True, env=env, game_obj=yellow_game
    # )
    # sim_robot_controller_attacker = RSimRobotController(
    #     is_team_yellow=False, env=env, game_obj=blue_game
    # )
    attacker_game, defender_game = yellow_game, blue_game

    if defender_is_yellow:
        sim_robot_controller_attacker, sim_robot_controller_defender = (
            sim_robot_controller_defender,
            sim_robot_controller_attacker,
        )
        attacker_game, defender_game = blue_game, yellow_game

    # ROLE DEFINITION
    goalie_attacker_id = 0
    main_attacker_id = 1
    support_attacker_id = 2

    support_offset_x = random.random()
    support_offset_y = random.random()

    goalie_defender_id = 0
    main_defender_id = 1
    support_defender_id = 2

    goal_scored = False

    for iter in range(ITERS):
        if not goal_scored:
            friendly = attacker_game.friendly_robots
            ball = attacker_game.ball

            # Attacker team: Keeper stays near the goal
            cmd = go_to_point(
                game=attacker_game,
                pid_oren=pid_oren_attacker,
                pid_trans=pid_2d_attacker,
                robot_id=goalie_attacker_id,
                target_coords=yellow_goal,
                target_oren=face_ball(
                    (friendly[goalie_attacker_id].x, friendly[0].y),
                    (attacker_game.ball.x, attacker_game.ball.y),
                ),
            )
            sim_robot_controller_attacker.add_robot_commands(cmd, goalie_attacker_id)

            # Attacker team: Main attacker tries to score or pass
            if sim_robot_controller_attacker.robot_has_ball(main_attacker_id):
                # If Main attacker has the ball, he will randomly decide to shoot or pass
                if random.random() < 0.5:
                    cmd = score_goal(
                        game_obj=attacker_game,
                        shooter_has_ball=True,
                        shooter_id=main_attacker_id,
                        pid_oren=pid_oren_attacker,
                        pid_trans=pid_2d_attacker,
                        is_yellow=not defender_is_yellow,
                    )
                else:  # Pass to support attacker
                    cmd = go_to_point(
                        attacker_game,
                        pid_oren_attacker,
                        pid_2d_attacker,
                        main_attacker_id,
                        (
                            friendly[support_attacker_id].x,
                            friendly[support_attacker_id].y,
                        ),  # Pass to support attacker (ID 2)
                        face_ball(
                            (
                                friendly[main_attacker_id].x,
                                friendly[main_attacker_id].y,
                            ),
                            (attacker_game.ball.x, attacker_game.ball.y),
                        ),
                    )
            else:
                # Main attacker moves toward the ball to gain possession
                # print(ball)
                cmd = go_to_point(
                    attacker_game,
                    pid_oren_attacker,
                    pid_2d_attacker,
                    main_attacker_id,
                    (ball.x, ball.y),
                    face_ball(
                        (friendly[main_attacker_id].x, friendly[main_attacker_id].y),
                        (attacker_game.ball.x, attacker_game.ball.y),
                    ),
                )
            sim_robot_controller_attacker.add_robot_commands(cmd, main_attacker_id)

            # Attacker team: Support attacker positions to receive a pass
            if not sim_robot_controller_attacker.robot_has_ball(support_attacker_id):
                # If support attacker does not have the ball
                # Move to a position ahead of the main attacker
                support_target = (
                    friendly[main_attacker_id].x + support_offset_x,
                    friendly[main_attacker_id].y + support_offset_y,
                )
                cmd = go_to_point(
                    attacker_game,
                    pid_oren_attacker,
                    pid_2d_attacker,
                    support_attacker_id,
                    support_target,
                    face_ball(
                        (
                            friendly[support_attacker_id].x,
                            friendly[support_attacker_id].y,
                        ),
                        (attacker_game.ball.x, attacker_game.ball.y),
                    ),
                )
                sim_robot_controller_attacker.add_robot_commands(
                    cmd, support_attacker_id
                )
            sim_robot_controller_attacker.send_robot_commands()

            # Check if a goal is scored
            if attacker_game.is_ball_in_goal(right_goal=True):
                logger.info("Goal Scored at Position: ", attacker_game.ball)
                goal_scored = True

            # Defender team: Goalie stays near the goal
            friendly = defender_game.friendly_robots
            ball = defender_game.ball

            cmd = go_to_point(
                defender_game,
                pid_oren_defender,
                pid_2d_defender,
                goalie_defender_id,
                blue_goal,
                face_ball(
                    (friendly[goalie_defender_id].x, friendly[goalie_defender_id].y),
                    (ball.x, ball.y),
                ),
            )
            if not find_likely_enemy_shooter(defender_game.enemy_robots, [ball]):
                cmd = go_to_point(
                    attacker_game,
                    pid_oren_defender,
                    pid_2d_defender,
                    goalie_defender_id,
                    blue_goal,
                    face_ball(
                        (
                            defender_game.enemy_robots[main_attacker_id].x,
                            defender_game.enemy_robots[main_attacker_id].y,
                        ),
                        (ball.x, ball.y),
                    ),
                )
            sim_robot_controller_defender.add_robot_commands(cmd, goalie_defender_id)

            # Defender team: Main defender focuses on intercepting the main attacker
            cmd = block_goal_and_attacker(
                defender_game,
                # friendly[main_defender_id],
                main_defender_id,
                defender_game.enemy_robots[main_attacker_id],
                blue_goal,
                pid_oren_defender,
                pid_2d_defender,
            )
            sim_robot_controller_defender.add_robot_commands(cmd, main_defender_id)

            # Defender team: Support defender marks the support attacker
            cmd = block_pass_between_attackers(
                defender_game,
                support_defender_id,
                main_attacker_id,
                support_attacker_id,
                pid_oren_defender,
                pid_2d_defender,
            )
            sim_robot_controller_defender.add_robot_commands(cmd, support_defender_id)
            sim_robot_controller_defender.send_robot_commands()

            if VISUALIZE:
                # Visualize main attacker target position and orientation
                env.draw_line(
                    [
                        (
                            attacker_game.friendly_robots[main_attacker_id].x,
                            attacker_game.friendly_robots[main_attacker_id].y,
                        ),
                        yellow_goal,
                    ],
                    width=4,
                    color="RED",
                )
                # Visualize support attacker target position and orientation
                env.draw_line(
                    [
                        (
                            attacker_game.friendly_robots[support_attacker_id].x,
                            attacker_game.friendly_robots[support_attacker_id].y,
                        ),
                        support_target,
                    ],
                    width=4,
                    color="RED",
                )

                # Visualize main defender target position and orientation
                env.draw_line(
                    [
                        (
                            defender_game.friendly_robots[main_defender_id].x,
                            defender_game.friendly_robots[main_defender_id].y,
                        ),
                        (
                            attacker_game.friendly_robots[main_attacker_id].x,
                            attacker_game.friendly_robots[main_attacker_id].y,
                        ),
                    ],
                    width=4,
                    color="BLUE",
                )
                # Visualize support defender target position and orientation
                env.draw_line(
                    [
                        (
                            defender_game.friendly_robots[support_defender_id].x,
                            defender_game.friendly_robots[support_defender_id].y,
                        ),
                        (
                            attacker_game.friendly_robots[support_attacker_id].x,
                            attacker_game.friendly_robots[support_attacker_id].y,
                        ),
                    ],
                    width=4,
                    color="BLUE",
                )
        # print(time.time() - start)

    assert not goal_scored


if __name__ == "__main__":
    try:
        test_man_marking(5, True, False)
    except KeyboardInterrupt:
        print("Exiting...")
