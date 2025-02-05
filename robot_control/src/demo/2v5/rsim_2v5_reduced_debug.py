import time
from motion_planning.src.pid.pid import get_rsim_pids
from team_controller.src.controllers import RSimRobotController
from rsoccer_simulator.src.ssl.envs.standard_ssl import SSLStandardEnv
from entities.game import Game
from global_utils.math_utils import distance
from robot_control.src.intent import score_goal, PassBall
from robot_control.src.skills import go_to_ball, turn_on_spot, go_to_point
import logging
import numpy as np

logger = logging.getLogger(__name__)

MAX_TIME = 20  # in seconds
N_YELLOW = 2
N_BLUE = 5

r0 = 0
r1 = 1

receiving_points = [(-2.3, -1.5), (-3, 2)]
passing_points = [(-1.2, 1), (-1.5, -2)]


def test_rsim_2v5():
    """When the tests are run with pytest, these parameters are filled in
    based on whether we are in full or quick test mode (see conftest.py)"""

    game = Game()

    # Shooting team gets full complement of robots, defending team only half
    env = SSLStandardEnv(
        n_robots_blue=N_BLUE,
        n_robots_yellow=N_YELLOW,
    )

    env.reset()

    ### Teleport blue robots
    env.teleport_robot(False, 1, -3, 0.5)
    env.teleport_robot(False, 2, -2.5, -0.5)
    env.teleport_robot(False, 3, -2, 0.5)
    env.teleport_robot(False, 4, -1.5, -0.5)

    ### Teleport yellow robots
    env.teleport_robot(True, 0, -0.5, 1)
    env.teleport_robot(True, 1, -1, -1.5)

    env.teleport_ball(-0.5, 0)

    pid_oren, pid_trans = get_rsim_pids(N_YELLOW)

    sim_robot_controller = RSimRobotController(
        is_team_yellow=True, env=env, game_obj=game
    )
    target_ball_pos = (game.ball.x, game.ball.y)
    try:
        while distance((game.ball.x, game.ball.y), target_ball_pos) < 0.1:
            r1_data = game.get_robot_pos(True, r1)
            cmd1 = go_to_ball(
                pid_oren,
                pid_trans,
                r1_data,
                r1,
                game.ball,
            )
            sim_robot_controller.add_robot_commands(cmd1, r1)
            sim_robot_controller.send_robot_commands()
        # r1 get to pos
        while True:
            env.draw_point(
                passing_points[1][0], passing_points[1][1], width=2, color="red"
            )
            env.draw_point(
                receiving_points[1][0], receiving_points[1][1], width=2, color="yellow"
            )
            r1_data = game.get_robot_pos(True, r1)
            r0_data = game.get_robot_pos(True, r0)
            if distance((r0_data.x, r0_data.y), receiving_points[1]) < 0.05:
                break
            cmd0 = go_to_point(
                pid_oren,
                pid_trans,
                r0_data,
                r0,
                receiving_points[1],
                r0_data.orientation,
            )
            cmd1 = go_to_point(
                pid_oren,
                pid_trans,
                r1_data,
                r1,
                passing_points[1],
                r1_data.orientation,
                True,
            )

            print(distance((r1_data.x, r1_data.y), passing_points[1]))

            sim_robot_controller.add_robot_commands(cmd1, r1)
            sim_robot_controller.add_robot_commands(cmd0, r0)
            sim_robot_controller.send_robot_commands()

        # r1 pass to r0
        pass_task1 = PassBall(pid_oren, pid_trans, game, r1, r0, receiving_points[1])
        start_t = time.time()
        while time.time() - start_t < 5:
            env.draw_point(
                receiving_points[1][0], receiving_points[1][1], width=2, color="yellow"
            )
            cmd1, cmd0 = pass_task1.enact(True)
            sim_robot_controller.add_robot_commands(cmd0, r0)
            sim_robot_controller.add_robot_commands(cmd1, r1)
            sim_robot_controller.send_robot_commands()

        del pass_task1

        # r0 score goal
        while not game.is_ball_in_goal(our_side=False):
            cmd0 = score_goal(game, True, r0, pid_oren, pid_trans, True, True)
            sim_robot_controller.add_robot_commands(cmd0, r0)
            sim_robot_controller.send_robot_commands()

    except KeyboardInterrupt:
        print("Exiting...")

    # try:
    #     cmd = score_goal(
    #         game,
    #         sim_robot_controller.robot_has_ball(),
    #         shooter_id=shooter_id,
    #         pid_oren=pid_oren,
    #         pid_trans=pid_trans,
    #         is_yellow=is_yellow,
    #         shoot_in_left_goal=is_yellow,
    #     )

    #     if game.is_ball_in_goal(our_side=not is_yellow):
    #         logger.info("Goal Scored at Position: ", game.get_ball_pos())
    #         goal_scored = True
    #         break

    #     sim_robot_controller.add_robot_commands(cmd, shooter_id)
    #     sim_robot_controller.send_robot_commands()

    # except KeyboardInterrupt:
    #     print("Exiting...")


if __name__ == "__main__":
    test_rsim_2v5()
