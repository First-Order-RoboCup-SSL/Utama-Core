import random
from motion_planning.src.pid.pid import get_rsim_pids
from robot_control.src.skills import face_ball, go_to_point, goalkeep, find_likely_enemy_shooter
from robot_control.src.tests.utils import setup_pvp
from team_controller.src.controllers import RSimRobotController
from rsoccer_simulator.src.ssl.envs.standard_ssl import SSLStandardEnv
from entities.game import Game
from robot_control.src.intent import score_goal
from motion_planning.src.pid import PID
from config.settings import TIMESTEP
import logging
import time

logger = logging.getLogger(__name__)


ITERS = 500
N_ROBOTS = 6


def test_shooting(shooter_id: int, defender_is_yellow: bool, headless: bool):
    """When the tests are run with pytest, these parameters are filled in
    based on whether we are in full or quick test mode (see conftest.py)"""
    game = Game()

    if defender_is_yellow:
        N_ROBOTS_YELLOW = 1
        N_ROBOTS_BLUE = 6
    else:
        N_ROBOTS_BLUE = 1
        N_ROBOTS_YELLOW = 6

    env = SSLStandardEnv(
        n_robots_blue=N_ROBOTS_BLUE,
        n_robots_yellow=N_ROBOTS_YELLOW,
        render_mode="ansi" if headless else "human",
    )
    env.reset()

    env.teleport_ball(-1, 1)

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

    goal_scored = False
    shoot_in_left_goal = random.random() > 0.5

    time.sleep(2)
    
    for iter in range(ITERS):
        # TODO: We should move robot_has_ball within game obj as well
        # This will do for now.
        if not goal_scored:
            friendly, enemy, balls = game.get_my_latest_frame(
                my_team_is_yellow=defender_is_yellow
            )
            
            f = game.predict_next_frame()
            if f:
                env.draw_point(f.friendly_robots[shooter_id].x, f.friendly_robots[shooter_id].y, color="YELLOW")
                env.draw_point(f.enemy_robots[0].x, f.enemy_robots[0].y, color="BLUE")
        
            attack_cmd = score_goal(
                game,
                sim_robot_controller_attacker.robot_has_ball(shooter_id),
                shooter_id=shooter_id,
                pid_oren=pid_oren_a,
                pid_trans=pid_2d_a,
                is_yellow=not defender_is_yellow,
                shoot_in_left_goal=True,
            )

            if game.is_ball_in_goal(right_goal=not defender_is_yellow):
                logger.info("Goal Scored at Position: ", game.get_ball_pos())
                goal_scored = True

            sim_robot_controller_attacker.add_robot_commands(attack_cmd, shooter_id)
            sim_robot_controller_attacker.send_robot_commands()

            if defender_is_yellow:
                target = game.predict_ball_pos_at_x(4.5 - 0.4)
            else:
                target = game.predict_ball_pos_at_x(-4.5 + 0.4)
            
            if target and not find_likely_enemy_shooter(enemy, balls):
                defend_cmd = go_to_point(
                    pid_oren_d,
                    pid_2d_d,
                    friendly[0],
                    0,
                    target,
                    face_ball(
                        (friendly[0].x, friendly[0].y), (game.ball.x, game.ball.y)
                    ),
                    dribbling=True,
                )
                sim_robot_controller_defender.add_robot_commands(defend_cmd, 0)
                sim_robot_controller_defender.send_robot_commands()
            else:
                defend_cmd = go_to_point(
                    pid_oren_d,
                    pid_2d_d,
                    friendly[0],
                    0,
                    [None, None],
                    face_ball(
                        (friendly[0].x, friendly[0].y), (game.ball.x, game.ball.y)
                    ),
                )
                sim_robot_controller_defender.add_robot_commands(defend_cmd, 0)
                sim_robot_controller_defender.send_robot_commands()

    assert not goal_scored


if __name__ == "__main__":
    try:
        test_shooting(4, False, False)
    except KeyboardInterrupt:
        print("Exiting...")
