import logging

from robot_control.src.intent import defend, score_goal
from robot_control.src.tests.utils import setup_pvp

from utama_core.entities.game import Game
from utama_core.motion_planning.src.pid import PID
from utama_core.motion_planning.src.pid.pid import TwoDPID, get_rsim_pids
from utama_core.rsoccer_simulator.src.ssl.envs.standard_ssl import SSLStandardEnv
from utama_core.team_controller.src.controllers import (
    RSimController,
    RSimRobotController,
)

logger = logging.getLogger(__name__)


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

    if game.is_ball_in_goal(defender_is_yellow):
        logger.info("Goal Scored at Position: ", game.get_ball_pos())
        return True
    return False


def test_single_defender(defender_id: int, shooter_id: int, defender_is_yellow: bool, headless: bool):
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
    env_controller = RSimController(env)
    env.reset()

    env.teleport_ball(2.25, -1)

    # Move the other defender out of the way
    for i in range(0, 6):
        if i != shooter_id:
            env_controller.set_robot_presence(i, not defender_is_yellow, False)

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

    any_scored = False
    defender_gets_ball = False
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
        cmd = defend(
            pid_oren_d,
            pid_2d_d,
            game,
            defender_is_yellow,
            defender_id,
            env,
        )
        sim_robot_controller_defender.add_robot_commands(cmd, defender_id)
        sim_robot_controller_defender.send_robot_commands()

        if sim_robot_controller_defender.robot_has_ball(defender_id):  # Sim ends when the defender gets the ball
            break
        defender_gets_ball = defender_gets_ball or sim_robot_controller_attacker.robot_has_ball(shooter_id)

    assert not any_scored
    assert defender_gets_ball


if __name__ == "__main__":
    try:
        test_single_defender(1, 3, True, False)
    except KeyboardInterrupt:
        print("Exiting...")
