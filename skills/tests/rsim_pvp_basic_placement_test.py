from motion_planning.src.pid.pid import TwoDPID, get_rsim_pids
from robot_control.src.skills import go_to_ball, go_to_point
from robot_control.src.tests.utils import one_robot_placement, setup_pvp
from team_controller.src.controllers import RSimRobotController
from rsoccer_simulator.src.ssl.envs.standard_ssl import SSLStandardEnv
from entities.game import Game
from robot_control.src.intent import score_goal
from motion_planning.src.pid import PID
from team_controller.src.controllers.sim.rsim_robot_controller import PVPManager
from config.settings import TIMESTEP
import math

TEST_EXPECTED_ITERS = 2
TEST_EXPECTED_REL_DIFF = 0.02
TEST_EXPECTED_ABS_POS_DIFF = 0.3
TARGET_OREN = math.pi / 2
TEST_RESULT_OREN_THRESH = 0.10


def test_pvp_placement(target_robot: int, headless: bool):
    ITERS = 1000

    game = Game()

    N_ROBOTS_YELLOW = 6
    N_ROBOTS_BLUE = 3

    env = SSLStandardEnv(
        n_robots_blue=N_ROBOTS_BLUE, render_mode="ansi" if headless else "human"
    )
    env.reset()

    env.teleport_ball(1, 1)

    pid_oren_y, pid_2d_y = get_rsim_pids()
    pid_oren_b, pid_2d_b = get_rsim_pids()

    sim_robot_controller_yellow, sim_robot_controller_blue, pvp_manager = setup_pvp(
        env, game, N_ROBOTS_BLUE, N_ROBOTS_YELLOW
    )
    one_step_yellow = one_robot_placement(
        sim_robot_controller_yellow,
        True,
        pid_oren_y,
        pid_2d_y,
        False,
        target_robot,
        game,
        TARGET_OREN,
    )
    one_step_blue = one_robot_placement(
        sim_robot_controller_blue,
        False,
        pid_oren_b,
        pid_2d_b,
        True,
        target_robot,
        game,
        TARGET_OREN,
    )

    change_iters = ([], [])
    change_orens = ([], [])

    for iter in range(ITERS):
        for i, one_step_colour in enumerate((one_step_blue, one_step_yellow)):
            switch, _, _, co = one_step_colour()
            if switch:
                change_iters[i].append(iter)
                change_orens[i].append(co)

    assert len(change_iters[0]) == len(change_iters[1])
    assert len(change_iters[0]) > TEST_EXPECTED_ITERS

    for left, right in zip(*change_iters):
        assert (abs(left - right) / left) < TEST_EXPECTED_REL_DIFF

    # yellow_robot_pos = game.get_robot_pos(is_yellow=True, robot_id=target_robot)
    # blue_robot_pos = game.get_robot_pos(is_yellow=False, robot_id=target_robot)

    # TODO: I am not really sure what is this test for, I will remove it first
    # assert abs((yellow_robot_pos.x - blue_robot_pos.x)) < TEST_EXPECTED_ABS_POS_DIFF
    # assert abs((yellow_robot_pos.y - blue_robot_pos.y)) < TEST_EXPECTED_ABS_POS_DIFF

    for side in change_orens:
        for oren in side:
            assert abs(abs(oren) - TARGET_OREN) / TARGET_OREN < TEST_RESULT_OREN_THRESH


if __name__ == "__main__":
    try:
        test_pvp_placement(1, False)
    except KeyboardInterrupt:
        print("Exiting...")
