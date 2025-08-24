from motion_planning.src.pid.pid import TwoDPID, get_rsim_pids
from robot_control.src.skills import go_to_point, goalkeep
from robot_control.src.tests.utils import setup_pvp
from rsoccer_simulator.src.ssl.envs.standard_ssl import SSLStandardEnv
from entities.game import Game
from robot_control.src.intent import PassBall, defend, score_goal
from motion_planning.src.pid import PID
from entities.data.command import RobotCommand
import math

CHARGE_ACHIEVED_THRESH = 0.1
CHARGE_FORWARD_DELTA = 0.7


class ChargeTask:
    def __init__(
        self,
        pid_oren: PID,
        pid_trans: TwoDPID,
        robot_id: int,
        is_yellow: bool,
        forward_oren: float,
        game: Game,
    ):
        robot_data = game.get_robot_pos(is_yellow, robot_id)
        self.target_coords = (
            robot_data.x + CHARGE_FORWARD_DELTA * math.cos(forward_oren),
            robot_data.y,
        )
        self.pid_oren = pid_oren
        self.pid_trans = pid_trans
        self.robot_id = robot_id
        self.is_yellow = is_yellow
        self.forward_oren = forward_oren
        self.game = game

    def enact(self):
        robot_data = self.game.get_robot_pos(self.is_yellow, self.robot_id)
        return go_to_point(
            self.pid_oren,
            self.pid_trans,
            robot_data,
            self.robot_id,
            self.target_coords,
            self.forward_oren,
            True,
        )

    def done(self):
        new_pos = self.game.get_robot_pos(self.is_yellow, self.robot_id)
        return (
            math.hypot(
                new_pos.x - self.target_coords[0], new_pos.y - self.target_coords[1]
            )
            < CHARGE_ACHIEVED_THRESH
        )


def test_three_one_one(attacker_is_yellow: bool, headless: bool):
    game = Game()

    N_ROBOTS_ATTACK = 3
    N_ROBOTS_DEFEND = 2

    N_ROBOTS_YELLOW = N_ROBOTS_ATTACK if attacker_is_yellow else N_ROBOTS_DEFEND
    N_ROBOTS_BLUE = N_ROBOTS_DEFEND if attacker_is_yellow else N_ROBOTS_ATTACK

    START_POS = 2
    SHOOT_THRESH = 2.5
    SPACING = 2.5

    env = SSLStandardEnv(
        n_robots_blue=N_ROBOTS_BLUE,
        n_robots_yellow=N_ROBOTS_YELLOW,
        render_mode="ansi" if headless else "human",
    )
    env.reset()

    if attacker_is_yellow:
        for i in range(3):
            env.teleport_robot(attacker_is_yellow, i, START_POS, SPACING - SPACING * i)
        env.teleport_ball(START_POS - 0.5, 0)
    else:
        for i in range(3):
            env.teleport_robot(attacker_is_yellow, i, -START_POS, SPACING - SPACING * i)
        env.teleport_ball(-START_POS + 0.5, 0)

    sim_robot_controller_yellow, sim_robot_controller_blue, pvp_manager = setup_pvp(
        env, game, N_ROBOTS_BLUE, N_ROBOTS_YELLOW
    )

    if attacker_is_yellow:
        sim_robot_controller_attacker = sim_robot_controller_yellow
        sim_robot_controller_defender = sim_robot_controller_blue
    else:
        sim_robot_controller_attacker = sim_robot_controller_blue
        sim_robot_controller_defender = sim_robot_controller_yellow

    pid_oren_attacker, pid_2d_attacker = get_rsim_pids()
    pid_oren_defender, pid_2d_defender = get_rsim_pids()

    possessor = 1
    dp = 1

    charge_tasks = None
    pass_task = None
    shooting = False
    goal_scored = False

    for iter in range(2000):
        if iter % 100 == 0:
            print(iter)

        sim_robot_controller_defender.add_robot_commands(
            defend(
                pid_oren_defender, pid_2d_defender, game, not attacker_is_yellow, 1, env
            ),
            1,
        )
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
        sim_robot_controller_defender.send_robot_commands()

        if iter > 10:  # give them chance to spawn in the correct place
            goal_scored = goal_scored or game.is_ball_in_goal(not attacker_is_yellow)
            if game.is_ball_in_goal(not attacker_is_yellow):
                break

            if shooting:
                cmd = score_goal(
                    game,
                    True,
                    possessor,
                    pid_oren_attacker,
                    pid_2d_attacker,
                    attacker_is_yellow,
                    attacker_is_yellow,
                )
                for npc_attacker in set(range(N_ROBOTS_ATTACK)).difference([possessor]):
                    sim_robot_controller_attacker.add_robot_commands(
                        RobotCommand(0, 0, 0, 0, 0, 0), npc_attacker
                    )
                sim_robot_controller_attacker.add_robot_commands(cmd, possessor)
            elif pass_task:  # Passing...
                if sim_robot_controller_attacker.robot_has_ball(
                    pass_task.receiver_id
                ):  # Finished passing... # TODO put this check in a done() method inside passing task
                    possessor = pass_task.receiver_id
                    pass_task = None
                    if possessor in (0, 2):
                        dp = -dp
                else:  # Still passing...
                    next_possessor = pass_task.receiver_id
                    passer_cmd, receiver_cmd = pass_task.enact(
                        passer_has_ball=sim_robot_controller_attacker.robot_has_ball(
                            possessor
                        )
                    )
                    sim_robot_controller_attacker.add_robot_commands(
                        passer_cmd, possessor
                    )
                    sim_robot_controller_attacker.add_robot_commands(
                        receiver_cmd, next_possessor
                    )
            else:  # Charging...
                if not charge_tasks:  # First time charging...
                    charge_tasks = [
                        ChargeTask(
                            pid_oren_attacker,
                            pid_2d_attacker,
                            i,
                            attacker_is_yellow,
                            math.pi if attacker_is_yellow else 0,
                            game,
                        )
                        for i in range(N_ROBOTS_ATTACK)
                    ]

                all_done = True
                for i, task in enumerate(charge_tasks):
                    if not task.done():
                        sim_robot_controller_attacker.add_robot_commands(
                            task.enact(), i
                        )
                        all_done = False

                if all_done:  # Finished charging...
                    if (
                        abs(game.get_robot_pos(attacker_is_yellow, possessor).x)
                        > SHOOT_THRESH
                    ):
                        shooting = True
                    else:
                        charge_tasks = None
                        next_possessor = possessor + dp
                        pass_task = PassBall(
                            pid_oren_attacker,
                            pid_2d_attacker,
                            game,
                            possessor,
                            next_possessor,
                            target_coords=game.get_robot_pos(
                                attacker_is_yellow, next_possessor
                            ),
                        )

                        npc_attacker = list(
                            set(range(N_ROBOTS_ATTACK)).difference(
                                set([possessor, next_possessor])
                            )
                        )[0]
                        sim_robot_controller_attacker.add_robot_commands(
                            RobotCommand(0, 0, 0, 0, 0, 0), npc_attacker
                        )

            sim_robot_controller_attacker.send_robot_commands()

    assert goal_scored

    # pass_ball_task =
    # possessor = next_possessor

    #     if not passed:

    #             logger.info("Passed.")
    #             passed = True
    #             time.sleep(1)
    #             break

    #         sim_robot_controller.add_robot_commands(passer_cmd, passer_id)
    #         sim_robot_controller.add_robot_commands(receiver_cmd, receiver_id)
    #         sim_robot_controller.send_robot_commands()

    # assert passed


if __name__ == "__main__":
    try:
        test_three_one_one(True, False)
    except KeyboardInterrupt:
        print("Exiting...")
