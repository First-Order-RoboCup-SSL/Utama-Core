from motion_planning.src.pid.pid import TwoDPID, get_rsim_pids
from robot_control.src.skills import empty_command, go_to_ball, go_to_point, goalkeep
from robot_control.src.tests.utils import one_robot_placement, setup_pvp
from robot_control.src.utils.shooting_utils import find_best_shot
from team_controller.src.controllers import RSimRobotController
from rsoccer_simulator.src.ssl.envs.standard_ssl import SSLStandardEnv
from entities.game import Game
from robot_control.src.intent import PassBall, defend, score_goal
from motion_planning.src.pid import PID
from team_controller.src.controllers.sim.rsim_robot_controller import PVPManager
from team_controller.src.config.settings import TIMESTEP
from entities.data.command import RobotCommand
import math
import random

random.seed(4)

def test_three_one_one_v_two(attacker_is_yellow: bool, headless: bool):
    game = Game()

    N_ROBOTS_ATTACK = 3
    N_ROBOTS_DEFEND = 2

    N_ROBOTS_YELLOW = N_ROBOTS_ATTACK if attacker_is_yellow else N_ROBOTS_DEFEND  
    N_ROBOTS_BLUE = N_ROBOTS_DEFEND if attacker_is_yellow else N_ROBOTS_ATTACK  
    
    START_POS = -2
    SPACING_Y = 1.5
    SPACING_X = 1
    SPAWN_BOX_SIZE = 1

    env = SSLStandardEnv(
        n_robots_blue=N_ROBOTS_BLUE, n_robots_yellow=N_ROBOTS_YELLOW, render_mode="ansi" if headless else "human"
    )
    env.reset()


    if attacker_is_yellow:
        target_pos = [(START_POS - (i + 1) % 2 * SPACING_X, SPACING_Y - SPACING_Y * i) for i in range(N_ROBOTS_ATTACK)]
    else:
        target_pos = [(-START_POS + (i + 1) % 2 * SPACING_X, SPACING_Y - SPACING_Y * i) for i in range(N_ROBOTS_ATTACK)]

    env.teleport_ball(random.uniform(-3, 3), random.uniform(-3, 3))
    
    for i in range(N_ROBOTS_ATTACK):
        # print(random.uniform(-SPAWN_BOX_SIZE, SPAWN_BOX_SIZE))
        # env.teleport_robot(attacker_is_yellow, i, target_pos[i][0] + random.uniform(-SPAWN_BOX_SIZE, SPAWN_BOX_SIZE), target_pos[i][1] + random.uniform(-SPAWN_BOX_SIZE, SPAWN_BOX_SIZE))
        env.teleport_robot(attacker_is_yellow, i, 0, -0.3 * (i - 1))

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

    pass_task = None
    shooting = False
    goal_scored = False

    stage = 0
    passes = 0

    for iter in range(2000):
        if iter % 100 == 0:
            print(iter)
    
        goal_scored = goal_scored or game.is_ball_in_goal(not attacker_is_yellow)
        if game.is_ball_in_goal(not attacker_is_yellow):
            break

        sim_robot_controller_defender.add_robot_commands(defend(pid_oren_defender, pid_2d_defender, game, not attacker_is_yellow, 1, env), 1)
        sim_robot_controller_defender.add_robot_commands(goalkeep(attacker_is_yellow, game, 0, pid_oren_defender, pid_2d_defender, not attacker_is_yellow, sim_robot_controller_defender.robot_has_ball(0)), 0)
        sim_robot_controller_defender.send_robot_commands()

        if stage == 0:
            if iter == 10: # give them chance to spawn in the correct place
                stage += 1
        elif stage == 1:
            closest_robot = None
            closest_distance = float("inf")
            for i in range(N_ROBOTS_ATTACK):
                robot_data = game.get_robot_pos(attacker_is_yellow, i)
                ball = game.get_ball_pos()[0]

                distance = math.dist((robot_data.x, robot_data.y), (ball.x, ball.y))  
                if distance < closest_distance:
                    closest_distance = distance
                    closest_robot = i
            
            all_in_pos = True
            for i in range(N_ROBOTS_ATTACK):
                robot_data = game.get_robot_pos(attacker_is_yellow, i)
                ball = game.get_ball_pos()[0]

                if i == closest_robot and not sim_robot_controller_attacker.robot_has_ball(i):
                    sim_robot_controller_attacker.add_robot_commands(go_to_ball(pid_oren_attacker, pid_2d_attacker, robot_data, i, ball), i)
                    possessor = i
                else:
                    sim_robot_controller_attacker.add_robot_commands(go_to_point(pid_oren_attacker, pid_2d_attacker, robot_data, i, target_pos[i], math.pi, True), i)

                all_in_pos = all_in_pos and math.dist(target_pos[i], (robot_data.x, robot_data.y)) < 0.01
            sim_robot_controller_attacker.send_robot_commands()

            if all_in_pos:
                stage += 1
        
        elif stage == 2:
            if not pass_task:
                target_goal_line = game.field.enemy_goal_line(attacker_is_yellow)
                latest_frame = game.get_my_latest_frame(attacker_is_yellow)
                if latest_frame:
                    friendly_robots, enemy_robots, balls = latest_frame

                    goal_x = target_goal_line.coords[0][0]
                    goal_y1 = target_goal_line.coords[1][1]
                    goal_y2 = target_goal_line.coords[0][1]

                    best_shot, size_of_shot = find_best_shot(
                        balls[0], enemy_robots, goal_x, goal_y1, goal_y2, attacker_is_yellow
                    )

                    print("SIZE OF SHOT", size_of_shot)

                    if size_of_shot > 0.41 and passes >= 5:
                        stage += 1

                passes += 1
                if possessor == 0:
                    next_possessor = 1
                elif possessor == N_ROBOTS_ATTACK - 1:
                    next_possessor = N_ROBOTS_ATTACK - 2
                else:
                    next_possessor = random.choice([possessor + 1, possessor - 1])

                pass_task = PassBall(
                    pid_oren_attacker,
                    pid_2d_attacker,
                    game,
                    possessor,
                    next_possessor,
                    target_coords=game.get_robot_pos(attacker_is_yellow, next_possessor),
                )
            # else:
            print("Possessor", possessor, sim_robot_controller_attacker.robot_has_ball(possessor), "Next possessor", next_possessor, sim_robot_controller_attacker.robot_has_ball(next_possessor))
            if sim_robot_controller_attacker.robot_has_ball(next_possessor):
                pass_task = None
                possessor = next_possessor
                print("RECEIVED")
                sim_robot_controller_attacker.add_robot_commands(empty_command(dribbler_on=True), possessor)
                sim_robot_controller_attacker.add_robot_commands(empty_command(dribbler_on=True), 0)
                sim_robot_controller_attacker.send_robot_commands()
            else:
                (possessor_cmd, next_possessor_cmd) = pass_task.enact(sim_robot_controller_attacker.robot_has_ball(possessor))
                sim_robot_controller_attacker.add_robot_commands(possessor_cmd, possessor)
                sim_robot_controller_attacker.add_robot_commands(next_possessor_cmd, next_possessor)
                print("Possessor dribblle", possessor_cmd.dribble, "Receiver dribble", next_possessor_cmd.dribble)
                sim_robot_controller_attacker.send_robot_commands()
        elif stage == 3:
            print("SCOOOORING")
            sim_robot_controller_attacker.add_robot_commands(score_goal(game, sim_robot_controller_attacker.robot_has_ball(possessor), possessor, pid_oren_attacker, pid_2d_attacker, attacker_is_yellow, attacker_is_yellow), possessor)
            sim_robot_controller_attacker.send_robot_commands()
    
    assert goal_scored

# Problems:
    # - ball gets spilled by the recevier
    # Attacker is way too slow to shoot - rotation is too slow
    # Goalkeeper reacts too slowly

if __name__ == "__main__":
    try:
        test_three_one_one_v_two(True, False)
    except KeyboardInterrupt:
        print("Exiting...")
