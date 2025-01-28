from entities.game.game_object import Robot as GameRobot

from motion_planning.src.pid.pid import TwoDPID, get_rsim_pids
from robot_control.src.skills import (
    go_to_point,
    mag,
    face_ball,
)
from team_controller.src.controllers import RSimRobotController
from rsoccer_simulator.src.ssl.envs.standard_ssl import SSLStandardEnv
from entities.game import Game
from team_controller.src.config.settings import TIMESTEP
from motion_planning.src.planning.path_planner import DynamicWindowPlanner
from math import dist
import random
import logging

logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.DEBUG)



def test_pathfinding(headless: bool):
    game = Game()
    N_ROBOTS_YELLOW = 6
    N_ROBOTS_BLUE = 6

    random.seed(3)
    env = SSLStandardEnv(
        n_robots_blue=N_ROBOTS_BLUE,
        n_robots_yellow=N_ROBOTS_YELLOW,
        render_mode="ansi" if headless else "human",
    )



    env.reset()
    mover_id = 1
    collider_ids = [2,3,4,5]
    collide_targets = []

    for i in range(6):
        env.teleport_robot(False, i, -4, i)
    env.teleport_ball(2.25, -1)

    # Spawn robots for head-on collision test

    for i, cid in enumerate(collider_ids):
        env.teleport_robot(True, cid, 1, -1+2*i / len(collider_ids))
        collide_targets.append([3, -1+2*i / len(collider_ids)])

    env.teleport_robot(True, mover_id, 3, 0)  # Robot 0 at origin

    is_yellow = True
    pid_oren, pid_2d = get_rsim_pids(N_ROBOTS_YELLOW if is_yellow else N_ROBOTS_BLUE)

    slow_pid2d = TwoDPID(TIMESTEP, 1, -1, 2, 0.1, 0.0, num_robots=6, normalize=False)

    
    sim_robot_controller = RSimRobotController(
        is_team_yellow=is_yellow, env=env, game_obj=game
    )
    planner = DynamicWindowPlanner(game)
    targets = [(0,0)]+[(random.uniform(-2, 2), random.uniform(-1, 1)) for _ in range(1000)]
    target = targets.pop(0)

    for _ in range(5000):
        env.draw_point(target[0], target[1], width=10, color="GREEN")

        velocity = game.get_object_velocity(GameRobot(True, mover_id))
        next_stop, _ = planner.path_to(mover_id, target)
        latest_frame = game.get_my_latest_frame(my_team_is_yellow=is_yellow)

        if latest_frame:
            friendly_robots, _, _ = latest_frame  
        r = friendly_robots[mover_id]
        if dist((r.x, r.y), target) < 0.02 and mag(velocity) < 0.02:
            target = targets.pop(0)

            env.draw_point(target[0], target[1], width=10, color="PINK")
            pid_oren.reset(mover_id)
            pid_2d.reset(mover_id)
        
        cmd = go_to_point(pid_oren, pid_2d, friendly_robots[mover_id], mover_id, next_stop, face_ball((r.x, r.y), next_stop))
        sim_robot_controller.add_robot_commands(cmd, mover_id)
        cmd_dict = {}
    
        for i, cid in enumerate(collider_ids):
            cmd_dict[cid] = go_to_point(pid_oren, slow_pid2d, friendly_robots[cid], cid, collide_targets[i], None)
        
        arrived = True

        for i, cid in enumerate(collider_ids):
            arrived = arrived and dist((friendly_robots[cid].x, friendly_robots[cid].y), collide_targets[i]) < 0.02
        
        if arrived:
            for x in collide_targets:
                x[0] *= -1
            
        cmd_dict = {}
        for i, cid in enumerate(collider_ids):
            cmd_dict[cid] = go_to_point(pid_oren, slow_pid2d, friendly_robots[cid], cid, collide_targets[i], None)

        sim_robot_controller.add_robot_commands(cmd_dict)
              
        sim_robot_controller.send_robot_commands()


if __name__ == "__main__":
    try:
        test_pathfinding(False)
    except KeyboardInterrupt:
        print("Exiting...")
