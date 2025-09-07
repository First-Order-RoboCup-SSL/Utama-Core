import random

from robot_control.src.high_level_skills import DribbleToTarget

from utama_core.entities.game import Game
from utama_core.motion_planning.src.pid.pid import get_rsim_pids
from utama_core.rsoccer_simulator.src.ssl.envs import SSLStandardEnv
from utama_core.team_controller.src.controllers.sim.rsim_robot_controller import (
    RSimRobotController,
)

robot_id = 3

game = Game()
env = SSLStandardEnv()
robot_controller = RSimRobotController(True, env, game)

pid_oren, pid_trans = get_rsim_pids()
my_team_is_yellow = True
target_coords = (random.random() * 4 - 2, random.random() * 4 - 2)
dribble_task = DribbleToTarget(
    pid_oren,
    pid_trans,
    game,
    robot_id,
    target_coords=target_coords,
)

for _ in range(100000):
    f, e, b = game.get_my_latest_frame(my_team_is_yellow=my_team_is_yellow)

    if ((f[robot_id].x - target_coords[0]) ** 2 + (f[robot_id].y - target_coords[1]) ** 2) < 0.1:
        target_coords = (random.random() * 4 - 2, random.random() * 4 - 2)

        dribble_task = DribbleToTarget(
            pid_oren,
            pid_trans,
            game,
            robot_id,
            target_coords=target_coords,
        )

    command = dribble_task.enact(robot_controller.robot_has_ball(robot_id))
    print(dribble_task.dribbled_distance)
    env.draw_point(target_coords[0], target_coords[1])
    robot_controller.add_robot_commands(command, robot_id)
    robot_controller.send_robot_commands()
