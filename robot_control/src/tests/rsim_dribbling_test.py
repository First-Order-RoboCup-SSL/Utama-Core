import random
from team_controller.src.controllers.sim.rsim_robot_controller import (
    RSimRobotController,
)
from motion_planning.src.pid.pid import get_rsim_pids
from robot_control.src.high_level_skills import DribbleToTarget
from rsoccer_simulator.src.ssl.envs import SSLStandardEnv
from entities.game import Game

robot_id = 3

game = Game()
env = SSLStandardEnv()
robot_controller = RSimRobotController(True, env, game)

pid_oren, pid_trans = get_rsim_pids(6)
my_team_is_yellow = True

target_coords = [(4, 2.5), (4, -2.5), (-4, -2.5), (-4, 2.5)]
idx = 0
dribble_task = DribbleToTarget(
    pid_oren,
    pid_trans,
    game,
    robot_id,
    target_coords=target_coords[idx],
)

for _ in range(100000):
    f, e, b = game.get_my_latest_frame(my_team_is_yellow=my_team_is_yellow)

    if (
        (f[robot_id].x - target_coords[idx][0]) ** 2
        + (f[robot_id].y - target_coords[idx][1]) ** 2
    ) < 0.1:
        idx = (idx + 1) % 4
        dribble_task.update_coord(target_coords[idx])

    command = dribble_task.enact(robot_controller.robot_has_ball(robot_id))
    env.draw_point(target_coords[idx][0], target_coords[idx][1])
    # print(dribble_task.dribbled_distance)
    print(command)
    robot_controller.add_robot_commands(command, robot_id)
    robot_controller.send_robot_commands()
