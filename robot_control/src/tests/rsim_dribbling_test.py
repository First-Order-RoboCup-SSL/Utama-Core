from team_controller.src.controllers.sim.rsim_robot_controller import (
    RSimRobotController,
)
from motion_planning.src.pid.pid import get_pids
from robot_control.src.high_level_skills import DribbleToTarget
from rsoccer_simulator.src.ssl.envs import SSLStandardEnv
from entities.game import Game

robot_id = 3
target_coords = (-1, 1)

game = Game()
env = SSLStandardEnv()
robot_controller = RSimRobotController(True, env, game)

pid_oren, pid_trans = get_pids(6)

dribble_task = DribbleToTarget(
    pid_oren,
    pid_trans,
    game,
    robot_id,
    target_coords=target_coords,
)

for _ in range(100000):
    command = dribble_task.enact(robot_controller.robot_has_ball(robot_id))
    env.draw_point(target_coords[0], target_coords[1])
    robot_controller.add_robot_commands(command, robot_id)
    robot_controller.send_robot_commands()
