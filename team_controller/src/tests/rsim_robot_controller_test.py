from team_controller.src.controllers import RSimRobotController
from entities.data.command import RobotCommand
import gymnasium as gym
from entities.game import Game

from rsoccer_simulator.src.ssl.envs.standard_ssl import SSLStandardEnv

game = Game()

# making environment
env = SSLStandardEnv()
env.reset()
robot_controller = RSimRobotController(is_team_yellow=True, env=env, game=game)

# Run for 1 episode and print reward at the end
for i in range(10000):
    terminated = False
    truncated = False
    if i > 100:
        robot_controller.add_robot_commands(RobotCommand(1, 0, 1, 0, 0, 0), 3)
        robot_controller.add_robot_commands(RobotCommand(0, 0, 1, 0, 0, 0), 4)
    robot_controller.send_robot_commands()
