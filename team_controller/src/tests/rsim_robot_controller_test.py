from team_controller.src.controllers import RSimRobotController
from entities.data.command import RobotCommand
from entities.game import Game

from rsoccer_simulator.src.ssl.envs.standard_ssl import SSLStandardEnv

game = Game()

# making environment
env = SSLStandardEnv(n_robots_blue=3, n_robots_yellow=5)
env.reset()
robot_controller = RSimRobotController(is_team_yellow=True, env=env, game_obj=game)

# Run for 1 episode and print reward at the end
for i in range(10000):
    if i > 100:
        robot_controller.add_robot_commands(RobotCommand(0.1, 0, 0, 0, 0, 1), 3)
        robot_controller.add_robot_commands(RobotCommand(0, 0, 1, 0, 0, 0), 4)
    robot_controller.send_robot_commands()
    print(robot_controller.robot_has_ball(3))
