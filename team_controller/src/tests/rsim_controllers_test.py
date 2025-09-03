import logging

from entities.data.command import RobotCommand
from entities.game import Game
from rsoccer_simulator.src.ssl.envs.standard_ssl import SSLStandardEnv
from team_controller.src.controllers import RSimController, RSimRobotController

logger = logging.getLogger(__name__)

game = Game()

# making environment: you can set the number of robots for blue and yellow
# their starting formation will follow the default start expressed in team_controller.src.config.starting_formation
# but you can modify this directly to create specific setup scenarios
env = SSLStandardEnv(n_robots_blue=3, n_robots_yellow=5)
robot_controller = RSimRobotController(is_team_yellow=True, env=env, game_obj=game)
game_controller = RSimController(env=env)

# Run for 1 episode and print reward at the end
for i in range(10000):
    if i > 100:
        # follows GRSimRootController setup. Add robot commands them send them like below
        robot_controller.add_robot_commands(RobotCommand(0, -0.01, 0, 0, 0, 1), 3)
        robot_controller.add_robot_commands(RobotCommand(0, 0, 1, 0, 0, 0), 4)
    if i == 150:
        # RSimController gives you the abulity to teleport the ball directly
        # note that neither teleport operations create a new frame, only modifies current frame.
        game_controller.teleport_ball(x=1.5, y=1.5, vx=1, vy=5)
        logger.info("Ball has been teleported!")
    if i == 152:
        # RSimController gives you the ability to teleport robots directly
        game_controller.teleport_robot(is_team_yellow=True, robot_id=3, x=1.2, y=-2, theta=-1.5)
        logger.info("Robot 3 (yellow) has been teleported!")

    # send robot commands after adding them
    robot_controller.send_robot_commands()
