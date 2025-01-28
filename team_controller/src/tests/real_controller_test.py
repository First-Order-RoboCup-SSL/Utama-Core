from team_controller.src.controllers import RealRobotController
from entities.game import Game
from entities.data.command import RobotCommand

import logging

logger = logging.getLogger(__name__)

stop_buffer = [0, 0, 0, 0, 0, 0, 0, 0]

game = Game()

robot_controller = RealRobotController(is_team_yellow=True, game_obj=game, n_robots=1)
try:
    robot_controller.add_robot_commands(RobotCommand(1, -0.1, 7.6, 0, 1, 1), 0)
    binary_representation = [f"{byte:08b}" for byte in robot_controller.out_packet]
    print(binary_representation)
    robot_controller.send_robot_commands()

except KeyboardInterrupt:
    robot_controller.serial.write(stop_buffer)
