from team_controller.src.controllers import RSimRobotController
from entities.data.command import RobotCommand
import gymnasium as gym

# importing this as this test is run within the rsoccer_gym package (to perform the __init__.py registration)
import rsoccer_gym

# making environment
env = gym.make("SSLStandard-v0")
env.reset()
robot_controller = RSimRobotController(is_team_yellow=True, env=env)
# Run for 1 episode and print reward at the end
for i in range(10000):
    terminated = False
    truncated = False
    while not (terminated or truncated):
        robot_controller.add_robot_commands(RobotCommand(0, 0.1, 0, 0, 0, 0), 3)
        robot_controller.add_robot_commands(RobotCommand(0, 0, 1, 0, 0, 0), 4)
        robot_controller.send_robot_commands()
