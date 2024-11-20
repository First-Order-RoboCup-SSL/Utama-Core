import os
import sys

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
print(project_root)
sys.path.insert(0, project_root)

from team_controller.src.controllers.sim_controller import SimulatorController

def main():
    controller = SimulatorController()
    controller.teleport_ball(0, 0)
    controller.set_robot_presence(0, team_colour_is_blue=False, should_robot_be_present=True)

if __name__ == "__main__":
    main()