import os
import sys
import numpy as np

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
print(project_root)
sys.path.insert(0, project_root)

from team_controller.src.controllers import GRSimController


def main():
    controller = GRSimController()
    controller.teleport_ball(0, 0)
    controller.set_robot_presence(
        robot_id=0, is_team_yellow=False, is_present=False
    )
    controller.teleport_robot(
        is_team_yellow=True, robot_id=0, x=1, y=1, theta=np.pi
    )


if __name__ == "__main__":
    main()
