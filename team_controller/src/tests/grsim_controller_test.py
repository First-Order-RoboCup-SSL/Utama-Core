import numpy as np

from team_controller.src.controllers import GRSimController


def main():
    controller = GRSimController()
    controller.teleport_ball(0, 0)
    controller.set_robot_presence(robot_id=0, is_team_yellow=False, is_present=False)
    controller.teleport_robot(is_team_yellow=True, robot_id=0, x=1, y=1, theta=np.pi)


if __name__ == "__main__":
    main()
