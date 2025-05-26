from typing import Tuple
import time

from config.starting_formation import (
    LEFT_START_ONE,
    RIGHT_START_ONE,
)
from team_controller.src.utils import network_manager
from config.settings import (
    LOCAL_HOST,
    SIM_CONTROL_PORT,
    TELEPORT_X_COORDS,
    ADD_Y_COORD,
    REMOVAL_Y_COORD,
)

from team_controller.src.generated_code.ssl_simulation_control_pb2 import (
    TeleportBall,
    TeleportRobot,
    SimulatorControl,
    SimulatorCommand,
)
from team_controller.src.generated_code.ssl_gc_common_pb2 import RobotId, Team

from team_controller.src.controllers.common.sim_controller_abstract import (
    AbstractSimController,
)


class GRSimController(AbstractSimController):
    """
    A controller for interacting with a simulation environment for robot soccer, allowing actions such as teleporting the ball
    and setting robot presence on the field.

    Args:
        ip (str): IP address of the simulator. Defaults to LOCAL_HOST.
        port (int): Port of the simulator. Defaults to SIM_CONTROL_PORT.
    """

    def __init__(self, ip: str = LOCAL_HOST, port: int = SIM_CONTROL_PORT):
        self.net = network_manager.NetworkManager(address=(ip, port))

    def _create_simulator_command(self, control_message: object) -> object:
        sim_command = SimulatorCommand()
        sim_command.control.CopyFrom(control_message)
        return sim_command

    def teleport_ball(self, x: float, y: float, vx: float = 0, vy: float = 0) -> None:
        """
        Teleports the ball to a specific location on the field.

        Args:
            x (float): The x-coordinate to place the ball at (in meters [-4.5, 4.5]).
            y (float): The y-coordinate to place the ball at (in meters [-3.0, 3.0]).

        This method creates a command for teleporting the ball and sends it to the simulator.
        """
        sim_control = self._create_teleport_ball_command(x, y, vx, vy)
        sim_command = self._create_simulator_command(sim_control)
        self.net.send_command(sim_command)
        time.sleep(0.1)  # Allow some time for the command to be processed

    def _create_teleport_ball_command(
        self, x: float, y: float, vx: float, vy: float
    ) -> object:
        tele_ball = TeleportBall(x=x, y=y, z=0.1, vx=vx, vy=vy, vz=0)
        sim_control = SimulatorControl()
        sim_control.teleport_ball.CopyFrom(tele_ball)
        return sim_control

    def reset(self):
        for idx, x in enumerate(RIGHT_START_ONE):
            self.teleport_robot(True, idx, x[0], x[1], x[2])
        for idx, x in enumerate(LEFT_START_ONE):
            self.teleport_robot(False, idx, x[0], x[1], x[2])
        self.teleport_ball(0, 0, 0, 0)

    def teleport_robot(
        self,
        is_team_yellow: bool,
        robot_id: int,
        x: float,
        y: float,
        theta: float = None,
    ) -> None:
        """
        Teleports a robot to a specific location on the field.

        Args:
            is_team_yellow (bool): if the robot is team yellow, else blue
            robot_id (int): robot id
            x (float): The x-coordinate to place the ball at (in meters [-4.5, 4.5]).
            y (float): The y-coordinate to place the ball at (in meters [-3.0, 3.0]).
            theta (float): radian angle of the robot heading, 0 degrees faces towards positive x axis

        This method creates a command for teleporting the ball and sends it to the simulator.
        """
        sim_control = self._create_teleport_robot_command(
            robot_id, is_team_yellow, x, y, theta
        )
        sim_command = self._create_simulator_command(sim_control)
        self.net.send_command(sim_command)

    def set_robot_presence(
        self, robot_id: int, is_team_yellow: bool, is_present: bool
    ) -> None:
        """
        Sets a robot's presence on the field by teleporting it on and off the field.

        Args:
            robot_id (int): The unique ID of the robot.
            team_colour_is_blue (bool): Whether the robot belongs to the blue team. If False, it's assumed to be yellow.
            is_present (bool): If True, the robot will be placed on the field; if False, it will be despawned.

        The method calculates a teleport location based on the team and presence status, then sends a command to the simulator.
        """
        x, y = self._get_teleport_location(robot_id, is_team_yellow, is_present)
        sim_control = self._create_teleport_robot_command(
            robot_id, is_team_yellow, x, y, is_present
        )
        sim_command = self._create_simulator_command(sim_control)
        self.net.send_command(sim_command)

    def _create_teleport_robot_command(
        self,
        robot_id: int,
        is_team_yellow: bool,
        x: float,
        y: float,
        theta: float,
        is_present: bool = True,
    ) -> object:
        robot = RobotId(id=robot_id, team=Team.YELLOW if is_team_yellow else Team.BLUE)
        tele_robot = TeleportRobot(
            id=robot, x=x, y=y, orientation=theta, present=is_present
        )
        sim_control = SimulatorControl()
        sim_control.teleport_robot.add().CopyFrom(tele_robot)
        return sim_control

    def _get_teleport_location(
        self, robot_id: int, is_team_yellow: bool, add: bool
    ) -> Tuple[float, float]:
        y_coord = REMOVAL_Y_COORD if not add else ADD_Y_COORD
        x_coord = (
            -TELEPORT_X_COORDS[robot_id]
            if is_team_yellow
            else TELEPORT_X_COORDS[robot_id]
        )
        return x_coord, y_coord
