from typing import Tuple

from team_controller.src.utils import network_manager
from team_controller.src.config.settings import (
    LOCAL_HOST,
    SIM_COMTROL_PORT,
    TELEPORT_X_COORDS,
    FIELD_Y_COORD,
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
        port (int): Port of the simulator. Defaults to SIM_COMTROL_PORT.
    """

    def __init__(self, ip: str = LOCAL_HOST, port: int = SIM_COMTROL_PORT):
        self.net = network_manager.NetworkManager(address=(ip, port))

    def _create_simulator_command(self, control_message: object) -> object:
        sim_command = SimulatorCommand()
        sim_command.control.CopyFrom(control_message)
        return sim_command

    def teleport_ball(self, x: float, y: float) -> None:
        """
        Teleports the ball to a specific location on the field.

        Args:
            x (float): The x-coordinate to place the ball at.
            y (float): The y-coordinate to place the ball at.

        This method creates a command for teleporting the ball and sends it to the simulator.
        """
        sim_control = self._create_teleport_ball_command(x, y)
        sim_command = self._create_simulator_command(sim_control)
        self.net.send_command(sim_command)

    def _create_teleport_ball_command(self, x: float, y: float) -> object:
        tele_ball = TeleportBall(x=x, y=y)
        sim_control = SimulatorControl()
        sim_control.teleport_ball.CopyFrom(tele_ball)
        return sim_control

    def set_robot_presence(
        self, robot_id: int, team_colour_is_blue: bool, should_robot_be_present: bool
    ) -> None:
        """
        Sets a robot's presence on the field by teleporting it to a specific location or removing it from the field.

        Args:
            robot_id (int): The unique ID of the robot.
            team_colour_is_blue (bool): Whether the robot belongs to the blue team. If False, it's assumed to be yellow.
            should_robot_be_present (bool): If True, the robot will be placed on the field; if False, it will be removed.

        The method calculates a teleport location based on the team and presence status, then sends a command to the simulator.
        """
        x, y = self._get_teleport_location(
            robot_id, team_colour_is_blue, should_robot_be_present
        )
        sim_control = self._create_teleport_robot_command(
            robot_id, team_colour_is_blue, x, y, should_robot_be_present
        )
        sim_command = self._create_simulator_command(sim_control)
        self.net.send_command(sim_command)

    def _create_teleport_robot_command(
        self,
        robot_id: int,
        team_colour_is_blue: bool,
        x: float,
        y: float,
        present: bool,
    ) -> object:
        robot = RobotId(
            id=robot_id, team=Team.BLUE if team_colour_is_blue else Team.YELLOW
        )
        tele_robot = TeleportRobot(id=robot, x=x / 1000, y=y / 1000, present=present)
        sim_control = SimulatorControl()
        sim_control.teleport_robot.add().CopyFrom(tele_robot)
        return sim_control

    def _get_teleport_location(
        self, robot_id: int, team_colour_is_blue: bool, add: bool
    ) -> Tuple[float, float]:
        y_coord = FIELD_Y_COORD if add else REMOVAL_Y_COORD
        x_coord = (
            -TELEPORT_X_COORDS[robot_id]
            if team_colour_is_blue
            else TELEPORT_X_COORDS[robot_id]
        )
        return x_coord, y_coord
