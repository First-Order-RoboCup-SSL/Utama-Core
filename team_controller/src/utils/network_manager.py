import socket
from typing import Tuple, Optional
from team_controller.src.utils import network_utils


class NetworkManager:
    """
    Manages network communication via a UDP socket for sending and receiving data.

    Args:
        address (Tuple[str, int]): The IP address and port to connect or bind to.
        bind_socket (bool): If True, binds the socket to the specified address for receiving data.
    """

    def __init__(self, address: Tuple[str, int], bind_socket: bool = False):
        # Initialize the NetworkManager and set up the socket.
        self.address = address
        self.sock = network_utils.setup_socket(
            socket.socket(socket.AF_INET, socket.SOCK_DGRAM), address, bind_socket
        )

    def send_command(self, command: object, is_sim_robot_cmd: bool = False) -> None:
        """
        Sends a command to the server at the specified address.

        Args:
            command (object): An object with in the form of a protocol buffer message to be serialized and sent.
            is_sim_robot_cmd (bool): If True, the function will attempt to receive a response from the server. (only used when sending robot control cmd)

        This method relies on a utility function for command transmission.
        """
        # Send a command to the server.
        return network_utils.send_command(self.address, command, is_sim_robot_cmd)

    def receive_data(self) -> Optional[bytes]:
        """
        Receives data from the server.

        Returns:
            Optional[bytes]: The received data as bytes if available, otherwise None.

        This method listens for incoming data from the socket using a utility function.
        """
        # Receive data from the server.
        return network_utils.receive_data(self.sock)

    def close(self) -> None:
        """
        Closes the socket connection safely.

        Attempts to close the socket and logs any exceptions that may occur during the process.
        """
        try:
            self.sock.close()
        except Exception as e:
            logger.error(f"Error closing socket: {e}")
