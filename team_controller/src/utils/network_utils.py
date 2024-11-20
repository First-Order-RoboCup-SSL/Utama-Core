import socket
import struct
import logging
from typing import Optional, Tuple

from team_controller.src.config.settings import MULTICAST_GROUP, LOCAL_HOST

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def setup_socket(
    sock: socket.socket, address: Tuple[str, int], bind_socket: bool = False
) -> socket.socket:
    """
    Configures a UDP socket with specified options, including multicast group settings if applicable.

    Args:
        sock (socket.socket): The socket to configure.
        address (Tuple[str, int]): The IP address and port to bind or connect the socket to.
        bind_socket (bool): If True, binds the socket to the given address for receiving data.

    Returns:
        socket.socket: The configured socket.

    Raises:
        socket.error: If socket configuration fails.

    This function sets up necessary socket options, binds the socket if required, and joins a multicast group if the
    address is not the local host. Additionally, a 1-second timeout is applied for non-blocking behavior.
    """
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8192)
        if bind_socket:
            sock.bind(address)

        if address[0] != LOCAL_HOST:
            group = socket.inet_aton(MULTICAST_GROUP)
            mreq = struct.pack("4sL", group, socket.INADDR_ANY)
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

        sock.settimeout(0.005)  # Set timeout to 1 frame period (60 FPS)
        logging.info(
            "Socket setup completed with address %s and bind_socket=%s",
            address,
            bind_socket,
        )
    except socket.error as e:
        logging.error("Socket setup failed for address %s with error: %s", address, e)
        raise  # Re-raise the exception to handle it further up if needed
    return sock


def receive_data(sock: socket.socket) -> Optional[bytes]:
    """
    Receives data from the socket.

    Args:
        sock (socket.socket): The socket from which to receive data.

    Returns:
        Optional[bytes]: The data received, or None if no data is received or if an error occurs.

    This function attempts to receive up to 8192 bytes of data. If a timeout or socket error occurs, it logs a warning
    or error and returns None. Unexpected errors are logged with an exception.
    """
    try:
        data, _ = sock.recvfrom(8192)
        return data
    except socket.timeout:
        logging.warning("Socket timed out while receiving data")
        return None
    except socket.error as e:
        logging.error("Socket error occurred while receiving data: %s", e)
        return None
    except Exception as e:
        logging.exception("Unexpected error receiving data")
        return None


def send_command(address: Tuple[str, int], command: object, is_sim_robot_cmd: bool = False) -> Optional[bytes]:
    """
    Sends a command to the specified address over a UDP socket.

    Args:
        address (Tuple[str, int]): The destination IP address and port.
        command object: An object with in the form of a protocol buffer message to be serialized and sent.
        is_sim_robot_cmd (bool): If True, the function will attempt to receive a response from the server.

    Returns:
        Optional[bytes]: The data received, or None if no data is received or if an error occurs.   
     
    This function creates a temporary UDP socket, serializes the command, and sends it to the specified address.
    If the command being sent is a RobotControl packet there will be a response packet which will be received.
    Errors during serialization or socket operations are logged, with specific handling if the `SerializeToString`
    method is missing.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as send_sock:
            serialized_command = command.SerializeToString()
            send_sock.sendto(serialized_command, address)
            if is_sim_robot_cmd:
                data = receive_data(send_sock)
                return data
            logging.info("Command sent to %s", address)
    except AttributeError:
        logging.error("Command object has no SerializeToString method")
    except socket.error as e:
        logging.error("Socket error when sending command to %s: %s", address, e)
    except Exception as e:
        logging.exception("Unexpected error sending command")
