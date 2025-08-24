import socket
import struct
import logging
from typing import Optional, Tuple

from config.settings import MULTICAST_GROUP, LOCAL_HOST, TIMESTEP

logger = logging.getLogger(__name__)


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
    timeout = TIMESTEP
    try:
        # Allow address reuse - crucial for servers restarting quickly
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # Set receive buffer size
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8192)
        actual_buffer_size = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
        if actual_buffer_size < 8192:
            logger.warning(
                f"OS reduced receive buffer size from {8192} to {actual_buffer_size}"
            )

        # Bind if requested (necessary for listening)
        if bind_socket:
            sock.bind(address)
            ip_addr = address[0]
            if ip_addr and ip_addr != LOCAL_HOST and ip_addr.startswith("224."):
                try:
                    group = socket.inet_aton(MULTICAST_GROUP)
                    mreq = struct.pack("4sL", group, socket.INADDR_ANY)
                    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
                    logger.info(f"Socket joined multicast group {MULTICAST_GROUP}")
                except socket.error as e:
                    logger.fatal(
                        f"Failed to join multicast group {MULTICAST_GROUP}: {e}"
                    )
                    raise

        # Set timeout for non-blocking behavior
        if timeout is not None:
            if timeout <= 0:
                logger.warning(
                    f"Timeout value ({timeout}) is not positive. Setting to None (blocking)."
                )
                sock.settimeout(None)
            else:
                sock.settimeout(timeout * 1.1)  # To account for network jitter
                logger.info(f"Socket timeout set to {timeout * 1.1:.4f} seconds.")
        else:
            logger.info("Socket configured for blocking operations (no timeout).")

        logger.info(f"Socket setup complete for address {address} (bind={bind_socket})")
        return sock

    except socket.error as e:
        logger.error(f"Socket setup failed for address {address}: {e}")
        sock.close()
        raise
    except Exception as e:
        logger.exception(f"Unexpected error during socket setup for {address}: {e}")
        sock.close()
        raise


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
        logger.warning("Socket timed out while receiving data")
        return None
    except socket.error as e:
        logger.error("Socket error occurred while receiving data: %s", e)
        return None
    except Exception as e:
        logger.exception("Unexpected error receiving data: %s", e)
        return None


def send_command(
    send_sock: socket.socket,
    address: Tuple[str, int],
    command: object,
    is_sim_robot_cmd: bool = False,
) -> Optional[bytes]:
    """
    Sends a command to the specified address over a UDP socket.

    Args:
        address (Tuple[str, int]): The destination IP address and port.
        command object: An object with in the form of a protocol buffer message to be serialized and sent.
        is_sim_robot_cmd (bool): If True, the function will attempt to receive a response from the server.

    Returns:
        Optional[bytes]: The data received, or None if no data is received or if an error occurs.

    If the command being sent is a RobotControl packet there will be a response packet which will be received.
    Errors during serialization or socket operations are logged, with specific handling if the `SerializeToString`
    method is missing.
    """
    try:
        if hasattr(command, "SerializeToString") and callable(
            command.SerializeToString
        ):
            serialized_data = command.SerializeToString()
        elif isinstance(command, bytes):
            serialized_data = command  # Allow sending raw bytes
        else:
            logger.error(f"Command object type {type(command)} cannot be serialized.")
            raise TypeError("Command must be bytes or have a SerializeToString method.")
        send_sock.sendto(serialized_data, address)

        # If the command is sent to the simulator, obtain the response
        if is_sim_robot_cmd:
            data = receive_data(send_sock)
            return data
        # logger.info("Command sent to %s", address)

    except AttributeError:
        logger.error("Command object has no SerializeToString method %s", command)
        raise AttributeError
    except socket.error as e:
        logger.error("Socket error when sending command to %s: %s", address, e)
    except Exception as e:
        logger.exception("Unexpected error sending command: %s", e)
