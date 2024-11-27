import queue
import abc


class BaseReceiver(abc.ABC):
    """Interface for receivers. Every receiver should add messages to the passed-in message queue which 
    will notify main of something to do."""

    def __init__(self, message_queue:queue.SimpleQueue):
        self._message_queue = message_queue
