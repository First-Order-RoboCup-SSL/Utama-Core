from enum import Enum


class MessageType(Enum):
    """Describes the type of message added to the global message queue by the receivers"""    
    VISION = 1
    REF = 2
