from enum import Enum


class RefereeCommand(Enum):
    """
    Enum representing a referee command.
    """

    HALT = 0
    STOP = 1
    NORMAL_START = 2
    FORCE_START = 3
    PREPARE_KICKOFF_YELLOW = 4
    PREPARE_KICKOFF_BLUE = 5
    PREPARE_PENALTY_YELLOW = 6
    PREPARE_PENALTY_BLUE = 7
    DIRECT_FREE_YELLOW = 8
    DIRECT_FREE_BLUE = 9
    INDIRECT_FREE_YELLOW = 10  # deprecated
    INDIRECT_FREE_BLUE = 11  # deprecated
    TIMEOUT_YELLOW = 12
    TIMEOUT_BLUE = 13
    GOAL_YELLOW = 14  # deprecated
    GOAL_BLUE = 15  # deprecated
    BALL_PLACEMENT_YELLOW = 16
    BALL_PLACEMENT_BLUE = 17

    @staticmethod
    def from_id(command_id: int):
        for command in RefereeCommand:
            if (
                command.value == command_id
            ):  # Check the enum's value (which is the command_id)
                return command
        raise ValueError(f"Invalid referee command ID: {command_id}")
