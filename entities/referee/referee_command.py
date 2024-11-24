class RefereeCommand:
    """
    Class representing a referee command.
    """

    def __init__(self, command_id: int, name: str):
        self.command_id = command_id
        self.name = name

    def __repr__(self):
        return f"RefereeCommand(id={self.command_id}, name={self.name})"

    @staticmethod
    def from_id(command_id: int):
        command_map = {
            0: "HALT",
            1: "STOP",
            2: "NORMAL_START",
            3: "FORCE_START",
            4: "PREPARE_KICKOFF_YELLOW",
            5: "PREPARE_KICKOFF_BLUE",
            6: "PREPARE_PENALTY_YELLOW",
            7: "PREPARE_PENALTY_BLUE",
            8: "DIRECT_FREE_YELLOW",
            9: "DIRECT_FREE_BLUE",
            10: "INDIRECT_FREE_YELLOW",
            11: "INDIRECT_FREE_BLUE",
            12: "TIMEOUT_YELLOW",
            13: "TIMEOUT_BLUE",
            14: "GOAL_YELLOW",
            15: "GOAL_BLUE",
            16: "BALL_PLACEMENT_YELLOW",
            17: "BALL_PLACEMENT_BLUE",
        }
        name = command_map.get(command_id, "UNKNOWN")
        return RefereeCommand(command_id, name)
