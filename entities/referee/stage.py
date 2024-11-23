class Stage:
    """
    Class representing a game stage.
    """

    def __init__(self, stage_id: int, name: str):
        self.stage_id = stage_id
        self.name = name

    def __repr__(self):
        return f"Stage(id={self.stage_id}, name={self.name})"

    @staticmethod
    def from_id(stage_id: int):
        stage_map = {
            0: "NORMAL_FIRST_HALF_PRE",
            1: "NORMAL_FIRST_HALF",
            2: "NORMAL_HALF_TIME",
            3: "NORMAL_SECOND_HALF_PRE",
            4: "NORMAL_SECOND_HALF",
            5: "EXTRA_TIME_BREAK",
            6: "EXTRA_FIRST_HALF_PRE",
            7: "EXTRA_FIRST_HALF",
            8: "EXTRA_HALF_TIME",
            9: "EXTRA_SECOND_HALF_PRE",
            10: "EXTRA_SECOND_HALF",
            11: "PENALTY_SHOOTOUT_BREAK",
            12: "PENALTY_SHOOTOUT",
            13: "POST_GAME",
        }
        name = stage_map.get(stage_id, "UNKNOWN")
        return Stage(stage_id, name)
