from dataclasses import dataclass


@dataclass(kw_only=True)
class ReplayMetadata:
    """Metadata for a replay session.

    Attributes:
        my_team_is_yellow (bool): Whether the user's team is yellow.
        exp_friendly (int): Number of friendly robots in the replay.
        exp_enemy (int): Number of enemy robots in the replay.
    """

    my_team_is_yellow: bool
    exp_friendly: int
    exp_enemy: int
