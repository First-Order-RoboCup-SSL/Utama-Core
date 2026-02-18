from typing import List, NamedTuple, Optional, Tuple

from utama_core.entities.game.team_info import TeamInfo
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.entities.referee.stage import Stage


class RefereeData(NamedTuple):
    """Namedtuple for referee data."""

    source_identifier: Optional[str]
    time_sent: float
    time_received: float
    referee_command: RefereeCommand
    referee_command_timestamp: float
    stage: Stage
    stage_time_left: float
    blue_team: TeamInfo
    yellow_team: TeamInfo
    designated_position: Optional[Tuple[float]] = None

    # Information about the direction of play.
    # True, if the blue team will have it's goal on the positive x-axis of the ssl-vision coordinate system.
    # Obviously, the yellow team will play on the opposite half.
    blue_team_on_positive_half: Optional[bool] = None

    # The command that will be issued after the current stoppage and ball placement to continue the game.
    next_command: Optional[RefereeCommand] = None

    # The time in microseconds that is remaining until the current action times out
    # The time will not be reset. It can get negative.
    # An autoRef would raise an appropriate event, if the time gets negative.
    # Possible actions where this time is relevant:
    #  * free kicks
    #  * kickoff, penalty kick, force start
    #  * ball placement
    current_action_time_remaining: Optional[int] = None

    # All game events detected since the last RUNNING state (e.g. foul type, ball-out side).
    # Stored as raw protobuf GameEvent objects. Cleared when the game resumes.
    # Useful for logging and future decision-making; not required for basic compliance.
    game_events: List = []

    # Meta information about the match type:
    # 0 = UNKNOWN_MATCH, 1 = GROUP_PHASE, 2 = ELIMINATION_PHASE, 3 = FRIENDLY
    match_type: int = 0

    # Human-readable message from the referee UI (e.g. reason for a stoppage).
    status_message: Optional[str] = None

    def __eq__(self, other):
        if not isinstance(other, RefereeData):
            return NotImplemented
        # game_events, match_type, and status_message are intentionally excluded
        # from equality so they do not trigger spurious re-records in RefereeRefiner.
        return (
            self.stage == other.stage
            and self.referee_command == other.referee_command
            and self.referee_command_timestamp == other.referee_command_timestamp
            and self.yellow_team == other.yellow_team
            and self.blue_team == other.blue_team
            and self.designated_position == other.designated_position
            and self.blue_team_on_positive_half == other.blue_team_on_positive_half
            and self.next_command == other.next_command
            and self.current_action_time_remaining == other.current_action_time_remaining
        )
