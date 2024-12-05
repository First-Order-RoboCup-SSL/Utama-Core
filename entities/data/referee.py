from collections import namedtuple
from typing import NamedTuple, Optional, Tuple
from enum import Enum

from entities.game.team_info import TeamInfo
from entities.referee.referee_command import RefereeCommand
from entities.referee.stage import Stage


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

    def __eq__(self, other):
        if not isinstance(other, RefereeData):
            return NotImplemented
        return (
            self.stage == other.stage
            and self.referee_command == other.referee_command
            and self.referee_command_timestamp == other.referee_command_timestamp
            and self.yellow_team == other.yellow_team
            and self.blue_team == other.blue_team
            and self.designated_position == other.designated_position
            and self.blue_team_on_positive_half == other.blue_team_on_positive_half
            and self.next_command == other.next_command
            and self.current_action_time_remaining
            == other.current_action_time_remaining
        )
