import dataclasses
from typing import Optional

from utama_core.data_processing.refiners.base_refiner import BaseRefiner
from utama_core.entities.data.referee import RefereeData
from utama_core.entities.game.team_info import TeamInfo
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.entities.referee.stage import Stage


class RefereeRefiner(BaseRefiner):
    def __init__(self):
        self._referee_records = []
        self._latest_referee_data: Optional[RefereeData] = None
        self._latest_stage_time_left: float = 0.0

    def refine(self, game_frame, data: Optional[RefereeData]):
        """Process referee data and update the game frame.

        Args:
            game_frame: Current GameFrame object
            data: Referee data to process (None if no referee)

        Returns:
            Updated GameFrame with referee data attached, or the original frame if data is None
        """
        if data is None:
            return game_frame

        # Always track the latest data so live properties (status_message, etc.)
        # stay current even when deduplication skips appending a new record.
        self._latest_referee_data = data
        self._latest_stage_time_left = data.stage_time_left

        # Add to history
        self.add_new_referee_data(data)

        # Return a new GameFrame with referee data injected
        return dataclasses.replace(game_frame, referee=data)

    def add_new_referee_data(self, referee_data: RefereeData) -> None:
        if not self._referee_records or referee_data != self._referee_records[-1]:
            self._referee_records.append(referee_data)

    def source_identifier(self) -> Optional[str]:
        return self._latest_referee_data.source_identifier if self._latest_referee_data else None

    @property
    def last_time_sent(self) -> float:
        return self._latest_referee_data.time_sent if self._latest_referee_data else 0.0

    @property
    def last_time_received(self) -> float:
        return self._latest_referee_data.time_received if self._latest_referee_data else 0.0

    @property
    def last_command(self) -> RefereeCommand:
        return self._latest_referee_data.referee_command if self._latest_referee_data else RefereeCommand.HALT

    @property
    def last_command_timestamp(self) -> float:
        return self._latest_referee_data.referee_command_timestamp if self._latest_referee_data else 0.0

    @property
    def stage(self) -> Stage:
        return self._latest_referee_data.stage if self._latest_referee_data else Stage.NORMAL_FIRST_HALF_PRE

    @property
    def stage_time_left(self) -> float:
        return self._latest_stage_time_left

    @property
    def blue_team(self) -> TeamInfo:
        return (
            self._latest_referee_data.blue_team
            if self._latest_referee_data
            else TeamInfo(
                name="",
                score=0,
                red_cards=0,
                yellow_card_times=[],
                yellow_cards=0,
                timeouts=0,
                timeout_time=0,
                goalkeeper=0,
            )
        )

    @property
    def yellow_team(self) -> TeamInfo:
        return (
            self._latest_referee_data.yellow_team
            if self._latest_referee_data
            else TeamInfo(
                name="",
                score=0,
                red_cards=0,
                yellow_card_times=[],
                yellow_cards=0,
                timeouts=0,
                timeout_time=0,
                goalkeeper=0,
            )
        )

    @property
    def designated_position(self) -> Optional[tuple[float]]:
        return self._latest_referee_data.designated_position if self._latest_referee_data else None

    @property
    def blue_team_on_positive_half(self) -> Optional[bool]:
        return self._latest_referee_data.blue_team_on_positive_half if self._latest_referee_data else None

    @property
    def next_command(self) -> Optional[RefereeCommand]:
        return self._latest_referee_data.next_command if self._latest_referee_data else None

    @property
    def current_action_time_remaining(self) -> Optional[int]:
        return self._latest_referee_data.current_action_time_remaining if self._latest_referee_data else None

    @property
    def last_status_message(self) -> Optional[str]:
        return self._latest_referee_data.status_message if self._latest_referee_data else None

    @property
    def last_next_command(self) -> Optional[RefereeCommand]:
        return self._latest_referee_data.next_command if self._latest_referee_data else None

    @property
    def is_halt(self) -> bool:
        return self.last_command == RefereeCommand.HALT

    @property
    def is_stop(self) -> bool:
        return self.last_command == RefereeCommand.STOP

    @property
    def is_normal_start(self) -> bool:
        return self.last_command == RefereeCommand.NORMAL_START

    @property
    def is_force_start(self) -> bool:
        return self.last_command == RefereeCommand.FORCE_START

    @property
    def is_prepare_kickoff_yellow(self) -> bool:
        return self.last_command == RefereeCommand.PREPARE_KICKOFF_YELLOW

    @property
    def is_prepare_kickoff_blue(self) -> bool:
        return self.last_command == RefereeCommand.PREPARE_KICKOFF_BLUE

    @property
    def is_prepare_penalty_yellow(self) -> bool:
        return self.last_command == RefereeCommand.PREPARE_PENALTY_YELLOW

    @property
    def is_prepare_penalty_blue(self) -> bool:
        return self.last_command == RefereeCommand.PREPARE_PENALTY_BLUE

    @property
    def is_direct_free_yellow(self) -> bool:
        return self.last_command == RefereeCommand.DIRECT_FREE_YELLOW

    @property
    def is_direct_free_blue(self) -> bool:
        return self.last_command == RefereeCommand.DIRECT_FREE_BLUE

    @property
    def is_timeout_yellow(self) -> bool:
        return self.last_command == RefereeCommand.TIMEOUT_YELLOW

    @property
    def is_timeout_blue(self) -> bool:
        return self.last_command == RefereeCommand.TIMEOUT_BLUE

    @property
    def is_ball_placement_yellow(self) -> bool:
        return self.last_command == RefereeCommand.BALL_PLACEMENT_YELLOW

    @property
    def is_ball_placement_blue(self) -> bool:
        return self.last_command == RefereeCommand.BALL_PLACEMENT_BLUE
