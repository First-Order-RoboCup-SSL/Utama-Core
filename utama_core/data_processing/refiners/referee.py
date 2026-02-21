from typing import Optional

from utama_core.data_processing.refiners.base_refiner import BaseRefiner
from utama_core.entities.data.referee import RefereeData
from utama_core.entities.game.team_info import TeamInfo
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.entities.referee.stage import Stage


class RefereeRefiner(BaseRefiner):
    def refine(self, game, data):
        return game

        self._referee_records = []

    def add_new_referee_data(self, referee_data: RefereeData) -> None:
        if not self._referee_records:
            self._referee_records.append(referee_data)
        elif referee_data[1:] != self._referee_records[-1][1:]:
            self._referee_records.append(referee_data)

    def source_identifier(self) -> Optional[str]:
        return self._referee_records[-1].source_identifier if self._referee_records else None

    @property
    def last_time_sent(self) -> float:
        return self._referee_records[-1].time_sent if self._referee_records else 0.0

    @property
    def last_time_received(self) -> float:
        return self._referee_records[-1].time_received if self._referee_records else 0.0

    @property
    def last_command(self) -> RefereeCommand:
        return self._referee_records[-1].referee_command if self._referee_records else RefereeCommand.HALT

    @property
    def last_command_timestamp(self) -> float:
        return self._referee_records[-1].referee_command_timestamp if self._referee_records else 0.0

    @property
    def stage(self) -> Stage:
        return self._referee_records[-1].stage if self._referee_records else Stage.NORMAL_FIRST_HALF_PRE

    @property
    def stage_time_left(self) -> float:
        return self._referee_records[-1].stage_time_left if self._referee_records else 0.0

    @property
    def blue_team(self) -> TeamInfo:
        return (
            self._referee_records[-1].blue_team
            if self._referee_records
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
            self._referee_records[-1].yellow_team
            if self._referee_records
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
        return self._referee_records[-1].designated_position if self._referee_records else None

    @property
    def blue_team_on_positive_half(self) -> Optional[bool]:
        return self._referee_records[-1].blue_team_on_positive_half if self._referee_records else None

    @property
    def next_command(self) -> Optional[RefereeCommand]:
        return self._referee_records[-1].next_command if self._referee_records else None

    @property
    def current_action_time_remaining(self) -> Optional[int]:
        return self._referee_records[-1].current_action_time_remaining if self._referee_records else None

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
