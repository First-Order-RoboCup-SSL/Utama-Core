from typing import Optional, List


class TeamInfo:
    """
    Class containing information about a team.
    """

    def __init__(
        self,
        name: str,
        score: int = 0,
        red_cards: int = 0,
        yellow_card_times: List[int] = [],
        yellow_cards: int = 0,
        timeouts: int = 0,
        timeout_time: int = 0,
        goalkeeper: int = 0,
        foul_counter: Optional[int] = None,
        ball_placement_failures: Optional[int] = None,
        can_place_ball: Optional[bool] = None,
        max_allowed_bots: Optional[int] = None,
        bot_substitution_intent: Optional[bool] = None,
        ball_placement_failures_reached: Optional[bool] = None,
        bot_substitution_allowed: Optional[bool] = None,
        bot_substitutions_left: Optional[int] = None,
        bot_substitution_time_left: Optional[int] = None,
    ):
        # The team's name (empty string if operator has not typed anything).
        self.name = name
        # The number of goals scored by the team during normal play and overtime.
        self.score = score
        # The number of red cards issued to the team since the beginning of the game.
        self.red_cards = red_cards
        # The amount of time (in microseconds) left on each yellow card issued to the team.
        self.yellow_card_times = yellow_card_times
        # The total number of yellow cards ever issued to the team.
        self.yellow_cards = yellow_cards
        # The number of timeouts this team can still call.
        # If in a timeout right now, that timeout is excluded.
        self.timeouts = timeouts
        # The number of microseconds of timeout this team can use.
        self.timeout_time = timeout_time
        # The pattern number of this team's goalkeeper.
        self.goalkeeper = goalkeeper
        # The total number of countable fouls that act towards yellow cards
        self.foul_counter = foul_counter
        # The number of consecutive ball placement failures of this team
        self.ball_placement_failures = ball_placement_failures
        # Indicate if the team is able and allowed to place the ball
        self.can_place_ball = can_place_ball
        # The maximum number of bots allowed on the field based on division and cards
        self.max_allowed_bots = max_allowed_bots
        # The team has submitted an intent to substitute one or more robots at the next chance
        self.bot_substitution_intent = bot_substitution_intent
        # Indicate if the team reached the maximum allowed ball placement failures and is thus not allowed to place the ball anymore
        self.ball_placement_failures_reached = ball_placement_failures_reached
        # The team is allowed to substitute one or more robots currently
        self.bot_substitution_allowed = bot_substitution_allowed
        # The number of bot substitutions left by the team in this halftime
        self.bot_substitutions_left = bot_substitutions_left
        # The number of microseconds left for current bot substitution
        self.bot_substitution_time_left = bot_substitution_time_left

    def __repr__(self):
        return (
            f"Team Name         : {self.name}\n"
            f"Score             : {self.score}\n"
            f"Red Cards         : {self.red_cards}\n"
            f"Yellow Cards      : {self.yellow_cards}\n"
            f"Timeouts Left     : {self.timeouts}\n"
            f"Timeout Time      : {self.timeout_time} usn"
        )

    def parse_referee_packet(self, packet):
        """
        Parses the SSL_Referee_TeamInfo packet and updates the team information.

        Args:
            packet (SSL_Referee_TeamInfo): The packet containing team information.
        """
        self.name = packet.name
        self.score = packet.score
        self.red_cards = packet.red_cards
        self.yellow_card_times = list(packet.yellow_card_times)
        self.yellow_cards = packet.yellow_cards
        self.timeouts = packet.timeouts
        self.timeout_time = packet.timeout_time
        self.goalkeeper = packet.goalkeeper
        self.foul_counter = packet.foul_counter
        self.ball_placement_failures = packet.ball_placement_failures
        self.can_place_ball = packet.can_place_ball
        self.max_allowed_bots = packet.max_allowed_bots
        self.bot_substitution_intent = packet.bot_substitution_intent
        self.ball_placement_failures_reached = packet.ball_placement_failures_reached
        self.bot_substitution_allowed = packet.bot_substitution_allowed
        self.bot_substitutions_left = packet.bot_substitutions_left
        self.bot_substitution_time_left = packet.bot_substitution_time_left

    def increment_score(self):
        self.score += 1

    def increment_red_cards(self):
        self.red_cards += 1

    def increment_yellow_cards(self):
        self.yellow_cards += 1

    def decrement_timeouts(self):
        if self.timeouts > 0:
            self.timeouts -= 1

    def add_timeout_time(self, time: int):
        self.timeout_time += time
