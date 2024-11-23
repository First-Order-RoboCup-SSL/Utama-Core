class TeamInfo:
    """
    Class containing information about a team.
    """

    def __init__(
        self,
        name: str,
        score: int = 0,
        red_cards: int = 0,
        yellow_cards: int = 0,
        timeouts: int = 0,
        timeout_time: int = 0,
    ):
        # The team's name (empty string if operator has not typed anything).
        self.name = name
        # The number of goals scored by the team during normal play and overtime.
        self.score = score
        # The number of red cards issued to the team since the beginning of the
        # game.
        self.red_cards = red_cards
        # The total number of yellow cards ever issued to the team.
        self.yellow_cards = yellow_cards
        # The number of timeouts this team can still call.
        # If in a timeout right now, that timeout is excluded.
        self.timeouts = timeouts
        # The number of microseconds of timeout this team can use.
        self.timeout_time = timeout_time

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
        self.yellow_cards = packet.yellow_cards
        self.timeouts_left = packet.timeouts
        self.timeout_time = packet.timeout_time

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
