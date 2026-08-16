"""DoubleTouchRule: the kicker of a restart (free kick, kickoff, penalty) may
not touch the ball again until another robot has touched it.

Scoped narrowly to the restart-kick window on purpose — SSL's double-touch
rule only applies to that one kick, not to open-play dribbling. A robot
releasing and reacquiring its own ball during normal possession (exactly
what `tactics/dribble.py`'s `DribbleTactic` does) is completely legal and
must not be flagged.
"""

from __future__ import annotations

from typing import Optional

from utama_core.custom_referee.geometry import RefereeGeometry
from utama_core.custom_referee.rules.base_rule import BaseRule, RuleViolation
from utama_core.entities.game.game_frame import GameFrame
from utama_core.entities.referee.referee_command import RefereeCommand

# Commands whose transition into NORMAL_START starts a restart-kick window.
_RESTART_COMMANDS = frozenset(
    {
        RefereeCommand.DIRECT_FREE_YELLOW,
        RefereeCommand.DIRECT_FREE_BLUE,
        RefereeCommand.PREPARE_KICKOFF_YELLOW,
        RefereeCommand.PREPARE_KICKOFF_BLUE,
        RefereeCommand.PREPARE_PENALTY_YELLOW,
        RefereeCommand.PREPARE_PENALTY_BLUE,
    }
)

# (is_friendly, robot_id) identifies a robot uniquely across both teams.
RobotKey = tuple[bool, int]


class DoubleTouchRule(BaseRule):
    """Detects the restart kicker touching the ball twice in a row with no
    intervening touch by a different robot.

    Armed only for the window starting when play resumes (`NORMAL_START`)
    immediately after a restart command (`DIRECT_FREE_*`, `PREPARE_KICKOFF_*`,
    `PREPARE_PENALTY_*`) and ending the moment any *other* robot touches the
    ball (a legal pass/interception — double-touch can no longer apply to
    that kick) or the game leaves active play. Outside that window the rule
    is dormant: ordinary dribbling, releasing, and reacquiring the ball in
    open play is unaffected.

    A "touch" is a rising edge of `has_ball` (False → True), not the
    continuous True while dribbling. The kicker is whichever robot registers
    the first touch after arming — not assumed in advance, since who
    actually takes the kick can vary by scenario.

    `has_ball` is IR-backed for friendly robots and a positional heuristic
    for enemy robots (see `entities/game/robot.py`) — enemy-side detection
    inherits that heuristic's imprecision, same caveat as
    `OutOfBoundsRule`'s last-touch fallback.
    """

    def __init__(self) -> None:
        self._prev_command: Optional[RefereeCommand] = None
        self._armed = False
        self._kicker: Optional[RobotKey] = None
        self._had_ball_last_frame: set[RobotKey] = set()

    def check(
        self,
        game_frame: GameFrame,
        geometry: RefereeGeometry,
        current_command: RefereeCommand,
    ) -> Optional[RuleViolation]:
        if current_command == RefereeCommand.NORMAL_START and self._prev_command in _RESTART_COMMANDS:
            self._armed = True
            self._kicker = None
            self._had_ball_last_frame = set()
        elif current_command != RefereeCommand.NORMAL_START:
            self._armed = False
            self._kicker = None

        self._prev_command = current_command

        if not self._armed:
            self._had_ball_last_frame = set()
            return None

        touching_now: set[RobotKey] = set()
        for robot in game_frame.friendly_robots.values():
            if robot.has_ball:
                touching_now.add((True, robot.id))
        for robot in game_frame.enemy_robots.values():
            if robot.has_ball:
                touching_now.add((False, robot.id))

        fresh_touches = touching_now - self._had_ball_last_frame
        self._had_ball_last_frame = touching_now

        violation: Optional[RuleViolation] = None
        for toucher in fresh_touches:
            if self._kicker is None:
                self._kicker = toucher
            elif toucher == self._kicker:
                violation = self._violation_for(toucher, game_frame)
                self._armed = False
                self._kicker = None
            else:
                # A different robot touched it — legal, window closes.
                self._armed = False
                self._kicker = None

        return violation

    def reset(self) -> None:
        # Deliberately does NOT clear _prev_command — reset() fires on every
        # command transition (see CustomReferee.step()), including the very
        # transition into NORMAL_START this rule needs to detect. Clearing
        # _prev_command here would make it impossible to ever observe the
        # RESTART_COMMAND -> NORMAL_START edge.
        self._armed = False
        self._kicker = None
        self._had_ball_last_frame = set()

    def reset_for_new_episode(self) -> None:
        self.reset()
        self._prev_command = None

    def _violation_for(self, kicker: RobotKey, game_frame: GameFrame) -> RuleViolation:
        kicker_is_friendly, _ = kicker
        my_team_is_yellow = game_frame.my_team_is_yellow
        if kicker_is_friendly:
            next_cmd = RefereeCommand.DIRECT_FREE_BLUE if my_team_is_yellow else RefereeCommand.DIRECT_FREE_YELLOW
        else:
            next_cmd = RefereeCommand.DIRECT_FREE_YELLOW if my_team_is_yellow else RefereeCommand.DIRECT_FREE_BLUE

        return RuleViolation(
            rule_name="double_touch",
            suggested_command=RefereeCommand.STOP,
            next_command=next_cmd,
            status_message="Double touch",
        )
