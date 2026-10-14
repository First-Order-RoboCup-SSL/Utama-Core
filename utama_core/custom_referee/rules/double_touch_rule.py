"""DoubleTouchRule: the kicker of a restart (free kick, kickoff, penalty) may
not touch the ball again until another robot has touched it.

Scoped narrowly to the restart-kick window on purpose — SSL's double-touch
rule only applies to that one kick, not to open-play dribbling. A robot
releasing and reacquiring its own ball during normal possession (exactly
what `tactics/dribble.py`'s `DribbleTactic` does) is completely legal and
must not be flagged.
"""

from __future__ import annotations

import math
from typing import Optional

from utama_core.config.physical_constants import BALL_RADIUS, ROBOT_RADIUS
from utama_core.config.referee_constants import OPPONENT_DEFENSE_AREA_KEEP_DISTANCE
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

_YELLOW_RESTARTS = frozenset(
    {
        RefereeCommand.DIRECT_FREE_YELLOW,
        RefereeCommand.PREPARE_KICKOFF_YELLOW,
        RefereeCommand.PREPARE_PENALTY_YELLOW,
    }
)

# A robot whose centre is this close to the ball's is touching it (1cm over
# robot + ball radius, for vision noise). `has_ball` alone only sees dribbler
# contact, so a deflection or a receive that never engaged the dribbler did not
# close the window, and the kicker's next touch was called a double touch: 63
# of 68 friendly double-touch fouls in the 2026-09-23 round-robin had another
# robot within 0.101-0.114m of the ball first.
_TOUCH_DISTANCE_M = ROBOT_RADIUS + BALL_RADIUS + 0.01

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

    `has_ball` is now filled for both teams by `RobotInfoRefiner`
    (IR/contact for friendly, sim contact physics for enemy — see
    `data_processing/refiners/robot_info.py`), so the rule sees both
    sides' touches symmetrically; an opponent's legal intervening touch
    closes the restart window.
    """

    def __init__(self) -> None:
        self._prev_command: Optional[RefereeCommand] = None
        self._armed = False
        self._kicker: Optional[RobotKey] = None
        self._kicking_team_is_yellow: Optional[bool] = None
        self._had_ball_last_frame: set[RobotKey] = set()

    def check(
        self,
        game_frame: GameFrame,
        geometry: RefereeGeometry,
        current_command: RefereeCommand,
        designated_position: Optional[tuple[float, float]] = None,
    ) -> Optional[RuleViolation]:
        if current_command == RefereeCommand.NORMAL_START and self._prev_command in _RESTART_COMMANDS:
            self._armed = True
            self._kicker = None
            self._kicking_team_is_yellow = self._prev_command in _YELLOW_RESTARTS
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

        if self._kicker is not None and self._other_robot_touching(game_frame):
            # Any other robot's contact, dribbler or not, ends the window.
            self._armed = False
            self._kicker = None
            return None

        violation: Optional[RuleViolation] = None
        for toucher in fresh_touches:
            if self._kicker is None:
                is_friendly, _ = toucher
                if is_friendly != (game_frame.my_team_is_yellow == self._kicking_team_is_yellow):
                    # The defending team touched it first: the kick was taken
                    # (without the kicker's dribbler registering it), so the ball
                    # is in play and there is no kicker left to double-touch.
                    self._armed = False
                    break
                self._kicker = toucher
            elif toucher == self._kicker:
                violation = self._violation_for(toucher, game_frame, geometry)
                self._armed = False
                self._kicker = None
            else:
                # A different robot touched it — legal, window closes.
                self._armed = False
                self._kicker = None

        return violation

    def _other_robot_touching(self, game_frame: GameFrame) -> bool:
        ball = game_frame.ball
        if ball is None:
            return False
        for is_friendly, robots in ((True, game_frame.friendly_robots), (False, game_frame.enemy_robots)):
            for robot in robots.values():
                if (is_friendly, robot.id) == self._kicker:
                    continue
                if math.hypot(robot.p.x - ball.p.x, robot.p.y - ball.p.y) <= _TOUCH_DISTANCE_M:
                    return True
        return False

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

    def _violation_for(self, kicker: RobotKey, game_frame: GameFrame, geometry: RefereeGeometry) -> RuleViolation:
        kicker_is_friendly, kicker_id = kicker
        my_team_is_yellow = game_frame.my_team_is_yellow
        if kicker_is_friendly:
            next_cmd = RefereeCommand.DIRECT_FREE_BLUE if my_team_is_yellow else RefereeCommand.DIRECT_FREE_YELLOW
        else:
            next_cmd = RefereeCommand.DIRECT_FREE_YELLOW if my_team_is_yellow else RefereeCommand.DIRECT_FREE_BLUE

        # Without an explicit designated_position, RuleViolation silently
        # carries over whatever restart position was last recorded --
        # possibly stale/unrelated to this kick, or (per
        # `RefereeGeometry.legal_restart_position`'s docstring) illegal if
        # it happens to sit inside a defense area. Take the restart from
        # the ball's own current position (this is where the double touch
        # happened), projected clear.
        ball = game_frame.ball
        placement = (
            geometry.legal_restart_position(ball.p.x, ball.p.y, OPPONENT_DEFENSE_AREA_KEEP_DISTANCE)
            if ball is not None
            else None
        )
        return RuleViolation(
            rule_name="double_touch",
            suggested_command=RefereeCommand.STOP,
            next_command=next_cmd,
            status_message="Double touch",
            designated_position=placement,
            offending_robots=((kicker_is_friendly == my_team_is_yellow, kicker_id),),
        )
