"""PushingRule: SSL rulebook §8.4.1 "Pushing" (stopping foul).

"A robot pushes an opponent robot if both robots keep contact to the ball
or to each other while the robot exerts force onto the opponent robot,
such that both robots travel towards the opponent robot. If both robots
are pushing each other with similar force, no team is at fault."

Detection is purely position/velocity-based via `robot_contact.py` — not
`has_ball` (see that module's docstring: two robots can be pinned
body-to-body, ball trapped between their fronts, with neither dribbler
ever registering contact; that scenario is exactly the "similar force,
no fault" case this rule needs to recognise, not miss).
"""

from __future__ import annotations

from typing import Optional

from utama_core.custom_referee.geometry import RefereeGeometry
from utama_core.custom_referee.rules.base_rule import BaseRule, RuleViolation
from utama_core.custom_referee.rules.robot_contact import find_robot_pair_contacts
from utama_core.entities.game.game_frame import GameFrame
from utama_core.entities.referee.referee_command import RefereeCommand

_ACTIVE_PLAY_COMMANDS = {
    RefereeCommand.NORMAL_START,
    RefereeCommand.FORCE_START,
}

# A robot must be closing distance at least this fast to count as
# "exerting force" toward the opponent at all — filters out robots that
# are merely stationary-in-contact (e.g. momentarily brushing past) rather
# than actively driving into each other.
_MIN_CLOSING_SPEED = 0.05  # m/s

# "Similar force" tolerance for the no-fault symmetric-push case: when both
# robots' closing speeds are within this margin of each other, neither is
# clearly dominant, so no team is at fault — mirrors this codebase's other
# "don't flip without a real reason" margin-based tie-breaks (e.g.
# FastPathPlanner's DETOUR_SWITCH_MARGIN), just applied to force/contest
# symmetry instead of path-choice stability.
_SIMILAR_FORCE_MARGIN = 0.15  # m/s

# Contact must be sustained this many consecutive ticks before a push is
# actually called — a single-frame graze while both robots are converging
# on a loose ball is normal play, not a foul. Same shape as KeepOutRule's
# violation_persistence_frames.
_DEFAULT_PERSISTENCE_FRAMES = 15


class PushingRule(BaseRule):
    """Detects sustained, force-asymmetric robot-on-robot pushing.

    Sustained contact with both robots closing on each other, tracked per
    (friendly_id, enemy_id) pair so unrelated pairs don't interfere with
    each other's persistence counters. On firing:
      - Clear force asymmetry (one side's closing speed beats the other's
        by more than `similar_force_margin`): the weaker/non-dominant side
        gets the free kick, taken from the ball's position at the moment
        sustained contact was first detected (rulebook: "resumes ... from
        the position where the ball was located when the foul began
        happening"). The dominant side is charged the foul.
      - Similar force (within margin): rulebook says "no team is at
        fault" — awarding either side a free kick would contradict that,
        so this is NOT resolved as a DIRECT_FREE_* to either team.
        Instead, following this codebase's existing precedent for
        no-fault-but-play-must-continue situations (state_machine.py's
        NORMAL_START -> FORCE_START kickoff-timeout auto-advance), it
        stops play and resumes with FORCE_START at the ball's current
        position — no side favoured, `offending_teams=()` so the foul
        counter isn't charged to either team either.
    """

    def __init__(
        self,
        min_closing_speed_mps: float = _MIN_CLOSING_SPEED,
        similar_force_margin_mps: float = _SIMILAR_FORCE_MARGIN,
        persistence_frames: int = _DEFAULT_PERSISTENCE_FRAMES,
    ) -> None:
        self._min_closing_speed = min_closing_speed_mps
        self._similar_force_margin = similar_force_margin_mps
        self._persistence = persistence_frames
        # (friendly_id, enemy_id) -> consecutive-tick count of a qualifying push.
        self._push_counts: dict[tuple[int, int], int] = {}
        # (friendly_id, enemy_id) -> ball position snapshot when this pair's
        # count started (so the eventual free kick/restart uses the ball's
        # position when the foul *began*, not wherever it drifted to by the
        # time persistence_frames is reached).
        self._push_start_ball_pos: dict[tuple[int, int], tuple[float, float]] = {}

    def check(
        self,
        game_frame: GameFrame,
        geometry: RefereeGeometry,
        current_command: RefereeCommand,
        designated_position: Optional[tuple[float, float]] = None,
    ) -> Optional[RuleViolation]:
        if current_command not in _ACTIVE_PLAY_COMMANDS:
            self._push_counts.clear()
            self._push_start_ball_pos.clear()
            return None

        ball = game_frame.ball
        active_pairs: set[tuple[int, int]] = set()
        violation: Optional[RuleViolation] = None

        for contact in find_robot_pair_contacts(game_frame):
            key = (contact.friendly.id, contact.enemy.id)
            friendly_closing = contact.closing_speed_friendly_into_enemy
            enemy_closing = contact.closing_speed_enemy_into_friendly

            # Both robots must be genuinely driving into each other, not
            # just resting in contact or separating.
            if friendly_closing < self._min_closing_speed or enemy_closing < self._min_closing_speed:
                continue

            active_pairs.add(key)
            count = self._push_counts.get(key, 0) + 1
            self._push_counts[key] = count
            if key not in self._push_start_ball_pos and ball is not None:
                self._push_start_ball_pos[key] = (ball.p.x, ball.p.y)

            if count < self._persistence:
                continue

            # Sustained push confirmed — resolve fault before returning
            # (only one violation can be reported per tick; first pair to
            # reach persistence wins, consistent with every other rule in
            # this package returning on first match).
            ball_pos = self._push_start_ball_pos.get(key)
            self._push_counts.pop(key, None)
            self._push_start_ball_pos.pop(key, None)

            force_diff = friendly_closing - enemy_closing
            if abs(force_diff) <= self._similar_force_margin:
                violation = RuleViolation(
                    rule_name="pushing",
                    suggested_command=RefereeCommand.STOP,
                    next_command=RefereeCommand.FORCE_START,
                    status_message="Pushing — similar force, no team at fault",
                    designated_position=ball_pos,
                    offending_teams=(),
                )
            else:
                my_team_is_yellow = game_frame.my_team_is_yellow
                # The dominant (higher closing-speed) side is the pusher.
                friendly_is_pusher = force_diff > 0
                pusher_is_yellow = friendly_is_pusher == my_team_is_yellow
                next_cmd = RefereeCommand.DIRECT_FREE_BLUE if pusher_is_yellow else RefereeCommand.DIRECT_FREE_YELLOW
                violation = RuleViolation(
                    rule_name="pushing",
                    suggested_command=RefereeCommand.STOP,
                    next_command=next_cmd,
                    status_message="Pushing foul",
                    designated_position=ball_pos,
                    offending_teams=(pusher_is_yellow,),
                )
            break

        # Any pair that dropped out of qualifying contact this tick resets
        # (contact must be *sustained*, not just accumulated on-and-off).
        for key in list(self._push_counts):
            if key not in active_pairs:
                self._push_counts.pop(key, None)
                self._push_start_ball_pos.pop(key, None)

        return violation

    def reset(self) -> None:
        self._push_counts.clear()
        self._push_start_ball_pos.clear()
