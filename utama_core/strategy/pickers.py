"""Helpers shared by more than one kernel strategy: possession and ball-zone reads, carrier ordering, allocation."""

from __future__ import annotations

import math
from typing import Optional

from utama_core.engine.tactic import RobotId
from utama_core.entities.data.object import TeamType
from utama_core.entities.game import Game
from utama_core.entities.referee.stage import Stage


def _fixed_ratio_picker(attack_id: str, defense_id: str, attack_fraction: float, min_attack: int = 0):
    """Builds a `Partitioner` that ignores game state entirely and splits the
    free pool by a constant fraction — the simplest possible allocation rule,
    useful as a deliberately non-reactive baseline/contrast to the
    possession-edge pickers above. Still respects commitment pinning and
    `available_tactic_ids` exactly like every other `Partitioner`; "fixed"
    only describes the *ratio* decision, not an exemption from the kernel's
    invariants.

    `min_attack`: a floor on the attack slot's robot count whenever attack is
    getting any robots at all. Exists because not every Tactic is
    robot-count-agnostic the way `GiveAndGoTactic`/`LeadAndSupportTactic`
    are — `PassAndShootTactic.tick()` unconditionally reads
    `robot_ids[1]`, so a rounded-down fraction that hands it a single robot
    crashes with an `IndexError` rather than degrading gracefully. This
    picker has no way to know that from the Tactic itself (no
    `min_robots`/`max_robots` declaration exists yet — see design doc §15's
    "explicitly deferred" list), so the caller states the floor explicitly.
    """

    def _partitioner(
        game: Game,
        free_robots: frozenset[RobotId],
        prev_partition: Optional[dict[str, frozenset[RobotId]]],
        available_tactic_ids: frozenset[str],
    ) -> dict[str, frozenset[RobotId]]:
        ordered = _carrier_first(game, free_robots)
        if not ordered:
            return {}

        attack_ok = attack_id in available_tactic_ids
        defense_ok = defense_id in available_tactic_ids

        if not attack_ok:
            return {defense_id: frozenset(ordered)} if defense_ok else {}
        if not defense_ok:
            return {attack_id: frozenset(ordered)}

        attack_count = max(0, min(len(ordered), round(len(ordered) * attack_fraction)))
        if 0 < attack_count < min_attack:
            attack_count = min(len(ordered), min_attack)
        partition = {}
        if attack_count > 0:
            partition[attack_id] = frozenset(ordered[:attack_count])
        if attack_count < len(ordered):
            partition[defense_id] = frozenset(ordered[attack_count:])
        return partition

    return _partitioner


# ---------------------------------------------------------------------------
# Arena strategies — plan-driven multi-tactic teams (2026-08-20 addition)
#
# The barebone factories (default/split_shape/low_block/...) exist to exercise
# the kernel machinery; these three are meant as *playable* teams: every
# posture reads live game state (ball ownership, ball zone), every posture
# has both an attack and a defense answer, and the allocation reacts to the
# game instead of being a fixed constant.
# ---------------------------------------------------------------------------


# Deadzone for `_friendly_closer_to_ball`'s distance comparison. Root-caused
# 2026-08-23: at a genuine tie (mirror-symmetric formations, ball equidistant
# — the normal case at kickoff, and possible any time both teams race a loose
# ball to a near-identical distance), rsim's physics does not resolve the two
# teams' positions with perfect left/right symmetry — traced directly at
# ~0.1-0.3mm off a true mirror after a single tick. A bare `<` comparison
# turns that sub-millimetre noise into a hard, match-shaping tactical branch
# (every caller below picks a materially different attack/press allocation
# based on this one boolean, and none of them revisit the choice once
# committed). `_CLOSER_TO_BALL_MARGIN` is chosen well above that noise floor
# (500-1500x) but well below `ROBOT_RADIUS` (0.09m, the smallest physically
# meaningful separation between two robots converging on the same ball), so a
# real contest between two robots that are genuinely almost equidistant still
# resolves by real distance, not by which one the deadzone happens to favour.
_CLOSER_TO_BALL_MARGIN = 0.05  # metres


def _friendly_closer_to_ball(game: Game) -> Optional[bool]:
    """True if a friendly robot is closer to the ball than every enemy by more
    than `_CLOSER_TO_BALL_MARGIN`; False if an enemy is closer by more than
    that margin; None when the gap is inside the margin (undecided).

    Inside the margin is a genuine near-tie, so the pickers that keep
    hysteresis (`prev_partition`: "did we hold attack last tick") read None as
    "keep what you had", while those without it read `is not True` as "not
    clearly ours", the conservative default. Returning False for a near-tie
    (as this once did) made the sticky pickers drop the ball on a noise-level
    gap, which is exactly what their hysteresis exists to prevent.

    None too when the proximity lookup cannot read the ball's side (ball
    missing or no robots on one side) — callers should fall back to the
    conservative posture in that case.
    """
    # A ball on our dribbler is ours however close an enemy presses — the
    # distance read alone flipped a held ball to "theirs" whenever an enemy
    # brushed within the margin, handing the carrier to the press slot and
    # resetting its attack tactic (counter_flow_vs_high_line_zone, 2026-09-27).
    if any(r.has_ball for r in game.friendly_robots.values()) and not any(
        r.has_ball for r in game.enemy_robots.values()
    ):
        return True
    _friendly_closest, friendly_dist = game.proximity_lookup.closest_to_ball(team_type_filter=TeamType.FRIENDLY)
    _enemy_closest, enemy_dist = game.proximity_lookup.closest_to_ball(team_type_filter=TeamType.ENEMY)
    if friendly_dist is None or enemy_dist is None:
        return None
    # Return literal True/False, never the comparison itself: the proximity
    # lookup returns numpy floats, so a raw comparison is np.bool_, whose
    # `is True` is False — a caller checking `edge is True` would read it as
    # "unknown/losing" forever.
    if friendly_dist < enemy_dist - _CLOSER_TO_BALL_MARGIN:
        return True
    if friendly_dist > enemy_dist + _CLOSER_TO_BALL_MARGIN:
        return False
    return None


def _ball_zone(game: Game) -> str:
    """Where the ball sits along our attacking axis: 'own', 'mid', or 'final'.

    "Final" means the third of the pitch nearest the enemy goal we attack;
    "own" the third our own goal defends. Uses the same
    `own_goal_sign = 1.0 if my_team_is_right else -1.0` convention as
    `strategy/referee/actions.py`.
    """
    half_length = game.field.half_length
    # progress measured from our own goal line toward the enemy goal.
    own_goal_x = (1.0 if game.my_team_is_right else -1.0) * half_length
    attack_dir = -1.0 if game.my_team_is_right else 1.0
    progress = (game.ball.p.to_2d().x - own_goal_x) * attack_dir
    third = 2.0 * half_length / 3.0
    if progress < third:
        return "own"
    if progress < 2.0 * third:
        return "mid"
    return "final"


def _carrier_first(game: Game, free_robots: frozenset[RobotId]) -> list[RobotId]:
    """`free_robots` in id order, except a robot holding the ball goes first.

    `_allocate_ordered` hands the first `primary_n` robots to the ball-side slot;
    by id alone the carrier could land in the off-ball slot instead, holding the
    ball still (clear_press_plus_vs_zone_fluid, 2026-09-28: carrier 4 in "block"
    while "overload" took robots 1 and 2 and waited on it, frozen for 38 s).
    """

    on_ball = _kicker_at_still_ball(game)

    def holding(rid: RobotId) -> bool:
        robot = game.friendly_robots.get(rid)
        return robot is not None and (robot.has_ball or rid == on_ball)

    return sorted(free_robots, key=lambda rid: (not holding(rid), rid))


# A free kick's kicker waits this close without dribbler contact (DirectFreeOursStep's
# kick-ready distance); a ball slower than this is one it can simply take. 0.3 m/s is the
# dead-ball speed `ball_is_loose` and the keeper's retrieval already use: the kicker's own
# approach nudges the ball at ~0.2 m/s, and at the old 0.1 that nudge dropped the kicker to
# an off-ball slot (clear_press_plus_vs_shadow_switch, 2026-09-29: it walked away from its
# unkicked free kick, which never moved the 5 cm to be in play; nobody touched it for 10 s).
_KICK_REACH_M = 0.16


_STILL_BALL_MPS = 0.3


def _kicker_at_still_ball(game: Game) -> Optional[RobotId]:
    """Our robot within kick reach of a still ball, if no enemy is nearer to it.

    overload_press_vs_switch_of_play (2026-09-28): at NORMAL_START of our free kick
    the kicker stood 0.11 m off the ball, went to "switch", and nobody kicked for 10 s.
    """
    ball = getattr(game, "ball", None)
    if ball is None or math.hypot(ball.v.x, ball.v.y) >= _STILL_BALL_MPS or not game.friendly_robots:
        return None

    def dist(robot) -> float:
        return math.hypot(robot.p.x - ball.p.x, robot.p.y - ball.p.y)

    rid, robot = min(game.friendly_robots.items(), key=lambda item: dist(item[1]))
    enemy_nearest = min((dist(e) for e in game.enemy_robots.values()), default=math.inf)
    return rid if dist(robot) <= _KICK_REACH_M and dist(robot) < enemy_nearest else None


def _clearer_first(game: Game, ordered: list[RobotId]) -> list[RobotId]:
    """`ordered` (from `_carrier_first`) with the robot that should clear first.

    The valve gives its one-robot slot to `ordered[0]`. By id alone that was the
    lowest id wherever nobody held the ball, however far from it; the clearer
    should be the robot nearest the ball. A carrier (already first in `ordered`)
    keeps the job: it is the robot with the ball.
    """
    ball = getattr(game, "ball", None)
    if len(ordered) < 2 or ball is None:
        return ordered
    first = game.friendly_robots.get(ordered[0])
    if first is not None and (first.has_ball or ordered[0] == _kicker_at_still_ball(game)):
        return ordered

    def dist(rid: RobotId) -> float:
        robot = game.friendly_robots.get(rid)
        return math.inf if robot is None else math.hypot(robot.p.x - ball.p.x, robot.p.y - ball.p.y)

    nearest = min(ordered, key=lambda rid: (dist(rid), rid))
    return [nearest] + [rid for rid in ordered if rid != nearest]


def _allocate_ordered(
    ordered: list[RobotId],
    primary: str,
    primary_n: int,
    secondary: Optional[str] = None,
) -> dict[str, frozenset[RobotId]]:
    """Split `ordered` into (primary: primary_n, secondary: rest) with coverage guaranteed.

    Only slots already confirmed available by the caller are named here; the
    caller is responsible for passing a `secondary` that is either None (when
    the primary alone may take everyone) or a real available slot id.
    """
    out: dict[str, frozenset[RobotId]] = {}
    if not ordered:
        return out
    if primary_n >= len(ordered):
        out[primary] = frozenset(ordered)
        return out
    if secondary is None:
        # Primary may take everyone (caller chose no backup slot).
        out[primary] = frozenset(ordered)
        return out
    out[primary] = frozenset(ordered[:primary_n])
    out[secondary] = frozenset(ordered[primary_n:])
    return out


def _friendly_score_diff(game: Game) -> Optional[int]:
    """Friendly score minus enemy score, or None if no referee data is present
    (e.g. a match run without a `CustomReferee`/real referee feed) — callers
    should fall back to their score-blind posture in that case, the same
    pattern `_friendly_closer_to_ball` uses for an unreadable ball side.
    """
    referee = game.referee
    if referee is None:
        return None
    friendly_team = referee.yellow_team if game.my_team_is_yellow else referee.blue_team
    enemy_team = referee.blue_team if game.my_team_is_yellow else referee.yellow_team
    return friendly_team.score - enemy_team.score


_LATE_GAME_THRESHOLD_SECONDS = 60.0


def _is_late_in_half(game: Game) -> bool:
    """True once `stage_time_left` is inside the last `_LATE_GAME_THRESHOLD_SECONDS`
    of a live playing half. False (not late) for stoppages, breaks, or any
    stage `stage_time_left` isn't counting down playing time in, and when no
    referee data is present — a picker should not treat "unreadable" as
    "late."
    """
    referee = game.referee
    if referee is None:
        return False
    if referee.stage not in (Stage.NORMAL_FIRST_HALF, Stage.NORMAL_SECOND_HALF):
        return False
    return referee.stage_time_left <= _LATE_GAME_THRESHOLD_SECONDS
