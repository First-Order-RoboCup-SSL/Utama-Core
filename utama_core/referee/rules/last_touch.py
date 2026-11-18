"""Colour-blind last-touch attribution shared by referee rules.

Both ``OutOfBoundsRule`` and ``BallSpeedRule`` need to know which team
last touched the ball to assign a free kick to the *non*-touching team.
Historically each rule implemented its own tracker, with the same two
colour asymmetries baked in:

1. Friendly robots' ``has_ball`` (IR-backed) was consulted first and the
   scan short-circuited on any friendly touch — the enemy's contact flag
   was never read at all, only a ≤0.15 m proximity fallback. In a scrum
   where both teams are touching, the friendly side always won the
   attribution.
2. If no touch could be detected at all, the free kick defaulted to
   YELLOW — a hardcoded colour preference inside the referee.

``infer_last_touch_team`` replaces both with a single symmetric scan in
which the two teams are processed by identical rules and every
tie-break is geometric (colour-blind).

The "closest robot at any distance" inference only runs when there is
*no prior attribution at all* — it mirrors the SSL GameController's
last-touch fallback for a ball exiting after a hard kick from a
placement (the kicker is often beyond any fixed touch radius by the
time the ball crosses the line). When a prior attribution exists it is
persisted instead, so the touch recorded during the kicker's contact
ticks cannot be corrupted by unrelated robots standing near the dead
ball after it has left the field.

A robot merely *near* the ball is not a toucher: the proximity fallback
(no contact flag on either side) also requires the ball's velocity to have
changed since the previous frame, as TIGERs' AutoReferee requires a bot to sit
on the ball's own line of travel (``BotLastTouchedBallCalculator``) rather than
just beside it. Without it a robot standing within 0.15 m of a ball it never
touched took the attribution, and the out-of-bounds free kick went to the wrong
team. Contact flags stay authoritative and are never second-guessed.

Requires the caller's frame to carry contact data for both teams —
``RobotInfoRefiner`` fills enemy ``has_ball`` from the sim's contact
physics (see ``data_processing/refiners/robot_info.py``).
"""

from __future__ import annotations

import math
from typing import Optional

from utama_core.entities.game.game_frame import GameFrame

# Robots closer than this are treated as confirmed touchers when neither
# side's contact flag is set.
_PROXIMITY_TOUCH_DIST = 0.15  # metres

# A nearby robot counts as having touched the ball only if the ball's velocity
# moved by more than this since the previous frame. A ball rolling freely loses
# about 0.011 m/s per frame to friction (the 75th percentile of frames with a
# robot near the ball and no contact flag, over saved rsim matches), so this sits
# above that floor and well below the change a push or kick makes. On saved
# matches, 2 of the 14 ball exits attributed by proximity had no change above it
# at the supposed touch (1 at 0.02, 3 at 0.1 or 0.2).
_TOUCH_BALL_VELOCITY_CHANGE = 0.05  # m/s


def touching_team(game_frame: GameFrame, previous_ball_v: Optional[tuple[float, float]] = None) -> Optional[bool]:
    """Return which team touches the ball in this frame: True = friendly,
    False = enemy, None = no robot does (or the frame has no ball).

    Args:
        game_frame: The frame to judge.
        previous_ball_v: The ball's (vx, vy) in the previous frame, or None if
            unknown. The proximity fallback only credits a nearby robot when the
            ball's velocity changed from this; with it unknown a nearby robot is
            no evidence of a touch.
    """
    ball = game_frame.ball
    if ball is None:
        return None

    bx, by = ball.p.x, ball.p.y

    friendly_touchers = [r for r in game_frame.friendly_robots.values() if r.has_ball]
    enemy_touchers = [r for r in game_frame.enemy_robots.values() if r.has_ball]

    if friendly_touchers and not enemy_touchers:
        return True
    if enemy_touchers and not friendly_touchers:
        return False
    if friendly_touchers and enemy_touchers:
        # Scrum (both teams in contact): the robot closest to the ball
        # among the confirmed touchers is the one actually owning the
        # contact — geometry, not colour, decides.
        closest_friendly = min(friendly_touchers, key=lambda r: math.hypot(r.p.x - bx, r.p.y - by))
        closest_enemy = min(enemy_touchers, key=lambda r: math.hypot(r.p.x - bx, r.p.y - by))
        return math.hypot(closest_friendly.p.x - bx, closest_friendly.p.y - by) <= math.hypot(
            closest_enemy.p.x - bx, closest_enemy.p.y - by
        )

    # No contact flags: a robot within touch distance that the ball's
    # velocity change backs up is a confirmed toucher — closest of either
    # team wins.
    closest, closest_team = _closest_robot(game_frame)
    if closest <= _PROXIMITY_TOUCH_DIST and previous_ball_v is not None:
        if math.hypot(ball.v.x - previous_ball_v[0], ball.v.y - previous_ball_v[1]) > _TOUCH_BALL_VELOCITY_CHANGE:
            return closest_team
    return None


def touching_robot(game_frame: GameFrame, is_friendly: bool) -> Optional[int]:
    """The id of the robot of that team touching the ball, for a frame where
    `touching_team` named the team: its robot in contact closest to the ball, or with
    no contact flag (the proximity fallback) its robot closest to the ball."""
    ball = game_frame.ball
    robots = list((game_frame.friendly_robots if is_friendly else game_frame.enemy_robots).values())
    if ball is None or not robots:
        return None
    candidates = [r for r in robots if r.has_ball] or robots
    return min(candidates, key=lambda r: math.hypot(r.p.x - ball.p.x, r.p.y - ball.p.y)).id


def infer_last_touch_team(
    game_frame: GameFrame,
    previous: Optional[bool] = None,
    previous_ball_v: Optional[tuple[float, float]] = None,
) -> Optional[bool]:
    """Return which team last touched the ball.

    Args:
        game_frame: The frame to judge (ball must be present).
        previous: The caller's prior attribution (True = friendly,
            False = enemy, None = unknown). Persisted when the frame
            contains no new evidence, so attribution is stable while the
            ball is dead.
        previous_ball_v: See `touching_team`.

    Returns:
        True if friendly last touched, False if enemy, None if unknown
        (no evidence ever and no robots to infer from).
    """
    if game_frame.ball is None:
        return previous

    team = touching_team(game_frame, previous_ball_v)
    if team is not None:
        return team

    # No new evidence: persist the prior attribution. If there never was
    # one, infer from the closest robot at any distance (GameController
    # style) rather than defaulting to a fixed colour.
    if previous is not None:
        return previous
    return _closest_robot(game_frame)[1]


def _closest_robot(game_frame: GameFrame) -> tuple[float, Optional[bool]]:
    """Distance from the ball to the closest robot of either team, and its team."""
    bx, by = game_frame.ball.p.x, game_frame.ball.p.y
    closest = math.inf
    closest_team: Optional[bool] = None
    for robot in game_frame.friendly_robots.values():
        d = math.hypot(robot.p.x - bx, robot.p.y - by)
        if d < closest:
            closest, closest_team = d, True
    for robot in game_frame.enemy_robots.values():
        d = math.hypot(robot.p.x - bx, robot.p.y - by)
        if d < closest:
            closest, closest_team = d, False
    return closest, closest_team
