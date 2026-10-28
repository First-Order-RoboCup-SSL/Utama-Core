# ruff: noqa: E402
"""examples/passing_rally.py — our two robots pass their way up the field and shoot into the empty goal.

Run:
    pixi run python examples/passing_rally.py            # real robots (MODE in pass_receive.py)
    pixi run python examples/passing_rally.py --mode rsim
    pixi run python examples/passing_rally.py --mode grsim   # grSim must be running
    # the dashboard is at http://localhost:8080 (--port to change): Force Start to play, Halt to freeze

What it does:
  - `pass_receive.py`'s pair, going somewhere: the receiver gets open `ADVANCE_M` nearer the
    `ATTACK_GOAL` than the ball, in the lane (`LANE_Y_M` off the centre line) across from it, so
    the passes zig-zag up the field. Once the ball is within `SHOOT_RANGE_M` of that goal line, the robot on
    it shoots at the goal's centre instead, without waiting for its partner, and the partner
    trails `ADVANCE_M` behind the ball for a rebound, out of the shot's way.
  - A shot rolling into the goal mouth is left alone: the partner does not step onto it as if it
    were a pass.
  - Start a rally with the ball at the other end, at least 0.5 m from its end line: the passer
    lines up behind the ball, and within 0.27 m of a wall (the planner's `OBSTACLE_CLEARANCE`)
    the planner steers it off the wall instead (1 of 21 rsim rallies stalled so). A goal (or a
    shot out) stops play: put the ball back and Force Start. The referee's scoreboard counts the goals; the Live tab shows each
    robot's role (pass, shoot, receive, get_open) under its tactic.
  - Robots and the readiness checks are `pass_receive.py`'s ("Test-day
    configuration" there); the field and the referee rules are `shooting_drill.py`'s. Both goals
    need not be in view, only the one attacked and the run-up to it.

On the real robots the fixed-power kick (about 2.1 m/s) easily reaches an empty goal from
`SHOOT_RANGE_M`; on the 3 x 2.25 m lab field a rally from one end is four passes and a shot
(23 of 23 rsim rallies scored with the kick set to 2.1 m/s). rsim's own kick is 5 m/s: the
receiver misses about half those passes and the ball rolls out.
"""

import sys
from pathlib import Path

# Run as `pixi run python examples/<this>.py`: utama_core is not installed, so put the repo on the path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import math
from typing import Optional

from pass_receive import (
    _MIN_PASS_M,
    _ROLLING_MPS,
    DASHBOARD_PORT,
    MODE,
    PassReceiveTactic,
    _partner,
    build_runner,
    get_open_spot,
    outside_boxes,
    partner_ready,
    pass_role,
)

from utama_core.config.physical_constants import BALL_RADIUS
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game
from utama_core.entities.game.robot import Robot
from utama_core.run import StrategyRunner
from utama_core.shared.pass_and_score_geometry import clamp_to_field

# ---------------------------------------------------------------------------
# Test-day configuration (the robots and the pass distance are pass_receive.py's)
# ---------------------------------------------------------------------------

ATTACK_GOAL = "left"  # the goal both robots attack, as the field is drawn: "right" (+x) or "left"

# How much nearer the goal than the ball the receiver gets open, in metres.
ADVANCE_M = 0.6

# The two lanes the robots take turns in, this far either side of the field's centre line. With
# ADVANCE_M 0.6 a pass from one lane to the other is about 1.2 m. The receiver used to stand
# PASS_DISTANCE from the ball instead, about 1 m sideways: from a ball near the centre that put
# it 0.25 m from the sideline, and a pass it did not quite meet rolled out (3 of 22 rsim rounds).
LANE_Y_M = 0.5

# The robot on the ball shoots from this close to the goal line (along the field), in metres.
SHOOT_RANGE_M = 1.0

# ---------------------------------------------------------------------------

_FIELD_MARGIN_M = 0.25  # how far inside the field lines the receiver stands (as pass_receive.py)
_CENTRE_M = 0.05  # a ball this near the centre line has no side: the receiver keeps its own
_SET_M = 0.2  # the partner is at its upfield spot within this


def _attack_sign() -> float:
    if ATTACK_GOAL not in ("right", "left"):
        raise ValueError(f"ATTACK_GOAL must be 'right' or 'left', got {ATTACK_GOAL!r}")
    return 1.0 if ATTACK_GOAL == "right" else -1.0


def goal_centre(game: Game) -> Vector2D:
    return Vector2D(_attack_sign() * game.field.half_length, 0.0)


def in_shooting_range(game: Game, point: Vector2D) -> bool:
    return abs(goal_centre(game).x - point.x) <= SHOOT_RANGE_M


def shot_on_goal(game: Game) -> bool:
    """The ball is rolling towards the attacked goal and its path crosses the goal line in the mouth."""
    goal = goal_centre(game)
    v = game.ball.v
    if math.hypot(v.x, v.y) < _ROLLING_MPS or v.x * _attack_sign() <= 0:
        return False
    ball = game.ball.p
    y_at_line = ball.y + v.y * (goal.x - ball.x) / v.x
    return abs(y_at_line) <= game.field.half_goal_width + BALL_RADIUS


def rally_role(game: Game, robot_id: int) -> str:
    """`pass_role`, but the robot on the ball in shooting range shoots ("shoot"), and nobody steps
    onto a shot rolling into the goal ("wait" rather than "receive")."""
    role = pass_role(game, robot_id)
    if role == "receive" and shot_on_goal(game):
        return "wait"
    if role == "pass" and in_shooting_range(game, game.ball.p.to_2d()):
        return "shoot"
    return role


def upfield_spot(game: Game, position: Vector2D) -> Optional[Vector2D]:
    """Where a robot at `position` gets open: `ADVANCE_M` nearer the goal than the ball (behind it
    once the ball is in shooting range, to trail the shot), in the lane across the centre line
    from the ball (`LANE_Y_M`); inside the field and out of both boxes. None where that leaves
    too short a pass (a corner)."""
    ball = game.ball.p.to_2d()
    ahead = -ADVANCE_M if in_shooting_range(game, ball) else ADVANCE_M
    side = -math.copysign(1.0, ball.y) if abs(ball.y) > _CENTRE_M else math.copysign(1.0, position.y - ball.y)
    spot = Vector2D(ball.x + _attack_sign() * ahead, side * LANE_Y_M)
    spot = outside_boxes(game, clamp_to_field(spot, game, margin=_FIELD_MARGIN_M))
    if spot.distance_to(ball) < _MIN_PASS_M:
        return None
    # A coarse grid, as `get_open_spot`: the robot's own drift does not move its target every tick.
    return Vector2D(round(spot.x / 0.05) * 0.05, round(spot.y / 0.05) * 0.05)


def rally_spot(game: Game, robot_id: int) -> Vector2D:
    """`upfield_spot`, or `get_open_spot` where there is none."""
    spot = upfield_spot(game, game.friendly_robots[robot_id].p)
    return get_open_spot(game, robot_id) if spot is None else spot


def partner_set(game: Game, partner: Robot) -> bool:
    """`partner_ready`, and at its upfield spot: a partner still and facing the ball where it
    stood got the ball straight back, and the rally went back and forth without going up the
    field (rsim, 2026-10-28)."""
    if not partner_ready(game, partner):
        return False
    spot = upfield_spot(game, partner.p)
    return spot is None or partner.p.distance_to(spot) <= _SET_M


class PassingRallyTactic(PassReceiveTactic):
    """One robot of the rally: `PassReceiveTactic` with the receiver getting open upfield and the
    robot on the ball shooting once in range. No memory: the role is re-read every tick."""

    def role(self, game: Game, robot_id: int) -> str:
        return rally_role(game, robot_id)

    def open_spot(self, game: Game, robot_id: int) -> Vector2D:
        return rally_spot(game, robot_id)

    def target(self, game: Game, robot_id: int) -> tuple[Vector2D, bool]:
        if in_shooting_range(game, game.ball.p.to_2d()):
            return goal_centre(game), False  # the goal is empty: no one to wait for
        return _partner(game).p, True

    def partner_set(self, game: Game, partner: Robot) -> bool:
        return partner_set(game, partner)


def build_rally_runner(mode: str, port: int = DASHBOARD_PORT) -> StrategyRunner:
    return build_runner(mode, port, tactic=PassingRallyTactic, name="passing_rally")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("real", "rsim", "grsim"), default=MODE)
    parser.add_argument("--port", type=int, default=DASHBOARD_PORT, help="dashboard port (default %(default)s)")
    args = parser.parse_args()
    build_rally_runner(args.mode, args.port).run()


if __name__ == "__main__":
    main()
