# ruff: noqa: E402
"""examples/passing_rally.py — our two robots pass their way up the field and back, for ever.

Run:
    pixi run python examples/passing_rally.py            # real robots (MODE in pass_receive.py)
    pixi run python examples/passing_rally.py --mode rsim
    pixi run python examples/passing_rally.py --mode grsim   # grSim must be running
    # the dashboard is at http://localhost:8080 (--port to change): Force Start to play, Halt to freeze

What it does:
  - `pass_receive.py`'s pair, going somewhere: the receiver gets open `ADVANCE_M` further along
    the field than the ball, in the lane (`LANE_Y_M` off the centre line) across from it, so the
    passes zig-zag along the field. Once the ball is within `TURN_M` of the end line they head
    for, they turn round and zig-zag back to the other end, and so on. No shots: until 2026-10-28
    the rally shot into the empty goal at the end.
  - It starts towards the end further from the ball. Start with the ball at least 0.5 m from an
    end line: the passer lines up behind the ball, and within 0.27 m of a wall (the planner's
    `OBSTACLE_CLEARANCE`) the planner steers it off the wall instead. A pass out stops play: put
    the ball back and Force Start. The Live tab shows each robot's role (pass, receive,
    get_open) under its tactic.
  - Robots and the readiness checks are `pass_receive.py`'s ("Test-day
    configuration" there); the field and the referee rules are `shooting_drill.py`'s.
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
    DASHBOARD_PORT,
    MODE,
    PassReceiveTactic,
    build_runner,
    get_open_spot,
    outside_boxes,
    partner_ready,
)

from utama_core.engine.context import TickContext
from utama_core.engine.tactic import RobotId
from utama_core.entities.data.command import RobotCommand
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game
from utama_core.entities.game.robot import Robot
from utama_core.run import StrategyRunner
from utama_core.shared.pass_and_score_geometry import clamp_to_field

# ---------------------------------------------------------------------------
# Test-day configuration (the robots and the pass distance are pass_receive.py's)
# ---------------------------------------------------------------------------

# How much further along the field than the ball the receiver gets open, in metres.
ADVANCE_M = 0.6

# The two lanes the robots take turns in, this far either side of the field's centre line. With
# ADVANCE_M 0.6 a pass from one lane to the other is about 1.2 m. The receiver used to stand
# PASS_DISTANCE from the ball instead, about 1 m sideways: from a ball near the centre that put
# it 0.25 m from the sideline, and a pass it did not quite meet rolled out (3 of 22 rsim rounds).
LANE_Y_M = 0.5

# The rally turns round once the ball is this near the end line it heads for, in metres. On the
# 3 m lab field that is about 0.7 m short of the centre: two or three passes each way.
TURN_M = 0.8

# ---------------------------------------------------------------------------

_FIELD_MARGIN_M = 0.25  # how far inside the field lines the receiver stands (as pass_receive.py)
_CENTRE_M = 0.05  # a ball this near the centre line has no side: the receiver keeps its own
_SET_M = 0.2  # the partner is at its upfield spot within this


def rally_direction(game: Game, sign: Optional[float]) -> float:
    """Which way the rally heads (+1 towards +x, -1 towards -x) after `sign`: turned round once
    the ball is within `TURN_M` of that end line. None (a new rally) heads for the end further
    from the ball."""
    ball_x = game.ball.p.x
    if sign is None:
        return -math.copysign(1.0, ball_x)
    if sign * ball_x >= game.field.half_length - TURN_M:
        return -sign
    return sign


def upfield_spot(game: Game, position: Vector2D, sign: float) -> Optional[Vector2D]:
    """Where a robot at `position` gets open: `ADVANCE_M` further than the ball the way the rally
    heads (`sign`), in the lane across the centre line from the ball (`LANE_Y_M`); inside the
    field and out of both boxes. None where that leaves too short a pass (a corner)."""
    ball = game.ball.p.to_2d()
    side = -math.copysign(1.0, ball.y) if abs(ball.y) > _CENTRE_M else math.copysign(1.0, position.y - ball.y)
    spot = Vector2D(ball.x + sign * ADVANCE_M, side * LANE_Y_M)
    spot = outside_boxes(game, clamp_to_field(spot, game, margin=_FIELD_MARGIN_M))
    if spot.distance_to(ball) < _MIN_PASS_M:
        return None
    # A coarse grid, as `get_open_spot`: the robot's own drift does not move its target every tick.
    return Vector2D(round(spot.x / 0.05) * 0.05, round(spot.y / 0.05) * 0.05)


def partner_set(game: Game, partner: Robot, sign: float) -> bool:
    """`partner_ready`, and at its upfield spot: a partner still and facing the ball where it
    stood got the ball straight back, and the rally went back and forth without going up the
    field (rsim, 2026-10-28)."""
    if not partner_ready(game, partner):
        return False
    spot = upfield_spot(game, partner.p, sign)
    return spot is None or partner.p.distance_to(spot) <= _SET_M


class PassingRallyTactic(PassReceiveTactic):
    """One robot of the rally: `PassReceiveTactic` with the receiver getting open along the way
    the rally heads. Its only memory is that way (`rally_direction`): both robots read the same
    ball, so they turn round together."""

    def __init__(self) -> None:
        super().__init__()
        self.sign: Optional[float] = None

    def tick(
        self, game: Game, ctx: TickContext, robot_ids: tuple[RobotId, ...], mem: None
    ) -> tuple[dict[RobotId, RobotCommand], None]:
        self.sign = rally_direction(game, self.sign)
        return super().tick(game, ctx, robot_ids, mem)

    def open_spot(self, game: Game, robot_id: int) -> Vector2D:
        spot = upfield_spot(game, game.friendly_robots[robot_id].p, self.sign)
        return get_open_spot(game, robot_id) if spot is None else spot

    def partner_set(self, game: Game, partner: Robot) -> bool:
        return partner_set(game, partner, self.sign)


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
