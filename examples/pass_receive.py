# ruff: noqa: E402
"""examples/pass_receive.py — our two robots pass the ball to each other, back and forth.

Run:
    pixi run python examples/pass_receive.py            # real robots (MODE below)
    pixi run python examples/pass_receive.py --mode rsim
    pixi run python examples/pass_receive.py --mode grsim   # grSim must be running
    # the dashboard is at http://localhost:8080 (--port to change): Force Start to play, Halt to freeze

What it does:
  - Two robots, both ours on one radio. Whoever is nearer the still ball fetches it, turns to
    the other and kicks once the other stands still, `PASS_DISTANCE` away, facing the ball. The
    other meets the rolling ball on its path, dribbler on, and then has the ball: the roles swap
    and it passes back. A pass that misses rolls out; once it slows, the robot nearer the ball
    collects it and passes again. A ball outside the field (or unseen) holds both robots until
    the operator puts it back.
  - Everything to edit for a test day is in the "Test-day configuration" block. The field, the
    ball-sensor/kicker profile type and the referee rules are `shooting_drill.py`'s
    (`FIELD`, `TestRobot`, `RULES`), so the examples agree.
  - Nothing advances on its own: every mode starts in HALT, play starts and stops from the
    dashboard. Ctrl+C sends stop commands to the robots before exiting. The Live tab's Strategy
    panel can stand either robot still ("still") or restart it ("pass_receive"), while halted.
  - The field check on the Live tab must be all green or amber before Force Start, as in
    `shooting_drill.py` (docs/real_robot_setup.md, "Each test day").

Why two teams: the robots wear different pattern colours (yellow 1, blue 0), and a runner team
is one colour: there is no way to map a blue pattern into the yellow team. So each robot is its
own team (`strategy` and `opp_strategy`, as in `shooting_drill.py`), and each runs the
same `PassReceiveTactic`, which reads the other robot (its only "enemy") as its partner. The
two never share state: each decides its role from the same world they both see (who is nearer
the ball, where the ball is rolling, whether the partner is set), so the pass needs no new
coordination machinery. The pieces are the existing ones: the free-kick taker's line-up behind
the ball (`DirectFreeOursStep`, `detour_around_circle`), `turn_on_spot` (pivoting with the ball),
`kick`, and `ball_line_receive_point`, as `_pass_and_score._pass_exec` uses them; `_pass_exec`
itself needs both robots on one team, so its passer and receiver halves are re-expressed here
for a partner on the other team.

Real hardware:
  - The kicker is fixed power: about 2.1 m/s off the foot measured (5 m/s in the simulators), so
    a 1.2 m pass arrives at about 1.5-2 m/s. Shorten `PASS_DISTANCE` if passes die short.
  - Robots are capped at REAL_PARAMS.MAX_VEL (1 m/s): a receiver can only step about 0.3-0.5 m
    sideways onto a pass that is off line before the ball arrives.
  - The passer kicks only when the partner is still (below `READY_SPEED`) and faces the ball
    within `READY_FACING_RAD`; noisy vision on the partner can delay a pass, never fire one early.
  - Every real (and grSim) run records a game log in replays/field/ (`RECORD`).
"""

import sys
from pathlib import Path

# Run as `pixi run python examples/<this>.py`: utama_core is not installed, so put the repo on the path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import math
import time
from typing import Optional

from shooting_drill import (
    BLUE_0,
    FIELD,
    YELLOW_1,
    TestRobot,
    _field_bounds,
    _one_robot_strategy,
    _referee_profile,
)

from utama_core.config.physical_constants import BALL_RADIUS, ROBOT_RADIUS
from utama_core.custom_referee import CustomReferee
from utama_core.dashboard import attach_dashboard
from utama_core.dashboard.views import referee as referee_view
from utama_core.dashboard.views import strategy as strategy_view
from utama_core.engine.context import TickContext
from utama_core.engine.tactic import BaseTactic, RobotId, TacticTag
from utama_core.entities.data.command import RobotCommand
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game
from utama_core.entities.game.robot import Robot
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.global_utils.math_utils import detour_around_circle
from utama_core.run import StrategyRunner
from utama_core.shared.pass_and_score_geometry import (
    ORIENTATION_TOLERANCE_RAD,
    ball_line_receive_point,
    clamp_outside_enemy_defense_area,
    clamp_outside_own_defense_area,
    clamp_to_field,
    has_ball,
    kick_bend,
    oriented_towards,
)
from utama_core.skills.src.utils.move_utils import (
    empty_command,
    kick,
    move,
    turn_on_spot,
)
from utama_core.tactics.scripted_moves import ScriptedMovesTactic

# ---------------------------------------------------------------------------
# Test-day configuration
# ---------------------------------------------------------------------------

MODE = "real"  # "real", "rsim" or "grsim"; --mode overrides it

# The part of the field the cameras see (see `shooting_drill.py`); both cameras: "full".
CAMERA_HALF = "full"

# Swapping a robot is a change here only. They must wear different colours (one per team).
ROBOT_A = YELLOW_1
ROBOT_B = BLUE_0
ROBOT_A_COLOUR = "yellow"  # robot B is the other colour

# How far apart the receiver stands from the ball, in metres. The fixed-power kick (~2.1 m/s on
# the real robots) suits about 1-1.5 m on the lab floor.
PASS_DISTANCE = 1.2

# The passer kicks only once the partner is set: still and facing the ball.
READY_SPEED = 0.2  # m/s
READY_FACING_RAD = math.radians(20)
AIM_TOLERANCE_RAD = ORIENTATION_TOLERANCE_RAD  # the passer's heading to the partner, from the ball

# With a camera (real, grsim), record every run as an SSL game log in replays/field/.
RECORD = True

DASHBOARD_PORT = 8080  # --port overrides it

# ---------------------------------------------------------------------------

_ROLLING_MPS = 0.5  # a ball faster than this is a pass (or a miss) under way: `ball_line_receive_point`'s
_TIE_M = 0.05  # nearer the ball by less than this is a tie, won by the yellow robot
_MIN_PASS_M = 0.6  # a partner nearer than this is not set: it is still walking out
_MAX_PASS_M = 2.5
_FIELD_MARGIN_M = 0.25  # how far inside the field lines the receiver stands
_APPROACH_M = ROBOT_RADIUS + 0.03  # the passer's centre behind the ball before driving onto it
_LINED_UP_M = 0.04
_LINE_UP_FACING_RAD = 0.15
# A passer this near the ball but not behind it orbits the ball, facing it, rather than walk
# round it: on the 2026-10-27 field run it sat 0.09-0.12 m off at up to 86 degrees, nudging
# the ball for seconds while it crept round.
_ORBIT_RANGE_M = 0.25
_ORBIT_M = ROBOT_RADIUS + 0.07  # orbit radius, clear of the ball; spirals in to _APPROACH_M near the line-up
# How far round the ball each target lies. The robot drives at about 1.8 x its target's distance,
# so a longer step orbits faster: 30 deg crept round at ~0.15 m/s (2026-10-28), 60 deg ~0.3 m/s,
# the straight line to the target still clear of the ball.
_ORBIT_STEP_RAD = math.radians(60)
_OPEN_MIN_M = 0.8  # a receiver spot nearer the ball than this is turned away from, keeping its side
# How far off either defense area a robot keeps. The shared default (0.57 m, sized for the 9 x 6 m
# field) kept the right-hand robot at x <= 0.43 on this one: on the 2026-10-27 field run blue stood
# on that line and could not step onto passes rolling past its other side (1 of 6 caught).
_BOX_MARGIN_M = ROBOT_RADIUS + 0.05


def _partner(game: Game) -> Optional[Robot]:
    """The other robot: the only one on the other team (the nearest to the ball if there are more)."""
    others = [r for r in game.enemy_robots.values() if r is not None]
    if not others or game.ball is None:
        return None
    ball = game.ball.p.to_2d()
    return min(others, key=lambda r: r.p.distance_to(ball))


def _ball_in_field(game: Game) -> bool:
    ball = game.ball.p
    return abs(ball.x) <= game.field.half_length and abs(ball.y) <= game.field.half_width


def _offset_from_path(ball: Vector2D, velocity: Vector2D, point: Vector2D) -> tuple[float, float]:
    """(along, across): `point` relative to the ball's path, along it and off it, in metres."""
    rel = point - ball
    speed = velocity.mag()
    return rel.dot(velocity) / speed, abs(rel.x * velocity.y - rel.y * velocity.x) / speed


def pass_role(game: Game, robot_id: int) -> str:
    """What `robot_id` does this tick; both robots compute it from the same world, so they agree.

    "hold": no ball, no partner, or the ball out of the field (the operator puts it back).
    "pass": it has the ball, or is nearer the still ball: fetch it, aim, kick.
    "receive": the ball is rolling towards it rather than the partner: meet it on its path.
    "wait": the ball is rolling towards the partner (or away from both): stand still.
    "get_open": the partner is nearer the still ball: stand `PASS_DISTANCE` from it.
    """
    partner = _partner(game)
    if partner is None or not _ball_in_field(game):
        return "hold"
    me = game.friendly_robots[robot_id]
    ball = game.ball.p.to_2d()
    if has_ball(game, robot_id, visual=True):
        return "pass"
    velocity = Vector2D(game.ball.v.x, game.ball.v.y)
    if velocity.mag() >= _ROLLING_MPS:
        my_along, my_across = _offset_from_path(ball, velocity, me.p)
        their_along, their_across = _offset_from_path(ball, velocity, partner.p)
        if my_along > 0 and (their_along <= 0 or my_across <= their_across):
            return "receive"
        return "wait"
    margin = me.p.distance_to(ball) - partner.p.distance_to(ball)
    if margin < -_TIE_M or (abs(margin) <= _TIE_M and game.my_team_is_yellow):
        return "pass"
    return "get_open"


def outside_boxes(game: Game, point: Vector2D) -> Vector2D:
    """`point` moved just outside whichever defense area it is in or near (`_BOX_MARGIN_M`)."""
    return clamp_outside_enemy_defense_area(
        game, clamp_outside_own_defense_area(game, point, _BOX_MARGIN_M), _BOX_MARGIN_M
    )


def receive_spot(game: Game, robot_id: int) -> Vector2D:
    """Where the receiver goes for a ball rolling to it: on the ball's path (where it stands while
    the ball is too slow to read), outside both defense areas."""
    spot = ball_line_receive_point(game, robot_id)
    return outside_boxes(game, game.friendly_robots[robot_id].p if spot is None else spot)


def get_open_spot(game: Game, robot_id: int) -> Vector2D:
    """`PASS_DISTANCE` from the ball, on the side the robot already is, inside the field and out
    of both defense areas. Where that falls short of `_OPEN_MIN_M` (the ball near a line or a
    box), turned round the ball only as far as it must be, so the receiver never crosses to the
    far side: on the 2026-10-27 field run a jump to the centre's side swung the passer's aim and
    sent it round the ball again, four times a minute."""
    ball = game.ball.p.to_2d()

    def _spot(angle: float) -> Vector2D:
        spot = clamp_to_field(
            ball + Vector2D(math.cos(angle), math.sin(angle)) * PASS_DISTANCE, game, margin=_FIELD_MARGIN_M
        )
        return outside_boxes(game, spot)

    side = game.friendly_robots[robot_id].p - ball
    own = math.atan2(side.y, side.x) if side.mag() > 1e-6 else math.pi / 2
    turns = [0.0] + [sign * math.radians(step) for step in range(10, 181, 10) for sign in (1, -1)]
    spots = [_spot(own + turn) for turn in turns]
    spot = next((s for s in spots if s.distance_to(ball) >= _OPEN_MIN_M), None)
    if spot is None:
        spot = max(spots, key=lambda s: s.distance_to(ball))
    # A coarse grid: the robot's own drift does not move its target every tick (see `_pass_exec`).
    return Vector2D(round(spot.x / 0.05) * 0.05, round(spot.y / 0.05) * 0.05)


def orbit_point(ball: Vector2D, robot: Vector2D, aim: float) -> Vector2D:
    """The passer's next target round the ball: up to `_ORBIT_STEP_RAD` from where it is towards
    the line-up behind the ball (opposite `aim`), at `_ORBIT_M`, closing to `_APPROACH_M` over
    the last half radian, so it arrives behind the ball already facing it."""
    at = math.atan2(robot.y - ball.y, robot.x - ball.x)
    left = (aim + math.pi - at + math.pi) % (2 * math.pi) - math.pi
    angle = at + max(-_ORBIT_STEP_RAD, min(_ORBIT_STEP_RAD, left))
    radius = _APPROACH_M + (_ORBIT_M - _APPROACH_M) * min(1.0, abs(left) / 0.5)
    return ball + Vector2D(math.cos(angle), math.sin(angle)) * radius


def partner_ready(game: Game, partner: Robot) -> bool:
    """The partner is set for a pass: still, at a passing distance, and facing the ball."""
    ball = game.ball.p.to_2d()
    distance = partner.p.distance_to(ball)
    facing = partner.p.angle_to(ball)
    off = abs((facing - partner.orientation + math.pi) % (2 * math.pi) - math.pi)
    return partner.v.mag() < READY_SPEED and _MIN_PASS_M <= distance <= _MAX_PASS_M and off <= READY_FACING_RAD


class PassReceiveTactic(BaseTactic[None]):
    """One robot of a passing pair whose partner is on the other team (see the module docstring).
    No memory: the role is re-read from the world every tick (`pass_role`)."""

    tag = TacticTag.ATTACK

    def initial_mem(self) -> None:
        return None

    def tick(
        self, game: Game, ctx: TickContext, robot_ids: tuple[RobotId, ...], mem: None
    ) -> tuple[dict[RobotId, RobotCommand], None]:
        return {rid: self._command(game, ctx, rid) for rid in robot_ids}, mem

    def role(self, game: Game, robot_id: int) -> str:
        return pass_role(game, robot_id)

    def open_spot(self, game: Game, robot_id: int) -> Vector2D:
        """Where the robot waits for the partner's pass."""
        return get_open_spot(game, robot_id)

    def target(self, game: Game, robot_id: int) -> tuple[Vector2D, bool]:
        """What the passer kicks at, and whether it waits for the partner to be set first."""
        return _partner(game).p, True

    def partner_set(self, game: Game, partner: Robot) -> bool:
        return partner_ready(game, partner)

    def _command(self, game: Game, ctx: TickContext, robot_id: int) -> RobotCommand:
        role = self.role(game, robot_id)
        ctx.trace(game, f"pass_receive[{robot_id}].role", role)
        if role in ("hold", "wait"):
            return empty_command()
        me = game.friendly_robots[robot_id]
        ball = game.ball.p.to_2d()
        mc = ctx.motion_controller
        if role == "get_open":
            return move(game, mc, robot_id, self.open_spot(game, robot_id), me.p.angle_to(ball))
        if role == "receive":
            return move(game, mc, robot_id, receive_spot(game, robot_id), me.p.angle_to(ball), dribbling=True)
        # "pass": aimed at the target from the ball (as `_pass_exec` aims).
        target, wait_for_partner = self.target(game, robot_id)
        aim = ball.angle_to(target)
        if has_ball(game, robot_id):  # on the kicker (the ball sensor)
            aim -= kick_bend(game, mc, robot_id)  # against the kick's bend off the heading
            if not oriented_towards(game, robot_id, aim, AIM_TOLERANCE_RAD):
                return turn_on_spot(game, mc, robot_id, aim, dribbling=True)  # pivots round the ball
            if not wait_for_partner or self.partner_set(game, _partner(game)):
                return kick()
            return empty_command(dribbler_on=True)
        return take_ball_facing(game, mc, robot_id, aim)


def take_ball_facing(game: Game, mc, robot_id: int, aim: float) -> RobotCommand:
    """Gets the ball onto `robot_id`'s dribbler with the robot facing `aim`: lines up behind the
    ball, then drives onto it (as `DirectFreeOursStep` approaches a free kick). Fetching it
    face-on and pivoting with a ball not yet on the dribbler pushed it round the field in rsim."""
    me = game.friendly_robots[robot_id]
    ball = game.ball.p.to_2d()
    behind = ball - Vector2D(math.cos(aim), math.sin(aim)) * _APPROACH_M
    if me.p.distance_to(behind) > _LINED_UP_M and me.p.distance_to(ball) <= _ORBIT_RANGE_M:
        point = orbit_point(ball, me.p, aim)
        return move(game, mc, robot_id, point, point.angle_to(ball))
    if me.p.distance_to(behind) > _LINED_UP_M:
        waypoint = detour_around_circle(me.p, behind, ball, ROBOT_RADIUS + BALL_RADIUS)
        return move(game, mc, robot_id, waypoint, aim)
    if not oriented_towards(game, robot_id, aim, _LINE_UP_FACING_RAD):
        return turn_on_spot(game, mc, robot_id, aim)
    return move(game, mc, robot_id, ball, aim, dribbling=True)


def _still():
    return ScriptedMovesTactic(script=())


def build_runner(
    mode: str, port: int = DASHBOARD_PORT, tactic=PassReceiveTactic, name: str = "pass_receive"
) -> StrategyRunner:
    """Both robots running `tactic` (a `PassReceiveTactic`), its log and Strategy-panel choice named `name`."""
    real = mode == "real"
    if ROBOT_A_COLOUR not in ("yellow", "blue"):
        raise ValueError(f"ROBOT_A_COLOUR must be 'yellow' or 'blue', got {ROBOT_A_COLOUR!r}")
    a_is_yellow = ROBOT_A_COLOUR == "yellow"
    # In the simulators each team's robots are numbered from 0.
    a_id = ROBOT_A.vision_id if real else 0
    b_id = ROBOT_B.vision_id if real else 0
    a_map = {ROBOT_A.vision_id: ROBOT_A.firmware_id}
    b_map = {ROBOT_B.vision_id: ROBOT_B.firmware_id}
    a_profiles, b_profiles = {a_id: ROBOT_A.profile}, {b_id: ROBOT_B.profile}

    profile = _referee_profile()
    referee = CustomReferee(profile, n_robots_yellow=1, n_robots_blue=1)
    server = attach_dashboard(port=port)
    referee_view.attach(server, referee, profile)

    game_log = (
        f"replays/field/{time.strftime('%Y%m%d_%H%M%S')}_{mode}_{name}.log" if RECORD and mode != "rsim" else None
    )
    if game_log:
        print(f"Recording to {game_log}")
    runner = StrategyRunner(
        game_log_path=game_log,
        strategy=_one_robot_strategy(a_id, tactic),
        opp_strategy=_one_robot_strategy(b_id, tactic),
        my_team_is_yellow=a_is_yellow,
        my_team_is_right=False,  # either way: neither robot plays a goal
        mode=mode,
        exp_friendly=1,
        exp_enemy=1,
        full_field_dims=FIELD,
        field_bounds=_field_bounds(FIELD, CAMERA_HALF),
        referee=referee,
        referee_initial_command=RefereeCommand.HALT,
        show_live_status=not real,
        yellow_vision_to_cmd_mapping=(a_map if a_is_yellow else b_map) if real else None,
        blue_vision_to_cmd_mapping=(b_map if a_is_yellow else a_map) if real else None,
        yellow_robot_profiles=a_profiles if a_is_yellow else b_profiles,
        blue_robot_profiles=b_profiles if a_is_yellow else a_profiles,
    )
    # The Live tab's Strategy panel: stand either robot still, or restart it, while halted.
    tactics = {name: tactic, "still": _still}
    strategy_view.attach(
        server,
        runner,
        choices={
            "my": {n: (lambda t=t: _one_robot_strategy(a_id, t)) for n, t in tactics.items()},
            "opp": {n: (lambda t=t: _one_robot_strategy(b_id, t)) for n, t in tactics.items()},
        },
        running={"my": name, "opp": name},
    )
    return runner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("real", "rsim", "grsim"), default=MODE)
    parser.add_argument("--port", type=int, default=DASHBOARD_PORT, help="dashboard port (default %(default)s)")
    args = parser.parse_args()
    build_runner(args.mode, args.port).run()


if __name__ == "__main__":
    main()
