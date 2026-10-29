# ruff: noqa: E402
"""examples/ball_trials.py — the ball part of `pixi run calibrate`: the robot fetches the ball itself.

`BallTrialsTactic` measures how fast one robot can carry the ball and how its kicks leave, with no
one placing the ball on its dribbler:

- **Carry ladder**: per direction (forward, backward, left, right, turning either way) it carries
  the ball `CARRY_M` (half a turn for a turn) at rising speeds, `repeats` times each, from
  standing, at the planner's acceleration (`REAL_PARAMS`), then stands `HOLD_S` with it. A trial
  is lost once the ball is `BALL_LOST_M` off the robot or its ball sensor has been off
  `SENSOR_OFF_S`. A direction stops at the first speed that loses the ball: the speed below is
  what the robot carries at (`RobotParams.CARRY_SPEED`, `CARRY_TURN`).
- **Kicks**: `kicks` kicks along the field from the ball wherever it lies, for kick speed and the
  kick's bend (`analysis/robot_calibration.py` fits it from the recorded log). With `aim`, each
  is aimed instead at the middle of the far end of `bounds`, against the robot's kick bend and
  bias (`kick_bend`), turned to within `AIM_RAD` first; `kick_results` has how far each missed.
  Every kick comes charged: within `CHARGE_KEEPS_S` of the kicker's last kick (it runs flat
  after), else the robot lets go of the ball and kicks once with none first, and at least
  `RECHARGE_S` after the last.

Before every trial and kick it fetches the ball (`pass_receive.take_ball_facing`), lined up so
the carry runs towards the middle of `bounds`, and lets it go after. It asks for the ball back
(`waiting_for`, answered by `proceed()`) only when the ball is out of view or within `EDGE_M` of
the edge of `bounds`, as after most kicks: roll it back towards the middle.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import math
from typing import NamedTuple, Optional

from pass_receive import take_ball_facing

from utama_core.config.robot_params import REAL_PARAMS
from utama_core.engine.context import TickContext
from utama_core.engine.tactic import BaseTactic, RobotId, TacticTag
from utama_core.entities.data.command import RobotCommand
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import FieldBounds, Game
from utama_core.shared.pass_and_score_geometry import kick_bend
from utama_core.skills.src.utils.move_utils import turn_on_spot

CARRY_M = 0.5  # how far a carry goes
TURN_RAD = math.pi  # how far a turn goes
HOLD_S = 0.5  # standing with the ball after a carry, before it counts as kept
SETTLE_S = 0.5  # standing with the ball after fetching it, before the carry starts
RELEASE_S = 0.75  # backing off the ball, dribbler off, after a trial or a kick
RELEASE_SPEED = 0.2
KICK_S = 0.3  # one kick
AFTER_KICK_S = 1.5
RECHARGE_S = 1.2  # the kicker recharges in about 1 s after a kick
CHARGE_KEEPS_S = 30.0  # unused this long, the kicker's capacitor runs down and the kick is weak
AIM_RAD = math.radians(1.5)  # an aimed kick waits until the robot faces its aim this near
AIM_HOLD_S = 0.2  # for this long
AIM_TIMEOUT_S = 4.0  # or kicks anyway after this long turning
MISS_FROM_M = 0.3  # the kick's direction: from the ball at the kick to where it is this far on
FETCH_TIMEOUT_S = 20.0  # fetching longer than this asks for the ball to be put back
BALL_LOST_M = 0.15  # ball centre this far from the robot's: off the dribbler (held, ~0.09 m)
SENSOR_OFF_S = 0.3
LINED_UP_RAD = 0.3  # the ball taken facing this near the heading wanted starts the trial
EDGE_M = 0.25  # a ball nearer the edge of the bounds than this is asked for back
_GAP_S = 0.1  # a tick this long after the last one means play was paused

CARRY_SPEEDS = (0.2, 0.3, 0.5, 0.7, 1.0)  # m/s
TURN_SPEEDS = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0)  # rad/s


class Direction(NamedTuple):
    name: str
    angle: float  # the carry's direction in the robot's frame (0 forward, anticlockwise +)
    turn: int  # 0 for a carry; +1 or -1 for a turn on the spot, anticlockwise +
    speeds: tuple[float, ...]


DIRECTIONS = (
    Direction("forward", 0.0, 0, CARRY_SPEEDS),
    Direction("backward", math.pi, 0, CARRY_SPEEDS),
    Direction("left", math.pi / 2, 0, CARRY_SPEEDS),
    Direction("right", -math.pi / 2, 0, CARRY_SPEEDS),
    Direction("turn anticlockwise", 0.0, 1, TURN_SPEEDS),
    Direction("turn clockwise", 0.0, -1, TURN_SPEEDS),
)


def speed_at(t: float, distance: float, speed: float, accel: float) -> float:
    """The speed `t` s into a move of `distance` from standing to standing: up at `accel` to
    `speed` (or as near as `distance` allows), then down at `accel` to 0 at its end."""
    top = min(speed, math.sqrt(distance * accel))
    total = distance / top + top / accel
    return max(0.0, min(top, accel * t, accel * (total - t)))


def move_seconds(distance: float, speed: float, accel: float) -> float:
    top = min(speed, math.sqrt(distance * accel))
    return distance / top + top / accel


def carry_heading(ball: Vector2D, bounds: FieldBounds, direction: Direction) -> float:
    """The heading to take the ball facing so that the carry runs towards the middle of `bounds`."""
    (x0, y0), (x1, y1) = bounds.top_left, bounds.bottom_right
    towards_middle = math.atan2((y0 + y1) / 2 - ball.y, (x0 + x1) / 2 - ball.x)
    return towards_middle - direction.angle


def kick_heading(ball: Vector2D, bounds: FieldBounds) -> float:
    """Along the field, towards the end further from the ball."""
    (x0, _), (x1, _) = bounds.top_left, bounds.bottom_right
    return 0.0 if ball.x < (x0 + x1) / 2 else math.pi


def kick_target(ball: Vector2D, bounds: FieldBounds) -> Vector2D:
    """The middle of the end of `bounds` further from the ball: what an aimed kick aims at."""
    (x0, y0), (x1, y1) = bounds.top_left, bounds.bottom_right
    return Vector2D(x1 if ball.x < (x0 + x1) / 2 else x0, (y0 + y1) / 2)


def at_caps(speed: float, turn: float) -> tuple[Direction, ...]:
    """Every direction at one speed: a robot's carry caps, for checking them."""
    return tuple(d._replace(speeds=(turn if d.turn else speed,)) for d in DIRECTIONS)


def in_reach(game: Game, bounds: FieldBounds) -> bool:
    """The ball is seen and far enough inside `bounds` to be fetched and carried."""
    if game.ball is None:
        return False
    (x0, y0), (x1, y1) = bounds.top_left, bounds.bottom_right
    b = game.ball.p
    return min(x1, x0) + EDGE_M <= b.x <= max(x1, x0) - EDGE_M and min(y0, y1) + EDGE_M <= b.y <= max(y0, y1) - EDGE_M


class Ladder:
    """Which trial is next: every direction's speeds in turn, `repeats` trials each, a direction
    ending at the first speed that loses the ball."""

    def __init__(self, directions: tuple[Direction, ...] = DIRECTIONS, repeats: int = 2):
        self.directions = directions
        self.repeats = repeats
        self.results: list[dict] = []
        self._direction = 0
        self._speed = 0
        self._repeat = 0
        self._lost_here = False

    @property
    def done(self) -> bool:
        return self._direction >= len(self.directions)

    def current(self) -> tuple[Direction, float, int]:
        d = self.directions[self._direction]
        return d, d.speeds[self._speed], self._repeat + 1

    def record(self, held: bool, t: float) -> None:
        d, speed, repeat = self.current()
        self.results.append({"direction": d.name, "speed": speed, "repeat": repeat, "held": held, "t": round(t, 2)})
        self._lost_here |= not held
        self._repeat += 1
        if self._repeat < self.repeats:
            return
        self._repeat = 0
        if self._lost_here or self._speed + 1 >= len(d.speeds):
            self._direction, self._speed, self._lost_here = self._direction + 1, 0, False
        else:
            self._speed += 1

    def summary(self) -> dict[str, Optional[float]]:
        """Per direction, the fastest speed every trial at it, and at every speed below, kept the ball."""
        out: dict[str, Optional[float]] = {}
        for d in self.directions:
            rows = [r for r in self.results if r["direction"] == d.name]
            if not rows:
                continue
            kept = None
            for speed in d.speeds:
                at = [r["held"] for r in rows if r["speed"] == speed]
                if not at or not all(at):
                    break
                kept = speed
            out[d.name] = kept
        return out


class BallTrialsTactic(BaseTactic[None]):
    """Runs the carry ladder then `kicks` kicks on its first robot, fetching the ball for each."""

    tag = TacticTag.MIXED

    def __init__(
        self,
        bounds: FieldBounds,
        repeats: int = 2,
        kicks: int = 10,
        directions: tuple[Direction, ...] = DIRECTIONS,
        aim: bool = False,
        accel: float = REAL_PARAMS.MAX_ACCELERATION,
        turn_accel: float = REAL_PARAMS.MAX_ANGULAR_ACCELERATION,
    ):
        self.bounds = bounds
        self.ladder = Ladder(directions, repeats)
        self.kicks_left = kicks
        self.kicks_done = 0
        self.aim = aim
        # Per kick: its time and how far the ball left off the aim (aim) or heading, degrees
        # anticlockwise +; None if the ball never got `MISS_FROM_M` away.
        self.kick_results: list[dict] = []
        self.accel = accel
        self.turn_accel = turn_accel
        # Progress lives here, not in `mem`: the kernel may reset `mem` when play resumes.
        self._phase = "charge"  # charge, fetch, settle, carry, hold, kick, release; "" once done
        self._elapsed = 0.0
        self._last_ts: Optional[float] = None
        self._heading = 0.0
        self._sensor_off = 0.0
        self._asking: Optional[str] = None
        self._proceed = False
        self._last_kick: Optional[float] = None
        self._fired_at: Optional[float] = None  # this phase's kick
        self._kick_from: Optional[Vector2D] = None
        self._kick_bearing = 0.0
        self._miss: Optional[float] = None
        self._aimed = 0.0

    def initial_mem(self) -> None:
        return None

    @property
    def done(self) -> bool:
        return self._phase == ""

    @property
    def waiting_for(self) -> Optional[str]:
        return self._asking

    def proceed(self) -> None:
        self._proceed = True

    @property
    def status(self) -> str:
        """What it is doing, for the operator; changes only when the step does."""
        if self.done:
            return "done"
        if self._phase in ("charge", "recharge"):
            return "charge the kicker (no ball)"
        if self._phase == "release":
            return "let go of the ball"
        if not self.ladder.done:
            d, speed, repeat = self.ladder.current()
            unit = "rad/s" if d.turn else "m/s"
            return f"carry {d.name} {speed:g} {unit} ({repeat}/{self.ladder.repeats}): {self._phase}"
        return f"kick {self.kicks_done + 1}/{self.kicks_done + self.kicks_left}: {self._phase}"

    def _to(self, phase: str) -> None:
        self._phase, self._elapsed, self._sensor_off = phase, 0.0, 0.0
        self._fired_at, self._aimed = None, 0.0

    def _fire(self, ts: float) -> bool:
        """Kicks now if the kicker has recharged since its last kick; False while it has not."""
        if self._fired_at is None:
            if self._last_kick is not None and ts - self._last_kick < RECHARGE_S:
                return False
            self._fired_at = self._last_kick = ts
        return ts - self._fired_at < KICK_S

    def _charged(self, ts: float) -> bool:
        return self._last_kick is not None and ts - self._last_kick < CHARGE_KEEPS_S

    def _aim(self, game: Game, mc, robot_id: int) -> float:
        ball = game.ball.p.to_2d()
        return ball.angle_to(kick_target(ball, self.bounds)) - kick_bend(game, mc, robot_id)

    def _next(self, game: Game) -> None:
        """After a trial or a kick: fetch for the next, or done."""
        if self.ladder.done and self.kicks_left == 0:
            self._phase = ""
        else:
            self._to("fetch")

    def _target_heading(self, game: Game) -> float:
        ball = game.ball.p.to_2d()
        if not self.ladder.done:
            return carry_heading(ball, self.bounds, self.ladder.current()[0])
        if self.aim:
            return ball.angle_to(kick_target(ball, self.bounds))
        return kick_heading(ball, self.bounds)

    def _lost(self, game: Game, robot_id: int, dt: float) -> bool:
        robot = game.friendly_robots[robot_id]
        self._sensor_off = 0.0 if robot.has_ball else self._sensor_off + dt
        far = game.ball is None or robot.p.distance_to(game.ball.p.to_2d()) > BALL_LOST_M
        return far or self._sensor_off >= SENSOR_OFF_S

    def tick(
        self, game: Game, ctx: TickContext, robot_ids: tuple[RobotId, ...], mem: None
    ) -> tuple[dict[RobotId, RobotCommand], None]:
        still = RobotCommand(0.0, 0.0, 0.0, kick=False, chip=False, dribble=False)
        commands = {rid: still for rid in robot_ids}
        if not robot_ids or self.done:
            return commands, mem
        rid = robot_ids[0]
        dt = 0.0 if self._last_ts is None else game.ts - self._last_ts
        self._last_ts = game.ts
        if dt > _GAP_S:
            dt = 0.0
            if self._phase in ("carry", "hold", "settle", "aim"):
                self._to("release")  # paused mid-trial: let go and fetch again
        self._elapsed += dt
        robot = game.friendly_robots.get(rid)
        if robot is None:
            return commands, mem

        if self._phase == "charge":
            # Left unused, the kicker's capacitor runs flat and its first kick barely moves the ball.
            if self._fire(game.ts):
                commands[rid] = still._replace(kick=True)
                return commands, mem
            self._to("fetch")

        if self._phase == "recharge":  # the ball let go of first: back off, kick at nothing, wait
            if self._elapsed < RELEASE_S:
                commands[rid] = still._replace(local_forward_vel=-RELEASE_SPEED)
            elif self._fire(game.ts):
                commands[rid] = still._replace(kick=True)
            elif self._fired_at is not None and game.ts - self._fired_at >= RECHARGE_S:
                self._to("fetch")
            return commands, mem

        if self._phase == "fetch":
            if self._asking is not None:
                if not self._proceed:
                    return commands, mem
                self._asking, self._proceed = None, False
                self._to("fetch")
            if not in_reach(game, self.bounds) or self._elapsed > FETCH_TIMEOUT_S:
                self._asking = "Put the ball back near the middle of the field, 1 m clear of the robot's way"
                return commands, mem
            self._heading = self._target_heading(game)
            commands[rid] = take_ball_facing(game, ctx.motion_controller, rid, self._heading)
            off = abs(math.remainder(robot.orientation - self._heading, 2 * math.pi))
            if robot.has_ball and off <= LINED_UP_RAD:
                self._to("settle")
            return commands, mem

        dribble = still._replace(dribble=True)
        if self._phase == "settle":
            commands[rid] = dribble
            if self._lost(game, rid, dt):
                self._to("fetch")
            elif self._elapsed >= SETTLE_S:
                if not self.ladder.done:
                    self._to("carry")
                elif not self._charged(game.ts):
                    self._to("recharge")
                else:
                    self._to("aim" if self.aim else "kick")
            return commands, mem

        if self._phase == "aim":
            heading = self._aim(game, ctx.motion_controller, rid)
            commands[rid] = turn_on_spot(game, ctx.motion_controller, rid, heading, dribbling=True)
            if self._lost(game, rid, dt):
                self._to("fetch")
                return commands, mem
            off = abs(math.remainder(robot.orientation - heading, 2 * math.pi))
            self._aimed = self._aimed + dt if off <= AIM_RAD else 0.0
            if self._aimed >= AIM_HOLD_S or self._elapsed >= AIM_TIMEOUT_S:
                self._to("kick" if self._charged(game.ts) else "recharge")
            return commands, mem

        if self._phase in ("carry", "hold"):
            d, speed, _ = self.ladder.current()
            if self._lost(game, rid, dt):
                self.ladder.record(False, game.ts)
                self._to("release")
                return commands, mem
            if self._phase == "carry":
                distance, accel = (TURN_RAD, self.turn_accel) if d.turn else (CARRY_M, self.accel)
                if self._elapsed >= move_seconds(distance, speed, accel):
                    self._to("hold")
                else:
                    v = speed_at(self._elapsed, distance, speed, accel)
                    if d.turn:
                        commands[rid] = dribble._replace(angular_vel=d.turn * v)
                    else:
                        commands[rid] = dribble._replace(
                            local_forward_vel=v * math.cos(d.angle), local_left_vel=v * math.sin(d.angle)
                        )
                    return commands, mem
            commands[rid] = dribble
            if self._elapsed >= HOLD_S:
                self.ladder.record(True, game.ts)
                self._to("release")
            return commands, mem

        if self._phase == "kick":
            if self._fired_at is None:
                ball = game.ball.p.to_2d() if game.ball is not None else robot.p
                self._kick_from, self._miss = ball, None
                self._kick_bearing = ball.angle_to(kick_target(ball, self.bounds)) if self.aim else robot.orientation
            if self._fire(game.ts):
                commands[rid] = dribble._replace(kick=True)
                return commands, mem
            if self._fired_at is None:  # still recharging
                commands[rid] = dribble
                return commands, mem
            if self._miss is None and game.ball is not None:
                b = game.ball.p.to_2d()
                if b.distance_to(self._kick_from) >= MISS_FROM_M:
                    self._miss = math.degrees(
                        math.remainder(self._kick_from.angle_to(b) - self._kick_bearing, 2 * math.pi)
                    )
            if game.ts - self._fired_at >= KICK_S + AFTER_KICK_S:
                miss = None if self._miss is None else round(self._miss, 1)
                self.kick_results.append({"t": round(self._fired_at, 2), "miss_deg": miss})
                self.kicks_left -= 1
                self.kicks_done += 1
                self._next(game)
            return commands, mem

        if self._phase == "release":
            if self._elapsed < RELEASE_S:
                commands[rid] = still._replace(local_forward_vel=-RELEASE_SPEED)
            else:
                self._next(game)
            return commands, mem
        return commands, mem
