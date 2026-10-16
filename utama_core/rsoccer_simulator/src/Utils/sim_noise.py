"""Seeded variation of rsim: latency, kicks, ball friction, motion limits and vision noise, all off
by default.

`SimNoise` holds the ranges and the seed (see `docs/sim_realism.md`, Phase 2). `SimNoise.draw()`
turns them into one match's values: each robot's latency and kick-speed offset, drawn once per
match. Kick-to-kick speed noise and vision noise come from their own seeded streams while the
match plays (`SSLStandardEnv`).

Every draw is keyed by team colour and robot id, never by which strategy plays, so two matches
with the same seed give the same robots the same physics: an A/B comparison against one
opponent is paired. Each robot's kicks and the vision noise use separate streams, so a strategy
that kicks more often does not shift anyone else's noise.

Ball friction, kick direction and the physics substep count are applied inside the C++ physics
(`vendor/rSim`: `SSL.set_physics_params` and the kick command's direction field), so they need a
robosim binary built from this repo's fork (`vendor/rSim/FORK_NOTES.md`).
"""

from __future__ import annotations

import dataclasses
import math
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from utama_core.rsoccer_simulator.src.Utils.gaussian_noise import RsimGaussianNoise

COLOURS = ("team_blue", "team_yellow")

# Stream ids under the seed: one for the per-match draw, one for vision, one per robot's kicks.
_DRAW_STREAM = 0
_VISION_STREAM = 1
_KICK_STREAM = 2
# Added later, as their own streams, so the draws above stay what they were for a given seed.
_PHYSICS_STREAM = 3
_KICK_ANGLE_STREAM = 4

Range = Optional[tuple[float, float]]
_RANGE_FIELDS = ("ball_slide_decel", "ball_roll_decel", "max_speed", "max_accel", "max_brake")


@dataclass(frozen=True)
class SimNoise:
    """Ranges for one match's sim variation; the defaults change nothing.

    Args:
        seed: every random value in the match follows from it.
        latency_s: each robot's command latency, drawn uniformly from this range once per match
            and rounded to whole ticks. A robot's commands reach the sim that many ticks late.
        kick_speed_offset_sd: each robot's mean kick speed differs from the nominal speed by a
            normal draw with this SD (m/s), once per match.
        kick_speed_sd: every kick's speed varies around its robot's mean by this SD (m/s).
        vision: Gaussian position and heading noise on every detection.
        vanishing: the chance that any one detection is dropped from a frame.
        ball_roll_decel: the rolling ball's deceleration (m/s^2), drawn uniformly once per match.
            Setting it switches rsim from its constant friction force (about 0.65 m/s^2) to
            two-phase friction: after a kick the ball slides at `ball_slide_decel` until its speed
            falls to `ball_slide_fraction` of the kick speed, then rolls at this.
        ball_slide_decel: the sliding phase's deceleration (m/s^2), drawn once per match. Needs
            `ball_roll_decel`; unset, a kicked ball rolls straight away.
        ball_slide_fraction: where sliding ends, as a fraction of the kick speed (5/7 for a solid
            ball whose spin has caught up with its speed).
        kick_angle_offset_sd_deg: each robot's kicks leave its facing direction by a constant normal
            draw with this SD (degrees), once per match.
        kick_angle_sd_deg: every kick's direction varies around its robot's offset by this SD.
        max_speed, max_accel, max_brake: each robot's hardware limits (m/s, m/s^2), drawn uniformly
            once per match and applied to the commands the sim acts on, below any planner. Braking is
            any command that lowers the robot's speed.
        substeps: physics steps per sim tick (rsim's own is 5).
    """

    seed: int = 0
    latency_s: tuple[float, float] = (0.0, 0.0)
    kick_speed_offset_sd: float = 0.0
    kick_speed_sd: float = 0.0
    vision: RsimGaussianNoise = RsimGaussianNoise()
    vanishing: float = 0.0
    ball_roll_decel: Range = None
    ball_slide_decel: Range = None
    ball_slide_fraction: float = 5.0 / 7.0
    kick_angle_offset_sd_deg: float = 0.0
    kick_angle_sd_deg: float = 0.0
    max_speed: Range = None
    max_accel: Range = None
    max_brake: Range = None
    substeps: int = 5

    def __post_init__(self):
        lo, hi = self.latency_s
        assert 0 <= lo <= hi, "latency_s must be a range of non-negative seconds"
        assert self.kick_speed_offset_sd >= 0 and self.kick_speed_sd >= 0
        assert 0 <= self.vanishing < 1
        for name in _RANGE_FIELDS:
            r = getattr(self, name)
            assert r is None or 0 <= r[0] <= r[1], f"{name} must be a non-negative (low, high) range"
        assert self.ball_slide_decel is None or self.ball_roll_decel is not None, "ball_slide_decel needs ball_roll_decel"
        assert 0 < self.ball_slide_fraction <= 1
        assert self.kick_angle_offset_sd_deg >= 0 and self.kick_angle_sd_deg >= 0
        assert self.substeps >= 1

    @property
    def native_physics(self) -> bool:
        """Whether any setting needs the fork's C++ physics (`set_physics_params`)."""
        return (
            self.ball_roll_decel is not None
            or self.substeps != 5
            or bool(self.kick_angle_offset_sd_deg or self.kick_angle_sd_deg)
        )

    @property
    def motion_limits(self) -> bool:
        return any(r is not None for r in (self.max_speed, self.max_accel, self.max_brake))

    @classmethod
    def from_dict(cls, d: dict) -> "SimNoise":
        d = dict(d)
        if "latency_s" in d:
            d["latency_s"] = tuple(d["latency_s"])
        if "vision" in d:
            d["vision"] = RsimGaussianNoise(**d["vision"])
        for name in _RANGE_FIELDS:
            if d.get(name) is not None:
                d[name] = tuple(d[name])
        return cls(**d)

    def to_dict(self) -> dict:
        """Every setting, except later ones still at their default, so a match key from before
        they existed stays the same."""
        d = dataclasses.asdict(self)
        d["latency_s"] = list(self.latency_s)
        for name in _LATER_FIELDS:
            if getattr(self, name) == _LATER_DEFAULTS[name]:
                del d[name]
            elif name in _RANGE_FIELDS:
                d[name] = list(d[name])
        return d

    def draw(self, n_robots: dict[str, int], time_step: float) -> "SimDraw":
        """One match's per-robot values, for `n_robots` robots of each colour (`COLOURS`)."""
        rng = np.random.default_rng([self.seed, _DRAW_STREAM])
        lo, hi = self.latency_s
        latency_ticks, kick_offset = {}, {}
        for colour in COLOURS:  # fixed order, and a fixed count of draws per robot
            n = n_robots[colour]
            lat = rng.uniform(lo, hi, size=n)
            off = rng.normal(0.0, 1.0, size=n) * self.kick_speed_offset_sd
            latency_ticks[colour] = tuple(int(round(x / time_step)) for x in lat)
            kick_offset[colour] = tuple(float(x) for x in off)

        # The later settings come from their own stream: per match first, then per robot.
        prng = np.random.default_rng([self.seed, _PHYSICS_STREAM])
        ball = {}
        for name in ("ball_slide_decel", "ball_roll_decel"):
            r = getattr(self, name)
            ball[name] = float(prng.uniform(*r)) if r is not None else None
        per_robot = {}
        for name in ("kick_angle_offset", "max_speed", "max_accel", "max_brake"):
            if name == "kick_angle_offset":
                on = bool(self.kick_angle_offset_sd_deg)
            else:
                on = getattr(self, name) is not None
            values = {}
            for colour in COLOURS:
                n = n_robots[colour]
                if name == "kick_angle_offset":
                    x = prng.normal(0.0, 1.0, size=n) * self.kick_angle_offset_sd_deg
                else:
                    x = prng.uniform(*(getattr(self, name) or (0.0, 0.0)), size=n)
                values[colour] = tuple(float(v) for v in x)
            per_robot[name] = values if on else None
        return SimDraw(latency_ticks=latency_ticks, kick_offset=kick_offset, **ball, **per_robot)

    def vision_rng(self, tick: int) -> np.random.Generator:
        """The vision noise for one tick: its own stream, so a tick's noise doesn't depend on how
        many detections earlier ticks drew."""
        return np.random.default_rng([self.seed, _VISION_STREAM, tick])

    def kick_rngs(self, n_robots: dict[str, int]) -> dict[str, list[np.random.Generator]]:
        return {
            colour: [np.random.default_rng([self.seed, _KICK_STREAM, c, i]) for i in range(n_robots[colour])]
            for c, colour in enumerate(COLOURS)
        }

    def kick_angle_rngs(self, n_robots: dict[str, int]) -> dict[str, list[np.random.Generator]]:
        return {
            colour: [np.random.default_rng([self.seed, _KICK_ANGLE_STREAM, c, i]) for i in range(n_robots[colour])]
            for c, colour in enumerate(COLOURS)
        }


_LATER_FIELDS = (
    "ball_roll_decel",
    "ball_slide_decel",
    "ball_slide_fraction",
    "kick_angle_offset_sd_deg",
    "kick_angle_sd_deg",
    "max_speed",
    "max_accel",
    "max_brake",
    "substeps",
)
_LATER_DEFAULTS = {f.name: f.default for f in dataclasses.fields(SimNoise) if f.name in _LATER_FIELDS}


@dataclass(frozen=True)
class SimDraw:
    """One match's drawn per-robot values, by colour (`COLOURS`) then robot id."""

    latency_ticks: dict[str, tuple[int, ...]]
    kick_offset: dict[str, tuple[float, ...]]
    # The later settings; None when off.
    ball_slide_decel: Optional[float] = None
    ball_roll_decel: Optional[float] = None
    kick_angle_offset: Optional[dict[str, tuple[float, ...]]] = None  # degrees
    max_speed: Optional[dict[str, tuple[float, ...]]] = None
    max_accel: Optional[dict[str, tuple[float, ...]]] = None
    max_brake: Optional[dict[str, tuple[float, ...]]] = None

    def to_dict(self) -> dict:
        d = {
            "latency_ticks": {k: list(v) for k, v in self.latency_ticks.items()},
            "kick_offset": {k: list(v) for k, v in self.kick_offset.items()},
        }
        for name in ("ball_slide_decel", "ball_roll_decel"):
            if getattr(self, name) is not None:
                d[name] = getattr(self, name)
        for name in ("kick_angle_offset", "max_speed", "max_accel", "max_brake"):
            v = getattr(self, name)
            if v is not None:
                d[name] = {k: list(x) for k, x in v.items()}
        return d


class CommandDelay:
    """Delays each robot's command by its own number of ticks.

    `push(action)` takes this tick's commands ({colour: [command per robot]}) and returns the
    ones the sim should act on now: robot i's from `latency_ticks[colour][i]` ticks ago, or an
    all-zero command (stand still, no kick) before that many ticks have passed.
    """

    def __init__(self, latency_ticks: dict[str, tuple[int, ...]]):
        self._latency = latency_ticks
        depth = max((t for ticks in latency_ticks.values() for t in ticks), default=0)
        self._history: deque = deque(maxlen=depth + 1)

    def reset(self) -> None:
        self._history.clear()

    def push(self, action: dict) -> dict:
        self._history.append(action)
        out = {}
        for colour, commands in action.items():
            ticks = self._latency.get(colour, ())
            delayed = []
            for i, cmd in enumerate(commands):
                lag = ticks[i] if i < len(ticks) else 0
                if lag < len(self._history):
                    delayed.append(self._history[-1 - lag][colour][i])
                else:
                    delayed.append(np.zeros_like(cmd))
            out[colour] = tuple(delayed)
        return out


class KickSpeedNoise:
    """Each kick's speed: nominal + the robot's offset + fresh noise, drawn when a kick command
    starts and held while it lasts (the kicker fires once per command, however many ticks it is
    held). Never below `min_speed`."""

    def __init__(
        self,
        offsets: dict[str, tuple[float, ...]],
        sd: float,
        rngs: dict[str, list[np.random.Generator]],
        min_speed: float = 0.5,
    ):
        self._offsets = offsets
        self._sd = sd
        self._rngs = rngs
        self._min = min_speed
        self._held: dict[tuple[str, int], Optional[float]] = {}

    def reset(self) -> None:
        self._held.clear()

    def speed(self, colour: str, robot_id: int, kicking: bool, nominal: float) -> float:
        key = (colour, robot_id)
        if not kicking:
            self._held[key] = None
            return 0.0
        if self._held.get(key) is None:
            noise = self._rngs[colour][robot_id].normal() * self._sd
            self._held[key] = max(self._min, nominal + self._offsets[colour][robot_id] + noise)
        return self._held[key]


class KickAngleNoise:
    """Each kick's direction error (radians): the robot's offset + fresh noise, drawn when a kick
    command starts and held while it lasts, like `KickSpeedNoise`."""

    def __init__(
        self,
        offsets_deg: Optional[dict[str, tuple[float, ...]]],
        sd_deg: float,
        rngs: dict[str, list[np.random.Generator]],
    ):
        self._offsets = offsets_deg
        self._sd = sd_deg
        self._rngs = rngs
        self._held: dict[tuple[str, int], Optional[float]] = {}

    def reset(self) -> None:
        self._held.clear()

    def angle(self, colour: str, robot_id: int, kicking: bool) -> float:
        key = (colour, robot_id)
        if not kicking:
            self._held[key] = None
            return 0.0
        if self._held.get(key) is None:
            offset = self._offsets[colour][robot_id] if self._offsets else 0.0
            noise = self._rngs[colour][robot_id].normal() * self._sd
            self._held[key] = math.radians(offset + noise)
        return self._held[key]


class MotionLimits:
    """Per-robot hardware limits on the commands the sim acts on, below any planner.

    Each tick a robot's commanded velocity is brought within `max_accel * dt` (`max_brake * dt` when
    the command lowers its speed) of its actual velocity, then within `max_speed`. Commands are in
    the robot's own frame and velocities in the world frame, so both are compared in the world
    frame using the robot's heading.
    """

    def __init__(self, draw: SimDraw, time_step: float):
        self._speed = draw.max_speed
        self._accel = draw.max_accel
        self._brake = draw.max_brake
        self._dt = time_step

    @staticmethod
    def _value(values: Optional[dict[str, tuple[float, ...]]], colour: str, robot_id: int) -> float:
        return values[colour][robot_id] if values else math.inf

    def apply(self, colour: str, robot_id: int, command, robot) -> None:
        """Limits `command` (an rsim `Robot` command: local `v_x`, `v_y`) in place, given the robot's
        actual state `robot` (world `v_x`, `v_y`, heading `theta` in degrees)."""
        th = math.radians(robot.theta)
        c, s = math.cos(th), math.sin(th)
        gx = c * command.v_x - s * command.v_y
        gy = s * command.v_x + c * command.v_y
        ax, ay = robot.v_x, robot.v_y
        dx, dy = gx - ax, gy - ay
        braking = math.hypot(gx, gy) < math.hypot(ax, ay)
        rate = self._value(self._brake if braking else self._accel, colour, robot_id)
        change = math.hypot(dx, dy)
        if change > rate * self._dt:
            k = rate * self._dt / change
            gx, gy = ax + dx * k, ay + dy * k
        top = self._value(self._speed, colour, robot_id)
        speed = math.hypot(gx, gy)
        if speed > top:
            gx, gy = gx * top / speed, gy * top / speed
        command.v_x = c * gx + s * gy
        command.v_y = -s * gx + c * gy
