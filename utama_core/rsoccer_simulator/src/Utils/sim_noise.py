"""Seeded variation of rsim: latency, kick speed and vision noise, all off by default.

`SimNoise` holds the ranges and the seed (see `docs/sim_realism.md`, Phase 2). `SimNoise.draw()`
turns them into one match's values: each robot's latency and kick-speed offset, drawn once per
match. Kick-to-kick speed noise and vision noise come from their own seeded streams while the
match plays (`SSLStandardEnv`).

Every draw is keyed by team colour and robot id, never by which strategy plays, so two matches
with the same seed give the same robots the same physics: an A/B comparison against one
opponent is paired. Each robot's kicks and the vision noise use separate streams, so a strategy
that kicks more often does not shift anyone else's noise.

Not here, because they need the C++ physics (`vendor/rSim`): kick direction noise (the kicker in
`sslrobot.cpp`) and ball deceleration (the ball's damping and friction in `sslworld.cpp`).
"""

from __future__ import annotations

import dataclasses
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
    """

    seed: int = 0
    latency_s: tuple[float, float] = (0.0, 0.0)
    kick_speed_offset_sd: float = 0.0
    kick_speed_sd: float = 0.0
    vision: RsimGaussianNoise = RsimGaussianNoise()
    vanishing: float = 0.0

    def __post_init__(self):
        lo, hi = self.latency_s
        assert 0 <= lo <= hi, "latency_s must be a range of non-negative seconds"
        assert self.kick_speed_offset_sd >= 0 and self.kick_speed_sd >= 0
        assert 0 <= self.vanishing < 1

    @classmethod
    def from_dict(cls, d: dict) -> "SimNoise":
        d = dict(d)
        if "latency_s" in d:
            d["latency_s"] = tuple(d["latency_s"])
        if "vision" in d:
            d["vision"] = RsimGaussianNoise(**d["vision"])
        return cls(**d)

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["latency_s"] = list(self.latency_s)
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
        return SimDraw(latency_ticks=latency_ticks, kick_offset=kick_offset)

    def vision_rng(self, tick: int) -> np.random.Generator:
        """The vision noise for one tick: its own stream, so a tick's noise doesn't depend on how
        many detections earlier ticks drew."""
        return np.random.default_rng([self.seed, _VISION_STREAM, tick])

    def kick_rngs(self, n_robots: dict[str, int]) -> dict[str, list[np.random.Generator]]:
        return {
            colour: [np.random.default_rng([self.seed, _KICK_STREAM, c, i]) for i in range(n_robots[colour])]
            for c, colour in enumerate(COLOURS)
        }


@dataclass(frozen=True)
class SimDraw:
    """One match's drawn per-robot values, by colour (`COLOURS`) then robot id."""

    latency_ticks: dict[str, tuple[int, ...]]
    kick_offset: dict[str, tuple[float, ...]]

    def to_dict(self) -> dict:
        return {
            "latency_ticks": {k: list(v) for k, v in self.latency_ticks.items()},
            "kick_offset": {k: list(v) for k, v in self.kick_offset.items()},
        }


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
