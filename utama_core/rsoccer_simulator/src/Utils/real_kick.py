"""Our robots' kickers as calibrated (`RobotProfile`), for the robots of a rehearsal in rsim.

A real kick leaves at the robot's own speed (about 2.1 m/s, not the sim's 5), off its heading by
its bias, bent towards the side the ball sits on the dribbler, and scattered kick to kick. A run
that gives rsim our robots' profiles (the examples do) kicks them so; a match without profiles
(the evaluation) keeps the sim's kick. Angles here are in our frame, anticlockwise +.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

# rsim's ball leaves this much slower than the kick speed commanded, at any speed (measured
# 2026-10-29: 1 to 5 m/s); a profile's kick speed is the ball's, as calibration measures it.
CONTACT_LOSS = 0.26


@dataclass(frozen=True)
class RealKick:
    speed: Optional[float] = None  # m/s; None: the sim's
    bend_deg_per_cm: float = 0.0
    bias_deg: float = 0.0
    spread_deg: float = 0.0  # SD of each kick's direction about the fit

    @classmethod
    def of(cls, profile) -> Optional["RealKick"]:
        """The kick `profile` (a `RobotProfile`) measured; None if it measured none."""
        kick = cls(
            profile.kick_speed, profile.kick_bend_deg_per_cm, profile.kick_bias_deg, profile.kick_spread_deg or 0.0
        )
        return None if kick == cls() else kick


class RealKicks:
    """Each kick of a robot with a `RealKick`: its speed, and its direction drawn when the kick
    command starts and held while it lasts (the kicker fires once per command)."""

    def __init__(self, kicks: dict[tuple[str, int], RealKick], seed: int = 0):
        self.kicks = kicks
        self._rngs = {key: np.random.default_rng([seed, i]) for i, key in enumerate(sorted(kicks))}
        self._held: dict[tuple[str, int], Optional[float]] = {}

    def __contains__(self, key: tuple[str, int]) -> bool:
        return key in self.kicks

    def reset(self) -> None:
        self._held.clear()

    def speed(self, key: tuple[str, int], nominal: float) -> float:
        speed = self.kicks[key].speed
        return nominal if speed is None else speed + CONTACT_LOSS

    def angle(self, key: tuple[str, int], kicking: bool, ball_left_m: float) -> float:
        """Radians off the heading, anticlockwise +, with the ball `ball_left_m` left of the
        robot's centre line."""
        if not kicking:
            self._held[key] = None
            return 0.0
        if self._held.get(key) is None:
            k = self.kicks[key]
            deg = k.bias_deg + k.bend_deg_per_cm * ball_left_m * 100.0 + self._rngs[key].normal() * k.spread_deg
            self._held[key] = math.radians(deg)
        return self._held[key]
