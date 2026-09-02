"""1D bang-bang (piecewise-constant-acceleration) trajectory, the primitive
TIGERs Mannheim's trajectory-sampling path planner is built from (see
https://download.tigers-mannheim.de/papers/2024-RoboCup-Champion.pdf, section
2: "We are still using the same bang-bang trajectories that we introduced
back in 2015"). Time-optimal for a single axis with symmetric accel/decel and
zero final velocity: accelerate at +-a_max, then decelerate at -+a_max to
land exactly on the target with v=0, with an optional constant-velocity
cruise phase in between if the axis is long enough to reach v_max first.

Two of these (x and y) combined by `Trajectory2D` give a full 2D motion
primitive that already respects velocity/acceleration limits, unlike
FastPathPlanner's geometric path (a polyline handed to a downstream PID that
re-derives velocity after the fact — see fastpathplanning/planner.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BangBang1D:
    """Time-optimal 1D trajectory from (p0, v0) to (p1, v=0) respecting
    |velocity| <= v_max and |acceleration| <= a_max.

    Internally re-expressed as motion from x=0 (so the accel-phase/cruise/
    decel-phase breakpoints are simple to derive), sign-corrected so it
    always accelerates in the direction that reduces |p1 - p0|, and shifted
    back to the real p0/t=0 origin by `_x`/`_v` below.
    """

    p0: float
    v0: float
    p1: float
    v_max: float
    a_max: float
    # Precomputed phase breakpoints and derived sign/cruise-velocity, all in
    # the "distance still to cover" frame (positive = target ahead).
    t1: float  # end of accel phase
    t2: float  # end of cruise phase (== t1 if there is no cruise)
    t_end: float  # end of decel phase == total trajectory duration
    sign: float  # +1 if net motion is toward increasing p, else -1
    v_cruise: float  # peak velocity reached (signed, in the real p0 frame)

    @staticmethod
    def compute(p0: float, v0: float, p1: float, v_max: float, a_max: float) -> "BangBang1D":
        if v_max <= 0 or a_max <= 0:
            raise ValueError(f"v_max and a_max must be positive, got v_max={v_max}, a_max={a_max}")

        d = p1 - p0
        sign = 1.0 if d >= 0 else -1.0
        # Work in a frame where the net displacement is non-negative and v0
        # is expressed along that same direction, so the rest of this method
        # never has to branch on which way the robot is initially moving.
        d = abs(d)
        v0_signed = v0 * sign

        # Time and distance to bring v0_signed to 0 under -a_max (used only
        # to detect the "overshoot" case: robot moving hard away from the
        # target). Symmetric bang-bang trajectories that must end at v=0
        # don't have a closed form when v0 opposes the direction of travel
        # by more than what a single decel-then-accel pass can absorb within
        # the remaining distance, so that case first "wastes" the existing
        # away-velocity, then plans normally from the point it would stop.
        if v0_signed < 0:
            t_kill = -v0_signed / a_max
            d_kill = (v0_signed * v0_signed) / (2 * a_max)  # positive: distance lost while killing v0
            d_eff = d + d_kill
            v0_eff = 0.0
        else:
            t_kill = 0.0
            d_eff = d
            v0_eff = v0_signed

        # Distance covered while accelerating from v0_eff to v_max, and while
        # decelerating from v_max to 0 — if these overlap (their sum exceeds
        # d_eff), v_max is never reached: solve for the peak velocity
        # actually attained instead (a symmetric triangle profile).
        d_to_vmax = (v_max * v_max - v0_eff * v0_eff) / (2 * a_max)
        d_decel_from_vmax = (v_max * v_max) / (2 * a_max)

        if d_to_vmax < 0:
            # v0_eff already exceeds v_max (e.g. entering from a fast
            # carry-over state) — treat as if starting exactly at v_max.
            d_to_vmax = 0.0
            v0_eff = v_max

        if d_to_vmax + d_decel_from_vmax <= d_eff:
            # Full trapezoid: accel to v_max, cruise, decel to 0.
            t_acc = (v_max - v0_eff) / a_max
            d_acc = d_to_vmax
            d_cruise = d_eff - d_acc - d_decel_from_vmax
            t_cruise = d_cruise / v_max
            t_dec = v_max / a_max
            v_peak = v_max
        else:
            # Triangle: accelerate until the decel-to-zero distance exactly
            # uses up the remaining distance, no cruise phase.
            # v_peak^2 = v0_eff^2 + 2*a_max*d_eff, solved from symmetric
            # accel-then-decel: d_eff = (v_peak^2 - v0_eff^2)/(2a) + v_peak^2/(2a)
            v_peak_sq = (2 * a_max * d_eff + v0_eff * v0_eff) / 2
            v_peak = math.sqrt(max(v_peak_sq, 0.0))
            t_acc = (v_peak - v0_eff) / a_max
            t_cruise = 0.0
            t_dec = v_peak / a_max

        t1 = t_kill + t_acc
        t2 = t1 + t_cruise
        t_end = t2 + t_dec

        return BangBang1D(
            p0=p0,
            v0=v0,
            p1=p1,
            v_max=v_max,
            a_max=a_max,
            t1=t1,
            t2=t2,
            t_end=t_end,
            sign=sign,
            v_cruise=v_peak * sign,
        )

    def state_at(self, t: float) -> tuple[float, float]:
        """Return (position, velocity) at time `t` (clamped to [0, t_end])."""
        if t <= 0:
            return self.p0, self.v0
        if t >= self.t_end:
            return self.p1, 0.0

        sign = self.sign
        v0_signed = self.v0 * sign
        a = self.a_max

        # Phase 0: killing an initially-opposing velocity (only present when
        # v0_signed < 0 -- see `compute`). Recompute t_kill/d_kill from the
        # stored fields rather than storing them separately: cheap, and
        # keeps the dataclass's field list to genuinely load-bearing values.
        t_kill = max(0.0, -v0_signed / a) if v0_signed < 0 else 0.0
        if t < t_kill:
            v_signed = v0_signed + a * t
            d_signed = v0_signed * t + 0.5 * a * t * t
            return self.p0 + sign * d_signed, v_signed * sign

        # Signed net displacement during the kill phase (negative: the robot
        # moves backward, away from the target, while shedding v0). This is
        # the mirror image of `compute`'s `d_kill`, which is defined as the
        # positive magnitude of that same distance for use in `d_eff`.
        d_kill_signed = -(v0_signed * v0_signed) / (2 * a) if v0_signed < 0 else 0.0
        v_after_kill = 0.0 if v0_signed < 0 else v0_signed
        t_rel = t - t_kill

        if t < self.t1:
            v_signed = v_after_kill + a * t_rel
            d_signed = d_kill_signed + v_after_kill * t_rel + 0.5 * a * t_rel * t_rel
            return self.p0 + sign * d_signed, v_signed * sign

        d_acc = d_kill_signed + v_after_kill * (self.t1 - t_kill) + 0.5 * a * (self.t1 - t_kill) ** 2
        v_peak = abs(self.v_cruise)

        if t < self.t2:
            t_cruise_rel = t - self.t1
            d_signed = d_acc + v_peak * t_cruise_rel
            return self.p0 + sign * d_signed, v_peak * sign

        d_cruise = v_peak * (self.t2 - self.t1)
        t_dec_rel = t - self.t2
        v_signed = v_peak - a * t_dec_rel
        d_signed = d_acc + d_cruise + v_peak * t_dec_rel - 0.5 * a * t_dec_rel * t_dec_rel
        return self.p0 + sign * d_signed, v_signed * sign


@dataclass(frozen=True)
class Trajectory2D:
    """A single `BangBang1D` profile along the straight line from p0 to p1,
    mapped back onto x/y by the direction unit vector.

    Deliberately not two independent per-axis `BangBang1D`s: those would
    each trapezoid on their own schedule (different distances along x vs y
    reach v_max/decelerate at different times) and the combined path would
    bow away from the straight line between p0 and p1, exactly the
    "axis-skewed curve" this is meant to avoid. Projecting `v0` onto the
    line first means axis-transverse velocity (e.g. still drifting sideways
    from a previous move) is simply dropped, same simplification TIGERs'
    own bang-bang trajectories make (see module docstring) -- acceptable
    because `check_segment`'s outer sampling loop replans every tick from
    the robot's real live (p, v) regardless of what one candidate assumed.
    """

    along: BangBang1D
    ux: float
    uy: float
    p0: tuple[float, float]

    @property
    def duration(self) -> float:
        return self.along.t_end

    @staticmethod
    def compute(
        p0: tuple[float, float],
        v0: tuple[float, float],
        p1: tuple[float, float],
        v_max: float,
        a_max: float,
    ) -> "Trajectory2D":
        """Straight-line bang-bang trajectory from (p0, v0) to (p1, v=0)."""
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        dist = math.hypot(dx, dy)
        if dist < 1e-9:
            ux, uy = 1.0, 0.0
        else:
            ux, uy = dx / dist, dy / dist

        v0_along = v0[0] * ux + v0[1] * uy
        along = BangBang1D.compute(0.0, v0_along, dist, v_max, a_max)
        return Trajectory2D(along=along, ux=ux, uy=uy, p0=p0)

    def state_at(self, t: float) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return ((x, y), (vx, vy)) at time `t`."""
        d, v = self.along.state_at(t)
        px = self.p0[0] + self.ux * d
        py = self.p0[1] + self.uy * d
        return (px, py), (v * self.ux, v * self.uy)
