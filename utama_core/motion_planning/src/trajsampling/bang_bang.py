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

        d_raw = p1 - p0
        sign_toward_target = 1.0 if d_raw >= 0 else -1.0
        d_toward_target = abs(d_raw)
        v0_toward_target = v0 * sign_toward_target

        # Required-overshoot detection: v0 points toward the target
        # (v0_toward_target > 0) but braking at -a_max from here travels
        # further than the remaining gap — the robot cannot land on p1
        # without first passing it. There is no closed-form single-pass
        # bang-bang for that (the trajectory has to reverse direction after
        # decelerating through the target), so it's re-expressed in the
        # frame of the RETURN leg instead: flip `sign` to point back from
        # the (as yet unknown) overshoot point to p1. In that flipped frame
        # v0 becomes an opposing velocity — the exact shape the branch below
        # already handles — and `p1` sits *behind* `p0` (`d` negative),
        # which is fine: `d_eff` below only needs `d + d_pre >= 0` overall,
        # not `d >= 0` on its own. Confirmed algebraically and by sweep: the
        # remaining unmodified pipeline reduces to "decelerate past the
        # target to v=0, then bang-bang straight back" with these inputs.
        overshoot = False
        if v0_toward_target > 0:
            brake_dist = (v0_toward_target * v0_toward_target) / (2 * a_max)
            if brake_dist > d_toward_target:
                overshoot = True

        if overshoot:
            sign = -sign_toward_target
            d = d_raw * sign  # negative: p1 is behind p0 in this flipped frame
            v0_signed = v0 * sign  # negative: mirrors an opposing v0
        else:
            sign = sign_toward_target
            d = d_toward_target
            v0_signed = v0_toward_target

        # Pre-phase: bring v0_signed to whatever speed the normal
        # accel/cruise/decel schedule below can start from, before that
        # schedule's own math (which assumes a valid non-negative,
        # <=v_max starting speed) applies.
        #   * v0_signed < 0: decelerate through zero. Covers both a real
        #     opposing v0 and an overshoot re-expressed above as one —
        #     symmetric bang-bang trajectories ending at v=0 don't have a
        #     closed form when v0 opposes the direction of travel by more
        #     than a single decel-then-accel pass can absorb within the
        #     remaining distance, so this phase "wastes" the existing
        #     away-velocity first, then plans normally from the point it
        #     would stop.
        #   * v0_signed > v_max: same-direction overspeed that does NOT
        #     require an overshoot (braking distance already fits within
        #     the remaining gap) — shed the excess speed down to v_max
        #     before cruising/decelerating normally; entering the schedule
        #     below still above v_max would make it plan as if starting at
        #     v_max (see the `d_to_vmax < 0` carry-over case) while
        #     `state_at` correctly reports the true higher v0, producing an
        #     effectively instantaneous velocity change right after t=0.
        #   * otherwise: no pre-phase, v0_signed is already valid.
        if v0_signed < 0:
            t_pre = -v0_signed / a_max
            d_pre = (v0_signed * v0_signed) / (2 * a_max)  # positive: distance covered while decelerating to 0
            d_eff = d + d_pre
            v0_eff = 0.0
        elif v0_signed > v_max:
            t_pre = (v0_signed - v_max) / a_max
            d_pre = (v0_signed * v0_signed - v_max * v_max) / (2 * a_max)
            d_eff = d - d_pre
            v0_eff = v_max
        else:
            t_pre = 0.0
            d_eff = d
            v0_eff = v0_signed

        # Distance covered while accelerating from v0_eff to v_max, and while
        # decelerating from v_max to 0 — if these overlap (their sum exceeds
        # d_eff), v_max is never reached: solve for the peak velocity
        # actually attained instead (a symmetric triangle profile).
        d_to_vmax = (v_max * v_max - v0_eff * v0_eff) / (2 * a_max)
        d_decel_from_vmax = (v_max * v_max) / (2 * a_max)

        if d_to_vmax < 0:
            # v0_eff already exceeds v_max — the pre-phase above guarantees
            # this can't happen for the same-direction case (it already
            # shed speed down to exactly v_max), so this only remains
            # reachable defensively; treat as if starting exactly at v_max.
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

        t1 = t_pre + t_acc
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

        # Pre-phase: mirrors `compute`'s pre-phase branch exactly (recomputed
        # from the stored fields rather than stored separately: cheap, and
        # keeps the dataclass's field list to genuinely load-bearing values).
        #   * v0_signed < 0: decelerate through zero at +a (a real opposing
        #     v0, or an overshoot re-expressed by `compute` with a flipped
        #     `sign` so the excess toward-target speed looks like one).
        #   * v0_signed > v_max: decelerate down to v_max at -a (same-
        #     direction overspeed that doesn't require an overshoot).
        #   * otherwise: no pre-phase.
        if v0_signed < 0:
            v_pre_target = 0.0
            a_pre = a
        elif v0_signed > self.v_max:
            v_pre_target = self.v_max
            a_pre = -a
        else:
            v_pre_target = v0_signed
            a_pre = 0.0
        t_pre = max(0.0, (v_pre_target - v0_signed) / a_pre) if a_pre != 0.0 else 0.0

        if t < t_pre:
            v_signed = v0_signed + a_pre * t
            d_signed = v0_signed * t + 0.5 * a_pre * t * t
            return self.p0 + sign * d_signed, v_signed * sign

        # Signed net displacement during the pre-phase (negative when it
        # decelerated through zero and briefly moved backward; positive when
        # it shed same-direction overspeed while still moving forward).
        d_pre_signed = v0_signed * t_pre + 0.5 * a_pre * t_pre * t_pre
        v_after_pre = v_pre_target
        t_rel = t - t_pre

        if t < self.t1:
            v_signed = v_after_pre + a * t_rel
            d_signed = d_pre_signed + v_after_pre * t_rel + 0.5 * a * t_rel * t_rel
            return self.p0 + sign * d_signed, v_signed * sign

        d_acc = d_pre_signed + v_after_pre * (self.t1 - t_pre) + 0.5 * a * (self.t1 - t_pre) ** 2
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
            # p0 == p1 (to within numerical noise): there is no well-defined
            # p0->p1 direction to project v0 onto, and projecting onto an
            # arbitrary fixed axis (the old behaviour) silently discarded
            # whichever component of v0 happened to be perpendicular to it --
            # a stationary target with real lateral v0 got a trajectory that
            # commanded exactly zero velocity forever, never actually
            # stopping the robot's real sideways motion. This case isn't
            # "travel from p0 to p1" at all; it's "come to rest at p0 from
            # whatever v0 currently is" -- use v0's own direction as the axis
            # instead, so the full speed (not just one arbitrary component)
            # feeds into the single BangBang1D braking profile below. Found
            # live: `turn_on_spot`/`move` call this every tick with
            # target_coords=robot.p (see move_utils.py), which is exactly
            # this case -- a robot pivoting on the ball with residual lateral
            # velocity would stall for seconds waiting for a stop that was
            # never actually commanded.
            speed = math.hypot(v0[0], v0[1])
            if speed < 1e-9:
                ux, uy = 1.0, 0.0  # not moving either -- direction is irrelevant
            else:
                ux, uy = v0[0] / speed, v0[1] / speed
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
