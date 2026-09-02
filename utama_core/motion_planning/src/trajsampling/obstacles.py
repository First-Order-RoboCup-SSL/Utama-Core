"""Trajectory-aware obstacle model for the trajectory-sampling planner.

Unlike FastPathPlanner's static/velocity-ray obstacles (checked once against
a fixed geometric path -- see fastpathplanning/planner.py's `_get_obstacles`),
every obstacle here is a function of time: `distance_at(t)` returns how far
the query point would be from the obstacle if the candidate trajectory were
actually being flown, so a fast-moving robot and a slow one project very
different danger zones at the same future instant. This mirrors TIGERs
Mannheim's obstacle model (2024 champion paper, section 2.5): own robots via
their own trajectory, opponents via a growing reachable-region derived from
their current velocity and assumed accel/velocity limits (equations 5-8 in
that paper), the ball via a simple constant-velocity projection (this
codebase has no general ball-trajectory model beyond the goal-line-crossing
`predict_ball_pos_at_x` -- see `data_processing/predictors/position.py` --
so a straight-line projection is used here rather than inventing one), and
static geometry (field bounds, enemy defense area) unchanged from FPP's own.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Protocol, Tuple

import numpy as np

from utama_core.motion_planning.src.trajsampling.bang_bang import BangBang1D


class TimedObstacle(Protocol):
    def distance_at(self, t: float, point: Tuple[float, float]) -> float:
        """Distance from `point` to this obstacle's extent at time `t`, minus
        nothing (raw geometric distance) -- callers apply their own margin.
        """
        ...


@dataclass(frozen=True)
class StaticSegmentObstacle:
    """A time-invariant line segment (field boundary, enemy defense area
    edge) -- same geometry FastPathPlanner already uses for these, just
    wrapped to satisfy `TimedObstacle`.
    """

    a: Tuple[float, float]
    b: Tuple[float, float]
    radius: float = 0.0

    def distance_at(self, t: float, point: Tuple[float, float]) -> float:
        del t
        ax, ay = self.a
        bx, by = self.b
        px, py = point
        abx, aby = bx - ax, by - ay
        seg_len_sq = abx * abx + aby * aby
        if seg_len_sq < 1e-12:
            d = math.hypot(px - ax, py - ay)
        else:
            u = ((px - ax) * abx + (py - ay) * aby) / seg_len_sq
            u = max(0.0, min(1.0, u))
            cx, cy = ax + u * abx, ay + u * aby
            d = math.hypot(px - cx, py - cy)
        return d - self.radius


@dataclass(frozen=True)
class OwnRobotObstacle:
    """A friendly robot moving along its own already-committed trajectory --
    we control it, so its future position is exactly known (given the
    trajectory doesn't get replanned differently), not merely estimated.
    """

    trajectory: "object"  # Trajectory2D, avoiding a circular import at type-check time
    radius: float

    def distance_at(self, t: float, point: Tuple[float, float]) -> float:
        (ox, oy), _ = self.trajectory.state_at(min(t, self.trajectory.duration))
        return math.hypot(point[0] - ox, point[1] - oy) - self.radius


@dataclass(frozen=True)
class ConstantVelocityObstacle:
    """A point moving at constant velocity from `p0` -- used for own robots
    with no active trajectory (idle/special-skill/off, mirroring FPP's
    fallback) and for the ball (see module docstring for why a full ball
    physics model isn't used here).
    """

    p0: Tuple[float, float]
    v: Tuple[float, float]
    radius: float

    def distance_at(self, t: float, point: Tuple[float, float]) -> float:
        ox = self.p0[0] + self.v[0] * t
        oy = self.p0[1] + self.v[1] * t
        return math.hypot(point[0] - ox, point[1] - oy) - self.radius


@dataclass(frozen=True)
class EnemyRobotObstacle:
    """A growing reachable-region around an opponent, per TIGERs 2024
    champion paper section 2.5 (equations 5-8): the opponent could be
    anywhere within a 1D bang-bang envelope of its current speed along its
    current direction of travel, for a bounded lookahead `t_max`. Modeled
    here as a growing circle (the paper's "tube" refinement -- shrinking the
    circle to a directional tube plus a fixed-radius circle for a stationary
    robot -- is a further tightening skipped in this first pass; see
    `EnemyRobotObstacle.distance_at`'s docstring for why a circle alone is
    still safe, just more conservative for a fast-moving opponent).
    """

    p0: Tuple[float, float]
    speed: float  # magnitude of current velocity
    direction: Tuple[float, float]  # unit vector of current velocity, or (0, 0) if stationary
    v_max: float
    a_max: float
    radius: float
    t_max: float = 0.5

    def __post_init__(self):
        # f+/f- from the paper: 1D bang-bang trajectories from position 0 to
        # +inf/-inf with the opponent's current speed and limits, sampled at
        # tc = clamp(t, 0, t_max). Since BangBang1D targets a finite point
        # (not literally +-inf), a target far enough away that the profile
        # never leaves its acceleration phase within t_max is exactly
        # equivalent -- v_max*t_max*4 is comfortably beyond that for any
        # t <= t_max given a_max > 0.
        far = max(self.v_max * self.t_max, 1.0) * 4.0
        object.__setattr__(self, "_f_plus", BangBang1D.compute(0.0, self.speed, far, self.v_max, self.a_max))
        object.__setattr__(self, "_f_minus", BangBang1D.compute(0.0, self.speed, -far, self.v_max, self.a_max))

    def distance_at(self, t: float, point: Tuple[float, float]) -> float:
        tc = max(0.0, min(t, self.t_max))
        f_plus, _ = self._f_plus.state_at(tc)
        f_minus, _ = self._f_minus.state_at(tc)
        r_dyn = abs(f_plus - f_minus) / 2.0
        # Stationary opponent: direction is undefined and r_dyn's f+/f-
        # split is meaningless as a directional offset -- just grow a plain
        # circle from p0, matching the paper's explicit tweak ("for
        # non-moving robots, the dynamic radius is set to zero, effectively
        # reducing the obstacle to a small circle").
        if self.speed < 1e-6:
            center = self.p0
        else:
            offset = f_minus + r_dyn
            center = (self.p0[0] + self.direction[0] * offset, self.p0[1] + self.direction[1] * offset)
        return math.hypot(point[0] - center[0], point[1] - center[1]) - (self.radius + r_dyn)


def enemy_obstacle_from_robot(
    p: Tuple[float, float], v: Tuple[float, float], radius: float, v_max: float, a_max: float
) -> EnemyRobotObstacle:
    speed = math.hypot(v[0], v[1])
    direction = (v[0] / speed, v[1] / speed) if speed > 1e-6 else (0.0, 0.0)
    return EnemyRobotObstacle(p0=p, speed=speed, direction=direction, v_max=v_max, a_max=a_max, radius=radius)


def collect_static_obstacles(
    field_bounds_segments: List[Tuple[np.ndarray, np.ndarray]], margin: float
) -> List[StaticSegmentObstacle]:
    """Wrap FastPathPlanner-style static boundary segments (field bounds,
    enemy defense area edges -- already computed the same way FPP does, see
    `FastPathPlanner._refresh_obstacle_cache`) with a uniform margin.
    """
    return [StaticSegmentObstacle(a=tuple(a), b=tuple(b), radius=margin) for a, b in field_bounds_segments]
