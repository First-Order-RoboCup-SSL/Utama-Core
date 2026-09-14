"""Numba-jitted core of `TrajectorySamplingPlanner._first_collision`.

Profiling (see `planner.py`'s module docstring) showed
`_first_collision`'s scan loop -- ~18 adaptive timesteps x ~20 obstacles per
call, tens of thousands of calls per match -- as the dominant cost of the
trajectory-sampling planner, responsible for it running ~2.4x slower than
FastPathPlanner in a real match. Two things were verified via standalone
microbenchmarks before writing this module:

1. Decorating tiny leaf functions (e.g. a lone `state_at`) with `@njit` and
   calling them one at a time from a Python loop is a REGRESSION (~0.73x --
   slower), because the Python<->native call boundary is crossed once per
   call and dominates over the actual arithmetic for a function this small.
2. Moving the ENTIRE timestep x obstacle scan loop into one `@njit` call
   (so the boundary is crossed once per `_first_collision` call, not once
   per obstacle per timestep) gives ~24x-143x on synthetic benchmarks of the
   same loop shape.

So this module does NOT decorate `BangBang1D.state_at` or any obstacle's
`distance_at` in place. Instead, `planner.py` flattens the querying
trajectory and the (heterogeneous) obstacle list into plain float arrays
once per `plan()` call, and this module's `first_collision_numba` runs the
*entire* scan loop natively, with each obstacle kind's distance formula
inlined directly (mirroring the exact math in `obstacles.py` and
`bang_bang.py` -- kept in lockstep with those by hand, since Numba can't
share code with the Protocol-dispatched Python classes).

Obstacle kinds, one flat 2D float array each (row per obstacle):
  - static:  [ax, ay, bx, by, radius]                     (segment)
  - traj:    [p0x, p0y, ux, uy, t_kill_src..., radius,     (own-robot /
              time_offset, traj_duration]                  committed-trajectory)
  - cv:      [p0x, p0y, vx, vy, radius]                    (constant velocity)
  - enemy:   [p0x, p0y, dirx, diry, speed, radius, t_max,  (growing envelope)
              f_plus params..., f_minus params...]

"traj" rows store enough of `BangBang1D`'s precomputed breakpoints
(`t1, t2, t_end, sign, v_cruise`, alongside `p0` of the 1D profile) to
re-evaluate `state_at` inline -- see `_bangbang_state_at` below, a line-for
-line port of `BangBang1D.state_at`.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
from numba import njit

# Column layout constants, kept here (not magic indices scattered through
# the njit functions) so `planner.py`'s array-building code and this
# module's row-unpacking code can't silently drift apart.
BANGBANG_1D_COLS = 8  # p0, v0, p1, v_max, a_max, t1, t2, t_end -- plus sign, v_cruise below
# Full per-axis bang-bang profile needs: p0, v0, p1, v_max, a_max, t1, t2,
# t_end, sign, v_cruise -- 10 floats. Stored as one row per profile.
BB_P0, BB_V0, BB_P1, BB_VMAX, BB_AMAX, BB_T1, BB_T2, BB_TEND, BB_SIGN, BB_VCRUISE = range(10)
BB_ROW_LEN = 10

# TRAJ obstacle row: a friendly robot's committed trajectory can itself be
# either a plain `Trajectory2D` (one `along` profile) or a
# `TwoSegmentTrajectory` (two, switched at `switch_t`) -- `_commit` stores
# whatever `plan()` picked, which is very often a two-segment candidate.
# Represented uniformly the same way `_flatten_query_trajectory` represents
# the QUERYING trajectory: leg1 always present; a plain `Trajectory2D` sets
# `switch_t` to its own duration (so `t <= switch_t` always holds and leg2,
# a dummy zero-profile, is never evaluated) -- see `_flatten_obstacles`.
# Row = leg1 profile (BB_ROW_LEN) + leg1 ux,uy,p0x,p0y (4) + switch_t (1) +
# leg2 profile (BB_ROW_LEN) + leg2 ux,uy,p0x,p0y (4) + radius, time_offset,
# duration (3).
TRAJ_LEG1_START = 0
TRAJ_LEG1_UX = BB_ROW_LEN
TRAJ_LEG1_UY = BB_ROW_LEN + 1
TRAJ_LEG1_P0X = BB_ROW_LEN + 2
TRAJ_LEG1_P0Y = BB_ROW_LEN + 3
TRAJ_SWITCH_T = BB_ROW_LEN + 4
TRAJ_LEG2_START = BB_ROW_LEN + 5
TRAJ_LEG2_UX = 2 * BB_ROW_LEN + 5
TRAJ_LEG2_UY = 2 * BB_ROW_LEN + 6
TRAJ_LEG2_P0X = 2 * BB_ROW_LEN + 7
TRAJ_LEG2_P0Y = 2 * BB_ROW_LEN + 8
TRAJ_RADIUS = 2 * BB_ROW_LEN + 9
TRAJ_TIME_OFFSET = 2 * BB_ROW_LEN + 10
TRAJ_DURATION = 2 * BB_ROW_LEN + 11
TRAJ_ROW_LEN = 2 * BB_ROW_LEN + 12

# ENEMY obstacle row: p0x, p0y, dirx, diry, speed, radius, t_max (7) + f_plus
# profile (BB_ROW_LEN) + f_minus profile (BB_ROW_LEN).
(EN_P0X, EN_P0Y, EN_DIRX, EN_DIRY, EN_SPEED, EN_RADIUS, EN_TMAX) = range(7)
EN_FPLUS_START = 7
EN_FMINUS_START = 7 + BB_ROW_LEN
ENEMY_ROW_LEN = 7 + 2 * BB_ROW_LEN

STATIC_ROW_LEN = 5  # ax, ay, bx, by, radius
CV_ROW_LEN = 5  # p0x, p0y, vx, vy, radius


def bangbang_row(
    p0: float,
    v0: float,
    p1: float,
    v_max: float,
    a_max: float,
    t1: float,
    t2: float,
    t_end: float,
    sign: float,
    v_cruise: float,
) -> np.ndarray:
    row = np.empty(BB_ROW_LEN, dtype=np.float64)
    row[BB_P0] = p0
    row[BB_V0] = v0
    row[BB_P1] = p1
    row[BB_VMAX] = v_max
    row[BB_AMAX] = a_max
    row[BB_T1] = t1
    row[BB_T2] = t2
    row[BB_TEND] = t_end
    row[BB_SIGN] = sign
    row[BB_VCRUISE] = v_cruise
    return row


@njit(cache=True, fastmath=True, inline="always")
def _bangbang_state_at(bb: np.ndarray, t: float) -> Tuple[float, float]:
    """Line-for-line port of `BangBang1D.state_at` -- see that method's
    docstring for the phase-by-phase derivation. Must be kept in exact sync
    with it; there is no shared source, only shared intent.
    """
    p0 = bb[BB_P0]
    v0 = bb[BB_V0]
    p1 = bb[BB_P1]
    a = bb[BB_AMAX]
    t1 = bb[BB_T1]
    t2 = bb[BB_T2]
    t_end = bb[BB_TEND]
    sign = bb[BB_SIGN]
    v_cruise = bb[BB_VCRUISE]

    if t <= 0.0:
        return p0, v0
    if t >= t_end:
        return p1, 0.0

    v0_signed = v0 * sign
    t_kill = max(0.0, -v0_signed / a) if v0_signed < 0.0 else 0.0
    if t < t_kill:
        v_signed = v0_signed + a * t
        d_signed = v0_signed * t + 0.5 * a * t * t
        return p0 + sign * d_signed, v_signed * sign

    d_kill = -(v0_signed * v0_signed) / (2.0 * a) if v0_signed < 0.0 else 0.0
    v_after_kill = 0.0 if v0_signed < 0.0 else v0_signed
    t_rel = t - t_kill

    if t < t1:
        v_signed = v_after_kill + a * t_rel
        d_signed = d_kill + v_after_kill * t_rel + 0.5 * a * t_rel * t_rel
        return p0 + sign * d_signed, v_signed * sign

    d_acc = d_kill + v_after_kill * (t1 - t_kill) + 0.5 * a * (t1 - t_kill) ** 2
    v_peak = abs(v_cruise)

    if t < t2:
        t_cruise_rel = t - t1
        d_signed = d_acc + v_peak * t_cruise_rel
        return p0 + sign * d_signed, v_peak * sign

    d_cruise = v_peak * (t2 - t1)
    t_dec_rel = t - t2
    v_signed = v_peak - a * t_dec_rel
    d_signed = d_acc + d_cruise + v_peak * t_dec_rel - 0.5 * a * t_dec_rel * t_dec_rel
    return p0 + sign * d_signed, v_signed * sign


@njit(cache=True, fastmath=True, inline="always")
def _trajectory2d_state_at(
    along: np.ndarray, ux: float, uy: float, p0x: float, p0y: float, t: float
) -> Tuple[float, float, float, float]:
    """Port of `Trajectory2D.state_at`: one 1D bang-bang profile mapped back
    onto x/y by the direction unit vector. Returns (px, py, vx, vy).
    """
    d, v = _bangbang_state_at(along, t)
    px = p0x + ux * d
    py = p0y + uy * d
    return px, py, v * ux, v * uy


@njit(cache=True, fastmath=True, inline="always")
def _query_trajectory_state(
    leg1: np.ndarray,
    ux1: float,
    uy1: float,
    p0x1: float,
    p0y1: float,
    switch_t: float,
    leg2: np.ndarray,
    ux2: float,
    uy2: float,
    p0x2: float,
    p0y2: float,
    t: float,
) -> Tuple[float, float, float, float]:
    """Port of `TwoSegmentTrajectory.state_at` (a plain `Trajectory2D`'s
    direct path is represented by passing `switch_t = leg1's own duration`
    and an unused/dummy `leg2`, so `t <= switch_t` always holds and `leg2`
    is never evaluated -- see `planner.py`'s array-building code).
    """
    if t <= switch_t:
        return _trajectory2d_state_at(leg1, ux1, uy1, p0x1, p0y1, t)
    return _trajectory2d_state_at(leg2, ux2, uy2, p0x2, p0y2, t - switch_t)


@njit(cache=True, fastmath=True, inline="always")
def _static_distance_at(row: np.ndarray, px: float, py: float) -> float:
    ax, ay, bx, by, radius = row[0], row[1], row[2], row[3], row[4]
    abx, aby = bx - ax, by - ay
    seg_len_sq = abx * abx + aby * aby
    if seg_len_sq < 1e-12:
        d = math.hypot(px - ax, py - ay)
    else:
        u = ((px - ax) * abx + (py - ay) * aby) / seg_len_sq
        u = max(0.0, min(1.0, u))
        cx, cy = ax + u * abx, ay + u * aby
        d = math.hypot(px - cx, py - cy)
    return d - radius


@njit(cache=True, fastmath=True, inline="always")
def _traj_obstacle_distance_at(row: np.ndarray, t: float, px: float, py: float) -> float:
    leg1 = row[TRAJ_LEG1_START : TRAJ_LEG1_START + BB_ROW_LEN]
    ux1 = row[TRAJ_LEG1_UX]
    uy1 = row[TRAJ_LEG1_UY]
    p0x1 = row[TRAJ_LEG1_P0X]
    p0y1 = row[TRAJ_LEG1_P0Y]
    switch_t = row[TRAJ_SWITCH_T]
    leg2 = row[TRAJ_LEG2_START : TRAJ_LEG2_START + BB_ROW_LEN]
    ux2 = row[TRAJ_LEG2_UX]
    uy2 = row[TRAJ_LEG2_UY]
    p0x2 = row[TRAJ_LEG2_P0X]
    p0y2 = row[TRAJ_LEG2_P0Y]
    radius = row[TRAJ_RADIUS]
    time_offset = row[TRAJ_TIME_OFFSET]
    duration = row[TRAJ_DURATION]
    # `_CommittedTrajectoryObstacle.distance_at`: query_t = clamp(t +
    # time_offset, 0, duration). `OwnRobotObstacle` (no offset, only clamps
    # to duration) is the `time_offset == 0.0` case of the exact same
    # formula, so both are represented by one row shape. The wrapped
    # trajectory itself may be a plain `Trajectory2D` (leg2 unused, since
    # `switch_t` is set to leg1's own duration) or a `TwoSegmentTrajectory`
    # -- see `_flatten_obstacles`'s traj-row construction.
    query_t = t + time_offset
    if query_t < 0.0:
        query_t = 0.0
    elif query_t > duration:
        query_t = duration
    ox, oy, _, _ = _query_trajectory_state(leg1, ux1, uy1, p0x1, p0y1, switch_t, leg2, ux2, uy2, p0x2, p0y2, query_t)
    return math.hypot(px - ox, py - oy) - radius


@njit(cache=True, fastmath=True, inline="always")
def _cv_distance_at(row: np.ndarray, t: float, px: float, py: float) -> float:
    p0x, p0y, vx, vy, radius = row[0], row[1], row[2], row[3], row[4]
    ox = p0x + vx * t
    oy = p0y + vy * t
    return math.hypot(px - ox, py - oy) - radius


@njit(cache=True, fastmath=True, inline="always")
def _enemy_distance_at(row: np.ndarray, t: float, px: float, py: float) -> float:
    p0x = row[EN_P0X]
    p0y = row[EN_P0Y]
    dirx = row[EN_DIRX]
    diry = row[EN_DIRY]
    speed = row[EN_SPEED]
    radius = row[EN_RADIUS]
    t_max = row[EN_TMAX]
    f_plus = row[EN_FPLUS_START : EN_FPLUS_START + BB_ROW_LEN]
    f_minus = row[EN_FMINUS_START : EN_FMINUS_START + BB_ROW_LEN]

    tc = t
    if tc < 0.0:
        tc = 0.0
    elif tc > t_max:
        tc = t_max
    fp, _ = _bangbang_state_at(f_plus, tc)
    fm, _ = _bangbang_state_at(f_minus, tc)
    r_dyn = abs(fp - fm) / 2.0

    if speed < 1e-6:
        cx, cy = p0x, p0y
    else:
        offset = fm + r_dyn
        cx = p0x + dirx * offset
        cy = p0y + diry * offset
    return math.hypot(px - cx, py - cy) - (radius + r_dyn)


@njit(cache=True, fastmath=True)
def first_collision_numba(
    leg1: np.ndarray,
    ux1: float,
    uy1: float,
    p0x1: float,
    p0y1: float,
    switch_t: float,
    leg2: np.ndarray,
    ux2: float,
    uy2: float,
    p0x2: float,
    p0y2: float,
    query_duration: float,
    start_t: float,
    max_lookahead_time: float,
    robot_radius: float,
    margin_v_max: float,
    margin_base: float,
    max_time_step: float,
    min_time_step: float,
    step_distance_ratio: float,
    static_obs: np.ndarray,  # shape (n_static, STATIC_ROW_LEN)
    traj_obs: np.ndarray,  # shape (n_traj, TRAJ_ROW_LEN)
    cv_obs: np.ndarray,  # shape (n_cv, CV_ROW_LEN)
    enemy_obs: np.ndarray,  # shape (n_enemy, ENEMY_ROW_LEN)
) -> float:
    """Native port of `TrajectorySamplingPlanner._first_collision`'s scan
    loop -- see that method's docstring for the exact semantics being
    preserved (margin formula, own-radius subtraction, adaptive stepping,
    `MAX_LOOKAHEAD_TIME` clamp). Returns the absolute collision time, or
    -1.0 if clear (Python wrapper converts to `Optional[float]`).
    """
    t = start_t
    duration = min(query_duration, max_lookahead_time)
    n_static = static_obs.shape[0]
    n_traj = traj_obs.shape[0]
    n_cv = cv_obs.shape[0]
    n_enemy = enemy_obs.shape[0]

    # A robot can start a plan already inside another (moving) obstacle's
    # clearance envelope -- e.g. GiveAndGoTactic's abandoned-receiver robot,
    # left standing right where it was waiting to catch a pass, replanning a
    # fresh route the instant the ball becomes a real obstacle again. Without
    # this, `d < margin` at the very first sample (t == start_t) fires for
    # EVERY candidate direction regardless of where it's headed -- the
    # robot's own starting position is the collision, not anything about the
    # trajectory -- so the planner can never find an escape and the robot
    # stays pinned near-zero velocity indefinitely (found live, 2026-09-03:
    # a robot 0.088m from the ball, inside the 0.09+0.0215=0.1115m combined
    # radius, reporting collision_time~=0 against every sampled direction).
    # Fixed by giving each obstacle a one-time "still escaping" grace at the
    # start of the scan: an obstacle already violating clearance at t ==
    # start_t doesn't veto the trajectory outright until the robot has first
    # cleared it (`d >= margin`) at some later sample -- from that point on
    # it's checked normally, including re-triggering if the trajectory turns
    # back into it. An obstacle that ISN'T penetrated at start_t is never
    # granted this grace, so a normal approaching trajectory is unaffected.
    static_escaping = np.ones(n_static, dtype=np.bool_)
    traj_escaping = np.ones(n_traj, dtype=np.bool_)
    cv_escaping = np.ones(n_cv, dtype=np.bool_)
    enemy_escaping = np.ones(n_enemy, dtype=np.bool_)
    first_sample = True

    while t <= duration:
        px, py, vx, vy = _query_trajectory_state(leg1, ux1, uy1, p0x1, p0y1, switch_t, leg2, ux2, uy2, p0x2, p0y2, t)
        speed = math.hypot(vx, vy)
        margin = (min(speed, margin_v_max) / margin_v_max) ** 2 * margin_base

        closest = math.inf
        collided = False

        for i in range(n_static):
            d = _static_distance_at(static_obs[i], px, py) - robot_radius
            still_penetrating = d < margin
            if first_sample:
                static_escaping[i] = still_penetrating
            elif static_escaping[i]:
                static_escaping[i] = still_penetrating
            if still_penetrating and not static_escaping[i]:
                collided = True
                break
            if d < closest:
                closest = d
        if not collided:
            for i in range(n_traj):
                d = _traj_obstacle_distance_at(traj_obs[i], t, px, py) - robot_radius
                still_penetrating = d < margin
                if first_sample:
                    traj_escaping[i] = still_penetrating
                elif traj_escaping[i]:
                    traj_escaping[i] = still_penetrating
                if still_penetrating and not traj_escaping[i]:
                    collided = True
                    break
                if d < closest:
                    closest = d
        if not collided:
            for i in range(n_cv):
                d = _cv_distance_at(cv_obs[i], t, px, py) - robot_radius
                still_penetrating = d < margin
                if first_sample:
                    cv_escaping[i] = still_penetrating
                elif cv_escaping[i]:
                    cv_escaping[i] = still_penetrating
                if still_penetrating and not cv_escaping[i]:
                    collided = True
                    break
                if d < closest:
                    closest = d
        if not collided:
            for i in range(n_enemy):
                d = _enemy_distance_at(enemy_obs[i], t, px, py) - robot_radius
                still_penetrating = d < margin
                if first_sample:
                    enemy_escaping[i] = still_penetrating
                elif enemy_escaping[i]:
                    enemy_escaping[i] = still_penetrating
                if still_penetrating and not enemy_escaping[i]:
                    collided = True
                    break
                if d < closest:
                    closest = d

        first_sample = False

        if collided:
            return t

        if closest == math.inf:
            step = max_time_step
        else:
            step = closest * step_distance_ratio
            if step < min_time_step:
                step = min_time_step
            elif step > max_time_step:
                step = max_time_step

        if t >= duration:
            break
        t = min(t + step, duration)

    return -1.0


def empty_row_array(row_len: int) -> np.ndarray:
    return np.empty((0, row_len), dtype=np.float64)
