"""Numba-jitted obstacle-scan kernels for FastPathPlanner's hot inner loops.

`collides()` and `_find_subgoal()` (planner.py) both do a bounding-box-pruned
linear scan over the same flat list of obstacle line segments, computing
`distance_point_to_segment`/`distance_between_line_segments` per surviving
candidate -- confirmed via cProfile on a real 65s match to be the largest
single cost in FastPathPlanner (`collides`: 8.3s cumtime/190917 calls,
`_find_subgoal`: 4.5s cumtime/248714 calls).

IMPORTANT lesson (measured directly, see `/home/isaac/.claude/jobs/d50ccf8c/
tmp/numba_fpp_bench.py`): decorating one tiny leaf function (e.g. a single
`distance_point_to_segment` call) with `@njit` and calling it from a Python
`for` loop does NOT reliably pay off -- the Python<->native call-boundary
cost is paid on every iteration, and for FPP's segment-distance math
specifically the benchmark showed only a mild 1.06x-1.13x gain that way
(FPP's math is heavier per call than e.g. a trajectory `state_at` lookup, so
this particular anti-pattern is less catastrophic here than it was for the
trajectory-sampling planner, but it's still not where the real win is). The
real win (5.6x-16.5x for `collides`-style scans, 4.0x-13.3x for
`_find_subgoal`-style scans in that same benchmark) comes from batching the
ENTIRE obstacle scan -- bounding-box prune plus distance math for every
obstacle -- into ONE `@njit` call per `collides()`/`_find_subgoal()`
invocation, operating on flat float64 arrays instead of a Python list of
`(np.ndarray, np.ndarray)` tuples. That's what this module provides.

Obstacles are flattened to four parallel float64 arrays (`ox0, oy0, ox1,
oy1`, one row per segment) once per `_path_to()` call (see
`planner.py`'s `_obstacle_arrays` cache, keyed by `id(obstacles)` since the
same filtered obstacle list is reused across many `collides()`/
`check_segment()` calls within one `_path_to()` call but a fresh list object
is built each such call) rather than once per scan.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

EPS = 1e-9


@njit(cache=True, fastmath=True)
def distance_point_to_segment_nb(px: float, py: float, sx: float, sy: float, ex: float, ey: float) -> float:
    """Exact port of `math_utils.distance_point_to_segment`'s formula."""
    seg_dx = ex - sx
    seg_dy = ey - sy
    pt_dx = px - sx
    pt_dy = py - sy
    seg_len_sq = seg_dx * seg_dx + seg_dy * seg_dy
    if seg_len_sq < EPS:
        return math.sqrt((px - sx) ** 2 + (py - sy) ** 2)
    t = (pt_dx * seg_dx + pt_dy * seg_dy) / seg_len_sq
    if t < 0.0:
        cx, cy = sx, sy
    elif t > 1.0:
        cx, cy = ex, ey
    else:
        cx, cy = sx + t * seg_dx, sy + t * seg_dy
    return math.sqrt((px - cx) ** 2 + (py - cy) ** 2)


@njit(cache=True, fastmath=True)
def _point_orientation_nb(p1x: float, p1y: float, p2x: float, p2y: float, p3x: float, p3y: float) -> int:
    """Exact port of `math_utils.point_orientation`."""
    val = (p2x - p1x) * (p3y - p1y) - (p2y - p1y) * (p3x - p1x)
    if abs(val) < EPS:
        return 0
    return 1 if val < 0.0 else 2


@njit(cache=True, fastmath=True)
def _on_segment_nb(px: float, py: float, qx: float, qy: float, rx: float, ry: float) -> bool:
    """Exact port of `math_utils.on_segment` (does q lie on segment p-r)."""
    return min(px, rx) - EPS <= qx <= max(px, rx) + EPS and min(py, ry) - EPS <= qy <= max(py, ry) + EPS


@njit(cache=True, fastmath=True)
def _segments_intersect_nb(
    p1x: float, p1y: float, q1x: float, q1y: float, p2x: float, p2y: float, q2x: float, q2y: float
) -> bool:
    """Exact port of `math_utils.segments_intersect`."""
    o1 = _point_orientation_nb(p1x, p1y, q1x, q1y, p2x, p2y)
    o2 = _point_orientation_nb(p1x, p1y, q1x, q1y, q2x, q2y)
    o3 = _point_orientation_nb(p2x, p2y, q2x, q2y, p1x, p1y)
    o4 = _point_orientation_nb(p2x, p2y, q2x, q2y, q1x, q1y)

    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and _on_segment_nb(p1x, p1y, p2x, p2y, q1x, q1y):
        return True
    if o2 == 0 and _on_segment_nb(p1x, p1y, q2x, q2y, q1x, q1y):
        return True
    if o3 == 0 and _on_segment_nb(p2x, p2y, p1x, p1y, q2x, q2y):
        return True
    if o4 == 0 and _on_segment_nb(p2x, p2y, q1x, q1y, q2x, q2y):
        return True
    return False


@njit(cache=True, fastmath=True)
def _distance_between_segments_nb(
    ax0: float, ay0: float, ax1: float, ay1: float, bx0: float, by0: float, bx1: float, by1: float
) -> float:
    """Exact port of `math_utils.distance_between_line_segments`."""
    if _segments_intersect_nb(ax0, ay0, ax1, ay1, bx0, by0, bx1, by1):
        return 0.0
    d1 = distance_point_to_segment_nb(ax0, ay0, bx0, by0, bx1, by1)
    d2 = distance_point_to_segment_nb(ax1, ay1, bx0, by0, bx1, by1)
    d3 = distance_point_to_segment_nb(bx0, by0, ax0, ay0, ax1, ay1)
    d4 = distance_point_to_segment_nb(bx1, by1, ax0, ay0, ax1, ay1)
    return min(d1, d2, d3, d4)


@njit(cache=True, fastmath=True)
def scan_collides_nb(
    seg_x0: float,
    seg_y0: float,
    seg_x1: float,
    seg_y1: float,
    clearance: float,
    ox0: np.ndarray,
    oy0: np.ndarray,
    ox1: np.ndarray,
    oy1: np.ndarray,
    sticky_idx: int,
) -> tuple:
    """Batched replacement for `collides()`'s obstacle-scan loop (bounding-box
    prune + `distance_between_line_segments` on survivors, tracking the
    obstacle closest to the segment's START point). `sticky_idx` is the index
    into the obstacle arrays of the sticky obstacle, or -1 if none/not
    applicable -- returns `sticky_dist_to_robot` as -1.0 when there is no
    sticky obstacle or it never entered range, mirroring the Python
    `Optional[float]` semantics via a sentinel (converted back to `None` by
    the caller).

    Returns `(closest_idx, min_dist_to_robot, sticky_dist_to_robot)` --
    `closest_idx` is -1 if nothing is within `clearance`.
    """
    seg_min_x = min(seg_x0, seg_x1) - clearance
    seg_max_x = max(seg_x0, seg_x1) + clearance
    seg_min_y = min(seg_y0, seg_y1) - clearance
    seg_max_y = max(seg_y0, seg_y1) + clearance

    closest_idx = -1
    min_dist_to_robot = 1e18
    sticky_dist_to_robot = -1.0

    n = ox0.shape[0]
    for i in range(n):
        ox0i, oy0i, ox1i, oy1i = ox0[i], oy0[i], ox1[i], oy1[i]
        o_min_x = min(ox0i, ox1i)
        o_max_x = max(ox0i, ox1i)
        if o_max_x < seg_min_x or o_min_x > seg_max_x:
            continue
        o_min_y = min(oy0i, oy1i)
        o_max_y = max(oy0i, oy1i)
        if o_max_y < seg_min_y or o_min_y > seg_max_y:
            continue

        dist_between_segs = _distance_between_segments_nb(ox0i, oy0i, ox1i, oy1i, seg_x0, seg_y0, seg_x1, seg_y1)
        if dist_between_segs < clearance:
            dist_to_robot = distance_point_to_segment_nb(seg_x0, seg_y0, ox0i, oy0i, ox1i, oy1i)
            if i == sticky_idx:
                sticky_dist_to_robot = dist_to_robot
            if dist_to_robot < min_dist_to_robot:
                min_dist_to_robot = dist_to_robot
                closest_idx = i

    return closest_idx, min_dist_to_robot, sticky_dist_to_robot


@njit(cache=True, fastmath=True)
def scan_find_subgoal_nb(
    sub_x: float,
    sub_y: float,
    clearance: float,
    ox0: np.ndarray,
    oy0: np.ndarray,
    ox1: np.ndarray,
    oy1: np.ndarray,
) -> int:
    """Batched replacement for `_find_subgoal`'s obstacle-scan loop
    (bounding-box prune + `distance_point_to_segment` against `subgoal`).
    Returns the index of the first obstacle found within `clearance` (scan
    order matches the original Python `for o in obstacles` iteration order,
    since `_find_subgoal` returns on the FIRST hit, not the closest), or -1
    if none.
    """
    box_min_x = sub_x - clearance
    box_max_x = sub_x + clearance
    box_min_y = sub_y - clearance
    box_max_y = sub_y + clearance

    n = ox0.shape[0]
    for i in range(n):
        ox0i, oy0i, ox1i, oy1i = ox0[i], oy0[i], ox1[i], oy1[i]
        o_min_x = min(ox0i, ox1i)
        o_max_x = max(ox0i, ox1i)
        if o_max_x < box_min_x or o_min_x > box_max_x:
            continue
        o_min_y = min(oy0i, oy1i)
        o_max_y = max(oy0i, oy1i)
        if o_max_y < box_min_y or o_min_y > box_max_y:
            continue

        if distance_point_to_segment_nb(sub_x, sub_y, ox0i, oy0i, ox1i, oy1i) < clearance:
            return i

    return -1


def flatten_obstacles(obstacles) -> tuple:
    """Convert a `List[Tuple[np.ndarray, np.ndarray]]` obstacle list into
    four parallel float64 arrays `(ox0, oy0, ox1, oy1)` for the njit kernels
    above. Called once per distinct obstacle-list object (see
    `planner.py`'s `_obstacle_arrays` cache), not once per scan.
    """
    n = len(obstacles)
    ox0 = np.empty(n, dtype=np.float64)
    oy0 = np.empty(n, dtype=np.float64)
    ox1 = np.empty(n, dtype=np.float64)
    oy1 = np.empty(n, dtype=np.float64)
    for i, (a, b) in enumerate(obstacles):
        ox0[i] = a[0]
        oy0[i] = a[1]
        ox1[i] = b[0]
        oy1[i] = b[1]
    return ox0, oy0, ox1, oy1
