"""Speed rewrites of `FastPathPlanner` internals must not change a single bit
of its output: rsim tournaments are deterministic, and a replay is only
comparable across commits if the planner is.

`_ReferencePlanner` keeps the pre-rewrite implementations verbatim (logic
only, comments trimmed). Each test drives it and the live planner through the
same randomized scenes -- robots with projected velocity segments, the field
walls and the inflated enemy box, stateful sticky/side memory carried across
ticks -- and requires exactly equal results.
"""

import math

import numpy as np
import pytest

from utama_core.config.field_params import STANDARD_FIELD_DIMS
from utama_core.global_utils.math_utils import (
    closest_point_on_segment,
    distance,
    distance_point_to_segment,
    find_intersection,
    rotate_vector,
)
from utama_core.motion_planning.src.fastpathplanning.numba_kernels import (
    scan_collides_nb,
    scan_find_subgoal_nb,
)
from utama_core.motion_planning.src.fastpathplanning.planner import (
    FastPathPlanner,
    _same_segment,
)

_RECT = (3.25, 4.75, -1.25, 1.25)


class _Bounds:
    top_left = (-STANDARD_FIELD_DIMS.full_field_half_length, STANDARD_FIELD_DIMS.full_field_half_width)
    bottom_right = (STANDARD_FIELD_DIMS.full_field_half_length, -STANDARD_FIELD_DIMS.full_field_half_width)


class _ReferencePlanner(FastPathPlanner):
    def _find_subgoal(
        self,
        robot_pos,
        target,
        obstacle_pos,
        obstacles,
        subgoal_direction,
        multiple,
        clearance,
        subgoal_distance,
        origin_obstacle=None,
        blocked_by_origin=False,
        forbidden_rect=None,
    ):
        if multiple > 10:
            return None if blocked_by_origin else obstacle_pos
        direction = target - robot_pos
        direction_norm = math.hypot(direction[0], direction[1])
        if direction_norm == 0.0:
            return obstacle_pos
        perp_dir = rotate_vector(direction[0], direction[1], math.pi * (subgoal_direction + 0.5))
        unitvec = np.array([perp_dir[0] / direction_norm, perp_dir[1] / direction_norm])
        subgoal = obstacle_pos + subgoal_distance * unitvec * multiple
        sub_x, sub_y = subgoal[0], subgoal[1]
        if forbidden_rect is not None:
            min_x, max_x, min_y, max_y = forbidden_rect
            if min_x < sub_x < max_x and min_y < sub_y < max_y:
                return self._find_subgoal(
                    robot_pos,
                    target,
                    obstacle_pos,
                    obstacles,
                    subgoal_direction,
                    multiple + 1,
                    clearance,
                    subgoal_distance,
                    origin_obstacle=origin_obstacle,
                    blocked_by_origin=True,
                    forbidden_rect=forbidden_rect,
                )
        ox0, oy0, ox1, oy1 = self._obstacle_arrays(obstacles)
        hit_idx = scan_find_subgoal_nb(sub_x, sub_y, clearance, ox0, oy0, ox1, oy1)
        if hit_idx >= 0:
            o = obstacles[hit_idx]
            is_origin = origin_obstacle is not None and (
                np.array_equal(o[0], origin_obstacle[0]) and np.array_equal(o[1], origin_obstacle[1])
            )
            return self._find_subgoal(
                robot_pos,
                target,
                obstacle_pos,
                obstacles,
                subgoal_direction,
                multiple + 1,
                clearance,
                subgoal_distance,
                origin_obstacle=origin_obstacle,
                blocked_by_origin=is_origin,
                forbidden_rect=forbidden_rect,
            )
        return subgoal

    def collides(self, segment, obstacles, sticky_obstacle=None, clearance=None):
        clearance = self.OBSTACLE_CLEARANCE if clearance is None else clearance
        sticky_key = (tuple(sticky_obstacle[0]), tuple(sticky_obstacle[1])) if sticky_obstacle is not None else None
        seg_key = (tuple(segment[0]), tuple(segment[1]), sticky_key, clearance)
        if seg_key in self._collision_cache:
            return self._collision_cache[seg_key]
        sticky_idx = -1
        if sticky_obstacle is not None:
            for i, o in enumerate(obstacles):
                if np.array_equal(o[0], sticky_obstacle[0]) and np.array_equal(o[1], sticky_obstacle[1]):
                    sticky_idx = i
                    break
        ox0, oy0, ox1, oy1 = self._obstacle_arrays(obstacles)
        seg_x0, seg_y0 = float(segment[0][0]), float(segment[0][1])
        seg_x1, seg_y1 = float(segment[1][0]), float(segment[1][1])
        closest_idx, min_dist_to_robot, sticky_dist_to_robot_raw = scan_collides_nb(
            seg_x0, seg_y0, seg_x1, seg_y1, clearance, ox0, oy0, ox1, oy1, sticky_idx
        )
        closest_obstacle = obstacles[closest_idx] if closest_idx >= 0 else None
        sticky_dist_to_robot = sticky_dist_to_robot_raw if sticky_dist_to_robot_raw >= 0.0 else None
        if (
            sticky_obstacle is not None
            and sticky_dist_to_robot is not None
            and closest_obstacle is not None
            and not (
                np.array_equal(closest_obstacle[0], sticky_obstacle[0])
                and np.array_equal(closest_obstacle[1], sticky_obstacle[1])
            )
            and sticky_dist_to_robot <= min_dist_to_robot + self.DETOUR_SWITCH_MARGIN
        ):
            closest_obstacle = sticky_obstacle
        obstacle_pos = None
        if closest_obstacle is not None:
            obstacle_pos = find_intersection(segment, closest_obstacle)
            if obstacle_pos is None:
                dists = [
                    distance_point_to_segment(closest_obstacle[0], segment[0], segment[1]),
                    distance_point_to_segment(closest_obstacle[1], segment[0], segment[1]),
                ]
                point_c = closest_point_on_segment(segment[0], closest_obstacle[0], closest_obstacle[1])
                point_d = closest_point_on_segment(segment[1], closest_obstacle[0], closest_obstacle[1])
                dists.extend([distance(segment[0], point_c), distance(segment[1], point_d)])
                points = [closest_obstacle[0], closest_obstacle[1], point_c, point_d]
                obstacle_pos = points[dists.index(min(dists))]
        result = (obstacle_pos, closest_obstacle)
        self._collision_cache[seg_key] = result
        return result

    def sanitize_target(self, target, obstacles, robot_pos, field_bounds=None, exempt_obstacles=None, clearance=None):
        if exempt_obstacles is None:
            exempt_obstacles = set()
        clearance = self.OBSTACLE_CLEARANCE if clearance is None else clearance
        if field_bounds is not None:
            tl = np.array(field_bounds.top_left)
            br = np.array(field_bounds.bottom_right)
            tr = np.array([br[0], tl[1]])
            bl = np.array([tl[0], br[1]])
            boundary_segments = {
                (tuple(tl), tuple(tr)),
                (tuple(tr), tuple(br)),
                (tuple(br), tuple(bl)),
                (tuple(bl), tuple(tl)),
            }
        else:
            boundary_segments = set()
        safe_target = np.copy(target)
        for _ in range(5):
            collision_found = False
            for o in obstacles:
                o_key = (tuple(o[0]), tuple(o[1]))
                if o_key in boundary_segments or o_key in exempt_obstacles:
                    continue
                if distance_point_to_segment(safe_target, o[0], o[1]) < clearance:
                    closest_pt = closest_point_on_segment(safe_target, o[0], o[1])
                    push_dir = safe_target - closest_pt
                    if math.hypot(push_dir[0], push_dir[1]) == 0:
                        push_dir = robot_pos - closest_pt
                    unit_push = push_dir / math.hypot(push_dir[0], push_dir[1])
                    safe_target = closest_pt + unit_push * (clearance * 1.05)
                    collision_found = True
            if not collision_found:
                break
        return safe_target


def _scene(rng, n_robots):
    """Robot ghost-wall segments plus the 8 static segments `_refresh_obstacle_cache` adds."""
    obstacles = []
    for _ in range(n_robots):
        p = rng.uniform([-4.5, -3.0], [4.5, 3.0])
        v = rng.normal(0.0, 1.0, 2) * (rng.random() < 0.7)
        obstacles.append((p, p + v * 0.25))
    tl, br = np.array(_Bounds.top_left), np.array(_Bounds.bottom_right)
    tr, bl = np.array([br[0], tl[1]]), np.array([tl[0], br[1]])
    min_x, max_x, min_y, max_y = _RECT
    c = [np.array(xy) for xy in ((min_x, max_y), (max_x, max_y), (max_x, min_y), (min_x, min_y))]
    obstacles += [(tl, tr), (tr, br), (br, bl), (bl, tl), (c[0], c[1]), (c[1], c[2]), (c[2], c[3]), (c[3], c[0])]
    return obstacles


def _same(a, b):
    if a is None or b is None:
        return a is None and b is None
    return np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)


def _same_trajectory(t1, t2):
    return len(t1) == len(t2) and all(_same(s1[0], s2[0]) and _same(s1[1], s2[1]) for s1, s2 in zip(t1, t2))


@pytest.mark.parametrize("seed", range(6))
def test_check_segment_matches_reference_across_ticks(seed):
    """Stateful: sticky obstacles and detour sides carry over between ticks, so
    the robots drift a little each tick rather than jumping to a new scene."""
    rng = np.random.default_rng(seed)
    new, ref = FastPathPlanner(env=None), _ReferencePlanner(env=None)
    obstacles = _scene(rng, n_robots=int(rng.integers(3, 12)))
    robots = {rid: rng.uniform([-4.4, -2.9], [4.4, 2.9]) for rid in range(4)}
    targets = {rid: rng.uniform([-4.4, -2.9], [4.4, 2.9]) for rid in range(4)}
    for tick in range(40):
        n_dyn = len(obstacles) - 8
        jitter = rng.normal(0.0, 0.01, (n_dyn, 2, 2))
        obstacles = [(a + j[0], b + j[1]) for (a, b), j in zip(obstacles[:n_dyn], jitter)] + obstacles[n_dyn:]
        for rid in robots:
            robots[rid] = robots[rid] + rng.normal(0.0, 0.02, 2)
            if rng.random() < 0.05:
                targets[rid] = rng.uniform([-4.4, -2.9], [4.4, 2.9])
            results = []
            for planner in (new, ref):
                planner._collision_cache.clear()
                planner._obstacle_arrays_cache.clear()
                results.append(
                    planner.check_segment(
                        (robots[rid], targets[rid]),
                        obstacles,
                        0,
                        targets[rid],
                        _Bounds,
                        robot_id=rid,
                        clearance=planner.OBSTACLE_CLEARANCE * (0.6 + 0.4 * (tick % 2)),
                        subgoal_distance=planner.SUBGOAL_DISTANCE,
                        forbidden_rect=_RECT if rid % 2 else None,
                    )
                )
            (traj_new, len_new), (traj_ref, len_ref) = results
            assert _same_trajectory(traj_new, traj_ref), (tick, rid)
            assert len_new == len_ref or (math.isnan(len_new) and math.isnan(len_ref))
        assert new._last_detour_side == ref._last_detour_side
        assert new._last_obstacle.keys() == ref._last_obstacle.keys()
        for key, seg in new._last_obstacle.items():
            assert _same(seg[0], ref._last_obstacle[key][0]) and _same(seg[1], ref._last_obstacle[key][1])


@pytest.mark.parametrize("seed", range(4))
def test_find_subgoal_matches_reference(seed):
    """Direct calls, covering what `check_segment` rarely reaches: entry past the
    step budget, a pre-set `blocked_by_origin`, the origin obstacle blocking
    at the cutoff, and collapsed (zero-length) directions."""
    rng = np.random.default_rng(100 + seed)
    new, ref = FastPathPlanner(env=None), _ReferencePlanner(env=None)
    for _ in range(300):
        obstacles = _scene(rng, n_robots=int(rng.integers(0, 12)))
        robot_pos = rng.uniform([-4.4, -2.9], [4.4, 2.9])
        target = robot_pos.copy() if rng.random() < 0.05 else rng.uniform([-4.4, -2.9], [4.4, 2.9])
        origin = obstacles[int(rng.integers(len(obstacles)))]
        if rng.random() < 0.3:
            origin = (origin[0].copy(), origin[1].copy())  # equal by value, not identity
        obstacle_pos = origin[0] + rng.random() * (origin[1] - origin[0])
        kwargs = dict(
            subgoal_direction=int(rng.integers(2)),
            multiple=int(rng.choice([1, 1, 1, 5, 10, 11])),
            clearance=float(rng.uniform(0.2, 0.5)),
            subgoal_distance=float(rng.uniform(0.1, 0.6)),
            origin_obstacle=origin if rng.random() < 0.8 else None,
            blocked_by_origin=bool(rng.random() < 0.5),
            forbidden_rect=_RECT if rng.random() < 0.5 else None,
        )
        a = new._find_subgoal(robot_pos, target, obstacle_pos, obstacles, **kwargs)
        b = ref._find_subgoal(robot_pos, target, obstacle_pos, obstacles, **kwargs)
        assert _same(a, b), kwargs


def test_same_segment_matches_array_equal_on_edge_values():
    values = [0.0, -0.0, 1.5, np.nan, np.inf, -np.inf, 1e-300]
    rng = np.random.default_rng(7)
    for _ in range(2000):
        a = (np.array(rng.choice(values, 2)), np.array(rng.choice(values, 2)))
        b = (np.array(rng.choice(values, 2)), np.array(rng.choice(values, 2)))
        expected = np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
        assert _same_segment(a, b) == expected
        assert _same_segment(a, a) == (np.array_equal(a[0], a[0]) and np.array_equal(a[1], a[1]))


@pytest.mark.parametrize("seed", range(4))
def test_sanitize_target_matches_reference(seed):
    """Targets placed near (often inside the clearance of) crowded obstacles, so
    the push-out loop runs several passes, with and without exemptions."""
    rng = np.random.default_rng(200 + seed)
    new, ref = FastPathPlanner(env=None), _ReferencePlanner(env=None)
    for _ in range(400):
        obstacles = _scene(rng, n_robots=int(rng.integers(0, 12)))
        anchor = obstacles[int(rng.integers(len(obstacles)))]
        target = anchor[0] + rng.normal(0.0, 0.15, 2)
        robot_pos = rng.uniform([-4.4, -2.9], [4.4, 2.9])
        exempt = {(tuple(o[0]), tuple(o[1])) for o in obstacles if rng.random() < 0.2}
        kwargs = dict(
            field_bounds=_Bounds if rng.random() < 0.8 else None,
            exempt_obstacles=exempt if rng.random() < 0.5 else None,
            clearance=float(rng.uniform(0.2, 0.5)),
        )
        a = new.sanitize_target(target, obstacles, robot_pos, **kwargs)
        b = ref.sanitize_target(target, obstacles, robot_pos, **kwargs)
        assert _same(a, b)
