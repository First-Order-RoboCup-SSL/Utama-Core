"""The fast numerics (`EXACT_MATH` off, the default) compute what the exact
ones do, up to rounding: on random scenes every fast planner step makes the
same choice (same obstacle, same None/not-None) and lands within 1e-9 m of
the exact result. fastpathplanner_equivalence_test.py pins the exact ones bit
for bit against the pre-rewrite code."""

import math

import numpy as np
import pytest

from utama_core.data_processing.refiners.filters import kalman
from utama_core.global_utils import math_utils
from utama_core.motion_planning.src.fastpathplanning.planner import (
    FastPathPlanner,
    _same_segment,
    _segment_key,
)

_RECT = (3.25, 4.75, -1.25, 1.25)
_TOL = 1e-9


class _Bounds:
    top_left = (-4.5, 3.0)
    bottom_right = (4.5, -3.0)


def _scene(rng, n_robots):
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


def _close(a, b):
    if a is None or b is None:
        return a is None and b is None
    return np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float), rtol=0.0, atol=_TOL, equal_nan=True)


@pytest.mark.parametrize("seed", range(4))
def test_fast_collides_picks_the_same_obstacle(seed):
    rng = np.random.default_rng(seed)
    planner = FastPathPlanner(env=None)
    for _ in range(500):
        obstacles = _scene(rng, n_robots=int(rng.integers(0, 12)))
        segment = (rng.uniform([-4.4, -2.9], [4.4, 2.9]), rng.uniform([-4.4, -2.9], [4.4, 2.9]))
        sticky = obstacles[int(rng.integers(len(obstacles)))] if rng.random() < 0.5 else None
        clearance = float(rng.uniform(0.2, 0.5))
        results = []
        for method in (FastPathPlanner._collides_fast, FastPathPlanner._collides_exact):
            planner._collision_cache.clear()
            planner._obstacle_arrays_cache.clear()
            results.append(method(planner, segment, obstacles, sticky_obstacle=sticky, clearance=clearance))
        (pos_fast, obs_fast), (pos_exact, obs_exact) = results
        assert (obs_fast is None) == (obs_exact is None)
        if obs_exact is not None:
            assert _same_segment(obs_fast, obs_exact)
        assert _close(pos_fast, pos_exact)


@pytest.mark.parametrize("seed", range(4))
def test_fast_find_subgoal_matches_exact(seed):
    rng = np.random.default_rng(100 + seed)
    planner = FastPathPlanner(env=None)
    for _ in range(500):
        obstacles = _scene(rng, n_robots=int(rng.integers(0, 12)))
        robot_pos = rng.uniform([-4.4, -2.9], [4.4, 2.9])
        target = robot_pos.copy() if rng.random() < 0.05 else rng.uniform([-4.4, -2.9], [4.4, 2.9])
        origin = obstacles[int(rng.integers(len(obstacles)))]
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
        fast = planner._find_subgoal_fast(robot_pos, target, obstacle_pos, obstacles, **kwargs)
        exact = planner._find_subgoal_exact(robot_pos, target, obstacle_pos, obstacles, **kwargs)
        assert _close(fast, exact), kwargs


@pytest.mark.parametrize("seed", range(4))
def test_fast_sanitize_target_matches_exact(seed):
    rng = np.random.default_rng(200 + seed)
    planner = FastPathPlanner(env=None)
    for _ in range(500):
        obstacles = _scene(rng, n_robots=int(rng.integers(0, 12)))
        anchor = obstacles[int(rng.integers(len(obstacles)))]
        target = anchor[0] + rng.normal(0.0, 0.15, 2)
        robot_pos = rng.uniform([-4.4, -2.9], [4.4, 2.9])
        exempt = {_segment_key(o) for o in obstacles if rng.random() < 0.2}
        kwargs = dict(
            field_bounds=_Bounds if rng.random() < 0.8 else None,
            exempt_obstacles=exempt if rng.random() < 0.5 else None,
            clearance=float(rng.uniform(0.2, 0.5)),
        )
        fast = planner._sanitize_target_fast(target, obstacles, robot_pos, **kwargs)
        exact = planner._sanitize_target_exact(target, obstacles, robot_pos, **kwargs)
        assert _close(fast, exact)


def test_fast_sanitize_target_degenerate_push_is_nan_not_an_error():
    """Target on an obstacle and the robot at the same point: no push direction.
    The exact version divides 0/0 into NaN (numpy semantics); the compiled one
    must do the same rather than raise ZeroDivisionError."""
    planner = FastPathPlanner(env=None)
    obstacles = [(np.array([0.0, 0.0]), np.array([1.0, 0.0]))]
    target = np.array([0.5, 0.0])
    with np.errstate(invalid="ignore", divide="ignore"):
        exact = planner._sanitize_target_exact(target, obstacles, target.copy(), clearance=0.3)
    fast = planner._sanitize_target_fast(target, obstacles, target.copy(), clearance=0.3)
    assert np.isnan(exact).all() and np.isnan(fast).all()


@pytest.mark.parametrize("seed", range(4))
def test_fast_clamp_to_obstacle_clearance_matches_exact(seed):
    rng = np.random.default_rng(300 + seed)
    planner = FastPathPlanner(env=None)
    for _ in range(500):
        obstacles = _scene(rng, n_robots=int(rng.integers(0, 12)))
        anchor = obstacles[int(rng.integers(len(obstacles)))]
        origin = anchor[0] + rng.normal(0.0, 0.3, 2)
        heading = rng.uniform(-math.pi, math.pi)
        args = (origin, np.array([math.cos(heading), math.sin(heading)]), float(rng.uniform(0.1, 1.0)), obstacles)
        clearance = float(rng.uniform(0.2, 0.5))
        fast = planner._clamp_to_obstacle_clearance_fast(*args, clearance=clearance)
        exact = planner._clamp_to_obstacle_clearance_exact(*args, clearance=clearance)
        assert abs(fast - exact) <= _TOL


def test_fast_math_utils_match_numpy():
    rng = np.random.default_rng(7)
    for _ in range(2000):
        p, s, e = (rng.uniform(-5, 5, 2) for _ in range(3))
        assert abs(math_utils._distance_fast(p, s) - math_utils._distance_numpy(p, s)) <= _TOL
        fast = math_utils._closest_point_on_segment_fast(p, s, e)
        exact = math_utils._closest_point_on_segment_numpy(p, s, e)
        assert _close(fast, exact)
    # Beyond either end both return the endpoint array itself, not a copy.
    s, e = np.array([0.0, 0.0]), np.array([1.0, 0.0])
    assert math_utils._closest_point_on_segment_fast(np.array([-1.0, 0.5]), s, e) is s
    assert math_utils._closest_point_on_segment_fast(np.array([2.0, 0.5]), s, e) is e


def test_fast_circular_mean_matches_numpy():
    rng = np.random.default_rng(11)
    for _ in range(2000):
        w, a, b = float(rng.random()), float(rng.uniform(-math.pi, math.pi)), float(rng.uniform(-math.pi, math.pi))
        fast = kalman._weighted_circular_mean_fast(w, a, b)
        exact = kalman._weighted_circular_mean_numpy(w, a, b)
        assert type(fast) is type(exact) is float
        # Equal up to rounding, modulo the +-pi wrap.
        assert abs(math.remainder(fast - exact, 2 * math.pi)) <= _TOL
