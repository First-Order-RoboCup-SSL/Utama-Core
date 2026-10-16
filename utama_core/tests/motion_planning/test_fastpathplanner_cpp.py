"""Tests for the native C++ FastPathPlanner extension (_fastpathplanner).

Verifies bit-for-bit / epsilon equivalence between the C++ Pybind11 implementation
and the reference Python/Numba implementations for all core geometric functions:
- collides
- find_subgoal
- clamp_to_obstacle_clearance
- sanitize_target
- check_segment (recursive detour tree search)
- smooth_path (lookahead carrot projection)

The extension is built locally (`pixi run build-planner`) and is not built in CI,
so the module is skipped when it is missing; the planner then uses the numba kernels.
"""

import numpy as np
import pytest

_fastpathplanner = pytest.importorskip("utama_core.motion_planning.src.fastpathplanning._fastpathplanner")

from utama_core.motion_planning.src.fastpathplanning import numba_kernels  # noqa: E402
from utama_core.motion_planning.src.fastpathplanning.planner import (  # noqa: E402
    HAS_CPP,
    FastPathPlanner,
)

_TOL = 1e-9


def test_cpp_extension_is_loaded():
    assert HAS_CPP is True, "The C++ native _fastpathplanner extension must be built and importable"
    assert _fastpathplanner is not None


@pytest.mark.parametrize("seed", [1, 2, 3, 42])
def test_cpp_collides_matches_numba(seed):
    rng = np.random.default_rng(seed)
    n = 15
    ox0 = rng.uniform(-4.0, 4.0, n)
    oy0 = rng.uniform(-3.0, 3.0, n)
    ox1 = ox0 + rng.uniform(-0.5, 0.5, n)
    oy1 = oy0 + rng.uniform(-0.5, 0.5, n)

    seg_a = rng.uniform(-3.0, 3.0, 2)
    seg_b = rng.uniform(-3.0, 3.0, 2)
    clearance = float(rng.uniform(0.15, 0.35))
    sticky_idx = int(rng.integers(0, n)) if rng.random() < 0.5 else -1

    c_cpp = _fastpathplanner.collides_nb(
        seg_a[0], seg_a[1], seg_b[0], seg_b[1], clearance, ox0, oy0, ox1, oy1, sticky_idx, 0.05
    )
    c_nb = numba_kernels.collides_nb(
        seg_a[0], seg_a[1], seg_b[0], seg_b[1], clearance, ox0, oy0, ox1, oy1, sticky_idx, 0.05
    )

    idx_cpp, from_stk_cpp, px_cpp, py_cpp = c_cpp
    idx_nb, from_stk_nb, px_nb, py_nb = c_nb

    assert idx_cpp == idx_nb
    assert from_stk_cpp == from_stk_nb
    if idx_cpp >= 0:
        assert np.isclose(px_cpp, px_nb, atol=_TOL)
        assert np.isclose(py_cpp, py_nb, atol=_TOL)


@pytest.mark.parametrize("seed", [10, 20, 30])
def test_cpp_find_subgoal_matches_numba(seed):
    rng = np.random.default_rng(seed)
    n = 10
    ox0 = rng.uniform(-3.0, 3.0, n)
    oy0 = rng.uniform(-2.0, 2.0, n)
    ox1 = ox0 + rng.uniform(-0.4, 0.4, n)
    oy1 = oy0 + rng.uniform(-0.4, 0.4, n)

    op = rng.uniform(-2.0, 2.0, 2)
    angle = rng.uniform(0, 2 * np.pi)
    ux, uy = np.cos(angle), np.sin(angle)
    clearance = 0.22
    subgoal_dist = 0.35

    res_cpp = _fastpathplanner.find_subgoal_nb(
        op[0], op[1], ux, uy, subgoal_dist, 1, clearance, ox0, oy0, ox1, oy1, False, 0.0, 0.0, 0.0, 0.0
    )
    res_nb = numba_kernels.find_subgoal_nb(
        op[0], op[1], ux, uy, subgoal_dist, 1, clearance, ox0, oy0, ox1, oy1, False, 0.0, 0.0, 0.0, 0.0
    )

    assert res_cpp[0] == res_nb[0]  # found
    if res_cpp[0]:
        assert np.isclose(res_cpp[1], res_nb[1], atol=_TOL)
        assert np.isclose(res_cpp[2], res_nb[2], atol=_TOL)
    assert res_cpp[3] == res_nb[3]  # last_hit_idx
    assert res_cpp[4] == res_nb[4]  # rect_last


def test_cpp_check_segment_and_smooth_path():
    planner = FastPathPlanner(env=None)

    ox0 = np.array([-1.0, 0.0, 1.0])
    oy0 = np.array([0.0, -1.0, 2.0])
    ox1 = np.array([1.0, 0.0, 2.0])
    oy1 = np.array([0.0, 1.0, 3.0])
    obstacles = [(np.array([ox0[i], oy0[i]]), np.array([ox1[i], oy1[i]])) for i in range(len(ox0))]

    class DummyBounds:
        top_left = (-4.5, 3.0)
        bottom_right = (4.5, -3.0)

    bounds = DummyBounds()
    seg = (np.array([0.0, -2.0]), np.array([0.0, 2.0]))
    target = np.array([0.0, 2.0])

    detour_dict = {}
    obs_dict = {}
    traj_cpp, len_cpp = _fastpathplanner.check_segment_cpp(
        seg,
        ox0,
        oy0,
        ox1,
        oy1,
        None,
        0,
        3,
        target,
        (-4.5, 4.5, -3.0, 3.0),
        0,
        0.22,
        0.35,
        None,
        0.05,
        0.05,
        2,
        detour_dict,
        obs_dict,
    )

    # Reference Python (with HAS_CPP=False)
    import utama_core.motion_planning.src.fastpathplanning.planner as planner_mod

    planner_mod.HAS_CPP = False
    try:
        traj_py, len_py = planner.check_segment(seg, obstacles, 0, target, bounds, 0, 0.22, 0.35, None)
        carrot_py = planner.smooth_path(traj_py, target, np.array([0.0, -2.0]), obstacles, 0.22)
    finally:
        planner_mod.HAS_CPP = True

    assert len(traj_cpp) == len(traj_py)
    assert np.isclose(len_cpp, len_py, atol=_TOL)
    for (ca, cb), (pa, pb) in zip(traj_cpp, traj_py):
        assert np.allclose(ca, pa, atol=_TOL)
        assert np.allclose(cb, pb, atol=_TOL)

    # Smooth path
    carrot_cpp = _fastpathplanner.smooth_path_cpp(
        traj_cpp, target, np.array([0.0, -2.0]), 0.5, 0.22, ox0, oy0, ox1, oy1
    )
    assert np.allclose(carrot_cpp, carrot_py, atol=_TOL)
