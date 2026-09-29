"""Every place `EXACT_MATH` (utama_core/config/settings.py) picks between the
original arithmetic and its fast counterpart, as
`(owner, attribute, exact implementation, fast implementation)`.

`test_exact_math_switch.py` checks the environment variable selects the
exact column in a fresh interpreter; `use_exact_math` lets a test that pins
bit-exact behaviour select it in-process whatever the environment says.
"""

from utama_core.data_processing.refiners.filters import kalman
from utama_core.global_utils import math_utils
from utama_core.motion_planning.src.fastpathplanning import planner
from utama_core.motion_planning.src.fastpathplanning.planner import FastPathPlanner

SWITCHES = [
    (math_utils, "distance", math_utils._distance_numpy, math_utils._distance_fast),
    (
        math_utils,
        "closest_point_on_segment",
        math_utils._closest_point_on_segment_numpy,
        math_utils._closest_point_on_segment_fast,
    ),
    # The planner module's own names for the two above, bound at its import.
    (planner, "distance", math_utils._distance_numpy, math_utils._distance_fast),
    (
        planner,
        "closest_point_on_segment",
        math_utils._closest_point_on_segment_numpy,
        math_utils._closest_point_on_segment_fast,
    ),
    (FastPathPlanner, "collides", FastPathPlanner._collides_exact, FastPathPlanner._collides_fast),
    (FastPathPlanner, "_find_subgoal", FastPathPlanner._find_subgoal_exact, FastPathPlanner._find_subgoal_fast),
    (
        FastPathPlanner,
        "_clamp_to_obstacle_clearance",
        FastPathPlanner._clamp_to_obstacle_clearance_exact,
        FastPathPlanner._clamp_to_obstacle_clearance_fast,
    ),
    (
        FastPathPlanner,
        "sanitize_target",
        FastPathPlanner._sanitize_target_exact,
        FastPathPlanner._sanitize_target_fast,
    ),
    (
        kalman,
        "_weighted_circular_mean",
        kalman._weighted_circular_mean_numpy,
        kalman._weighted_circular_mean_fast,
    ),
]


def use_exact_math(monkeypatch) -> None:
    """Select every exact implementation for the rest of the test."""
    for owner, name, exact, _fast in SWITCHES:
        monkeypatch.setattr(owner, name, exact)


def use_fast_math(monkeypatch) -> None:
    """Select every fast implementation for the rest of the test."""
    for owner, name, _exact, fast in SWITCHES:
        monkeypatch.setattr(owner, name, fast)
