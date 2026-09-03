"""The same representative black-box scenarios for every motion planner."""

import math
import random

import pytest

from utama_core.config.field_params import STANDARD_FIELD_DIMS
from utama_core.config.referee_constants import OPPONENT_DEFENSE_AREA_KEEP_DISTANCE
from utama_core.tests.motion_planning.standardized.harness import (
    Disturbance,
    MotionScenario,
    run_scenario,
)

CONTROL_SCHEMES = ("fpp", "dwa", "trajsample")

# Real field geometry (metres) for the boundary/defense-area scenarios below,
# rather than made-up coordinates -- see STANDARD_FIELD_DIMS, the default
# `StrategyRunner` uses.
_HALF_LENGTH = STANDARD_FIELD_DIMS.full_field_half_length  # 4.5
_HALF_WIDTH = STANDARD_FIELD_DIMS.full_field_half_width  # 3.0
_DEFENSE_DEPTH = 2 * STANDARD_FIELD_DIMS.half_defense_area_depth  # 1.0

SCENARIOS = (
    MotionScenario(
        name="open_field",
        friendly_starts=((-1.5, -1.0),),
        friendly_targets=((1.5, -1.0),),
        timeout=6.0,
    ),
    MotionScenario(
        name="stationary_blocker",
        friendly_starts=((-2.0, 0.0),),
        friendly_targets=((2.0, 0.0),),
        enemy_starts=((0.0, 0.0),),
        timeout=8.0,
    ),
    MotionScenario(
        name="perpendicular_crossing",
        friendly_starts=((-1.5, 0.0), (0.0, 1.5)),
        friendly_targets=((1.5, 0.0), (0.0, -1.5)),
        endpoint_tolerance=0.2,
        timeout=8.0,
    ),
    MotionScenario(
        name="narrow_passage",
        # Two static obstacles 0.42m apart (centre-to-centre) leave a gap of
        # 0.42 - 2*ROBOT_RADIUS = 0.24m -- wide enough for one robot (0.18m
        # diameter) to fit through with a small margin, tight enough that a
        # planner must route precisely rather than swing wide.
        friendly_starts=((-2.0, 0.0),),
        friendly_targets=((2.0, 0.0),),
        enemy_starts=((0.0, 0.21), (0.0, -0.21)),
        endpoint_tolerance=0.15,
        timeout=10.0,
    ),
    MotionScenario(
        name="head_on_swap",
        # Two robots drive directly at each other's start position along the
        # same line -- distinct from `mirror_swap`'s dense 6v6 case, this
        # isolates the single-pair head-on symmetry-breaking behaviour.
        friendly_starts=((-1.5, 0.0),),
        friendly_targets=((1.5, 0.0),),
        enemy_starts=((1.5, 0.0),),
        enemy_targets=((-1.5, 0.0),),
        endpoint_tolerance=0.2,
        timeout=10.0,
    ),
    MotionScenario(
        name="field_boundary_corner",
        # Target sits just inside the field corner (within one endpoint
        # tolerance of both boundary lines), exercising boundary
        # clamping/`is_point_in_field`-style logic near a real corner rather
        # than mid-field.
        friendly_starts=((0.0, 0.0),),
        friendly_targets=((_HALF_LENGTH - 0.2, _HALF_WIDTH - 0.2),),
        endpoint_tolerance=0.15,
        timeout=10.0,
    ),
    MotionScenario(
        name="defense_area_boundary",
        # Target sits just outside the planner-enforced keep-distance around
        # the (enemy, i.e. right-side) defense area -- OPPONENT_DEFENSE_AREA_
        # KEEP_DISTANCE beyond the raw geometric boundary, since planners
        # clamp targets to that margin and can never close on a point placed
        # only outside the raw defense-area line itself. Uses
        # STANDARD_FIELD_DIMS' actual defense-area depth rather than an
        # arbitrary point.
        friendly_starts=((0.0, 0.0),),
        friendly_targets=((_HALF_LENGTH - _DEFENSE_DEPTH - OPPONENT_DEFENSE_AREA_KEEP_DISTANCE - 0.1, 0.0),),
        # Looser than the other fixed-target scenarios: FPP's own
        # defense-area clamp settles its closest reachable point ~0.2m short
        # of this target (verified directly), so a tighter tolerance would
        # fail on FPP's legitimate clamping behaviour, not a real defect.
        endpoint_tolerance=0.25,
        timeout=10.0,
    ),
    MotionScenario(
        name="interception",
        # The friendly robot must intercept an enemy-controlled point moving
        # at a slow constant velocity across the field, rather than chasing a
        # fixed point -- the fixed `enemy_targets` entry here is only the
        # bookkeeping endpoint at scenario timeout, the actual live target is
        # `enemy_target_trajectories`' linear sweep. Speed/duration chosen so
        # the sweep stays in-bounds for the full timeout (a target that runs
        # off-field mid-scenario is unreachable by construction, not a
        # planner defect) and so the friendly robot's ~2.7m closing distance
        # is coverable at RSIM_PARAMS.MAX_VEL=2 m/s well before the sweep ends.
        friendly_starts=((-2.0, -1.5),),
        friendly_targets=((0.0, 0.0),),
        enemy_starts=((-1.0, 1.0),),
        enemy_targets=((1.0, -1.0),),
        enemy_target_trajectories={0: lambda t: (-1.0 + 0.25 * t, 1.0 - 0.25 * t)},
        endpoint_tolerance=0.2,
        timeout=8.0,
    ),
    MotionScenario(
        name="sudden_obstacle",
        # The corridor is clear for the first 2s (long enough for a planner
        # to have already committed to a direct path), then an obstacle
        # teleports into the middle of that path -- tests replan
        # responsiveness rather than up-front avoidance.
        friendly_starts=((-2.0, 0.0),),
        friendly_targets=((2.0, 0.0),),
        enemy_starts=((3.5, 3.0 - 0.01),),  # parked off to the side, out of the way, until the disturbance fires
        disturbances=(Disturbance(at_time=2.0, team="enemy", robot_id=0, position=(0.0, 0.0)),),
        endpoint_tolerance=0.15,
        timeout=12.0,
    ),
    MotionScenario(
        name="disturbance_recovery",
        # The friendly robot is knocked off its planned path partway through
        # (simulating a collision/foul reset) and must recognise the
        # now-invalid stale trajectory and replan to still reach the target,
        # rather than continuing to execute the pre-disturbance plan.
        friendly_starts=((-2.0, 0.0),),
        friendly_targets=((2.0, 0.0),),
        disturbances=(Disturbance(at_time=1.5, team="friendly", robot_id=0, position=(-1.0, 1.5)),),
        endpoint_tolerance=0.15,
        timeout=12.0,
    ),
    MotionScenario(
        name="start_inside_obstacle",
        # The friendly robot begins with its centre only 0.10m from a
        # stationary enemy -- genuinely overlapping the enemy's physical
        # footprint (2*ROBOT_RADIUS = 0.18m apart is normal contact; 0.10m
        # is 0.08m *inside* that). This is the exact condition 2e53f3e
        # ("Fix trajsample planner deadlock when a robot starts inside an
        # obstacle") fixed: the collision-avoidance margin at a stationary
        # robot's own start position (speed = 0) is 0, so only a TRUE
        # overlap -- not merely a close approach -- makes `d < margin` fire
        # at the very first sample, rejecting every candidate direction
        # regardless of where it points and pinning the robot indefinitely
        # (the live repro this fix names used the ball, radius 0.0215m, for
        # exactly this reason: 0.088m separation was inside the 0.09+0.0215
        # = 0.1115m combined radius). The standardized harness only has
        # robot-sized obstacles, so this scenario uses a scenario-local,
        # tighter `collision_distance` (5cm) purely to avoid flagging the
        # deliberately-scripted starting overlap as a scenario failure in
        # itself; reaching the 2m-distant target still requires the robot to
        # genuinely separate well past the real 0.18m contact radius, so an
        # actual escape is still required, and a real new collision picked
        # up while doing so is still caught. Planner-agnostic: any planner
        # must accept a start point already inside an obstacle and still
        # make progress toward a distant target. Note: this single-static-
        # obstacle, straight-line-escape geometry reliably passes even with
        # 2e53f3e's fix reverted (verified directly) -- the live deadlock
        # was a *braking*-driven near-zero-velocity crawl from an emergent
        # multi-tick, multi-obstacle interaction (a teammate's committed
        # trajectory, per the fix's own roadmap entry), not a hard veto this
        # simpler shape reproduces standalone. The exact boundary condition
        # (`first_collision_numba`'s escape grace) is pinned deterministically
        # by a dedicated unit test instead (see trajsampling_correctness_test.py);
        # this scenario keeps the realistic "start already overlapping" case
        # in the black-box regression suite going forward.
        friendly_starts=((0.10, 0.0),),
        friendly_targets=((2.0, 0.0),),
        enemy_starts=((0.0, 0.0),),
        endpoint_tolerance=0.15,
        collision_distance=0.05,
        timeout=10.0,
    ),
)

# Seed for `jittering_target`'s deterministic ~1mm jitter -- fixed so the
# scenario is exactly repeatable, matching `TrajectorySamplingPlanner`'s own
# `random.Random(0)` convention for its intermediate-target sampling.
_JITTER_SEED = 0
# Matches DirectFreeOursStep's kicker-approach point being recomputed from
# the ball's live position every tick: real physics-sim position jitter is on
# the order of 1e-4m, this uses a full 1mm (10x that) as a deliberately
# generous stand-in that stays far below any real target change (which jumps
# by centimetres to metres -- see planner.py's _TRAJECTORY_TARGET_TOLERANCE
# comment) while still exercising the same "recomputed every tick" pattern.
_JITTER_RADIUS_M = 0.001


def _make_jitter_trajectory(center: tuple[float, float], seed: int):
    """A deterministic, per-call ~1mm-jittered target around `center`,
    the same shape `DirectFreeOursStep`'s kicker-approach point takes when
    it's recomputed every tick from the ball's live (noisy) position --
    see 5183ed1 ("Fix trajsample planner target-jitter stall on DIRECT_FREE
    restarts"). A fresh `random.Random(seed)` is captured per trajectory
    (not shared module state) so repeated scenario runs are independently
    reproducible regardless of call order.
    """
    rng = random.Random(seed)

    def _target(_elapsed: float) -> tuple[float, float]:
        angle = rng.uniform(0.0, 2.0 * math.pi)
        radius = rng.uniform(0.0, _JITTER_RADIUS_M)
        return (center[0] + radius * math.cos(angle), center[1] + radius * math.sin(angle))

    return _target


_JITTER_FIXED_TARGET = (2.0, 0.0)

JITTER_COMPARISON_SCENARIOS = (
    MotionScenario(
        name="jittering_target_fixed_baseline",
        # Same geometry as `jittering_target` below but with a plain fixed
        # target -- the reference completion time `jittering_target`'s
        # upper-bound assertion is measured against.
        friendly_starts=((-2.0, 0.0),),
        friendly_targets=(_JITTER_FIXED_TARGET,),
        endpoint_tolerance=0.15,
        timeout=12.0,
    ),
    MotionScenario(
        name="jittering_target",
        # The target is recomputed every tick with ~1mm deterministic random
        # jitter around the same fixed point as the baseline above -- the
        # same pattern `DirectFreeOursStep`'s kicker-approach point exhibits
        # when it's re-derived from the ball's live, noisy position every
        # tick (5183ed1). A planner that replans from t=0 on every
        # sub-millimetre "change" crawls instead of executing its committed
        # trajectory; this scenario's own test asserts completion time stays
        # close to the fixed-target baseline rather than checking a bare
        # timeout, so it fails on *slowdown*, not just on total stall.
        friendly_starts=((-2.0, 0.0),),
        friendly_targets=(_JITTER_FIXED_TARGET,),
        friendly_target_trajectories={0: _make_jitter_trajectory(_JITTER_FIXED_TARGET, _JITTER_SEED)},
        endpoint_tolerance=0.15,
        timeout=12.0,
    ),
)


# These are black-box failures observed by this suite, not relaxed contracts.
# Strict xfails keep CI useful while making a planner improvement visible as an
# XPASS that requires removing the corresponding entry.
KNOWN_LIMITATIONS = {
    ("stationary_blocker", "dwa"): "DWA reaches physical contact with the stationary robot",
    ("perpendicular_crossing", "dwa"): "DWA reaches physical contact during the crossing",
    ("narrow_passage", "dwa"): "DWA reaches physical contact threading the gap (short lookahead, no committed path)",
}


def _cases():
    for scenario in SCENARIOS:
        for control_scheme in CONTROL_SCHEMES:
            limitation = KNOWN_LIMITATIONS.get((scenario.name, control_scheme))
            marks = pytest.mark.xfail(strict=True, reason=limitation) if limitation else ()
            yield pytest.param(scenario, control_scheme, id=f"{scenario.name}-{control_scheme}", marks=marks)


@pytest.mark.parametrize(("scenario", "control_scheme"), tuple(_cases()))
def test_standardized_motion_scenario(
    headless: bool,
    scenario: MotionScenario,
    control_scheme: str,
) -> None:
    passed, metrics = run_scenario(scenario, control_scheme, headless=headless)
    context = f"{scenario.name}/{control_scheme}: {metrics.summary()}"

    assert passed, context
    assert metrics.collision_pair is None, f"{context}, collision={metrics.collision_pair}"
    # `final_errors` is measured against each scenario's *fixed* `*_targets`
    # entry. For a robot with a moving-target override (interception
    # scenarios) that fixed point is only a timeout-bookkeeping reference,
    # not the live goal `passed` was judged against above, so it's excluded
    # here rather than asserted against.
    moving_target_keys = {("friendly", robot_id) for robot_id in scenario.friendly_target_trajectories} | {
        ("enemy", robot_id) for robot_id in scenario.enemy_target_trajectories
    }
    stationary_final_errors = {
        key: error for key, error in metrics.final_errors.items() if key not in moving_target_keys
    }
    assert all(error <= scenario.endpoint_tolerance for error in stationary_final_errors.values()), context
    assert metrics.samples > 0, context
    assert metrics.peak_speed > 0.0, context


# How much slower `jittering_target` may be than `jittering_target_fixed_baseline`
# before it counts as a real slowdown rather than ordinary run-to-run noise.
# Deliberately a multiplicative bound on measured completion time (not a
# magic absolute number): the actual bug this guards against (5183ed1) was
# not a modest slowdown, it was a ~0.02 m/s crawl -- roughly two orders of
# magnitude slower than a normal ~6s traversal of this distance -- so 1.5x
# comfortably separates "genuinely stalling on jitter" from "a bit slower".
_JITTER_SLOWDOWN_TOLERANCE = 1.5


@pytest.mark.parametrize("control_scheme", CONTROL_SCHEMES)
def test_jittering_target_completes_within_fixed_target_time_bound(headless: bool, control_scheme: str) -> None:
    """A ~1mm-jittered target (5183ed1's DIRECT_FREE-restart pattern) must
    not meaningfully slow down completion relative to the same geometry with
    a plain fixed target -- see `JITTER_COMPARISON_SCENARIOS` above.
    """
    fixed_scenario, jitter_scenario = JITTER_COMPARISON_SCENARIOS

    fixed_passed, fixed_metrics = run_scenario(fixed_scenario, control_scheme, headless=headless)
    fixed_context = f"{fixed_scenario.name}/{control_scheme}: {fixed_metrics.summary()}"
    assert fixed_passed, fixed_context
    assert fixed_metrics.collision_pair is None, fixed_context

    jitter_passed, jitter_metrics = run_scenario(jitter_scenario, control_scheme, headless=headless)
    jitter_context = f"{jitter_scenario.name}/{control_scheme}: {jitter_metrics.summary()}"
    assert jitter_passed, jitter_context
    assert jitter_metrics.collision_pair is None, jitter_context

    bound = fixed_metrics.elapsed * _JITTER_SLOWDOWN_TOLERANCE
    assert jitter_metrics.elapsed <= bound, (
        f"{jitter_scenario.name}/{control_scheme} took {jitter_metrics.elapsed:.2f}s, "
        f"more than {_JITTER_SLOWDOWN_TOLERANCE}x the {fixed_scenario.name} baseline "
        f"({fixed_metrics.elapsed:.2f}s, bound {bound:.2f}s)"
    )
