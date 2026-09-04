"""Internal correctness tests for trajectory-sampling motion primitives."""

import math

import numpy as np
import pytest

from utama_core.config.field_params import STANDARD_FIELD_DIMS
from utama_core.entities.data.vector import Vector2D, Vector3D
from utama_core.entities.game import Ball, Field, Game, GameFrame, GameHistory, Robot
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.motion_planning.src.trajsampling import collision_numba as collision
from utama_core.motion_planning.src.trajsampling.bang_bang import (
    BangBang1D,
    Trajectory2D,
)
from utama_core.motion_planning.src.trajsampling.obstacles import (
    ConstantVelocityObstacle,
    EnemyRobotObstacle,
    OwnRobotObstacle,
    StaticSegmentObstacle,
)
from utama_core.motion_planning.src.trajsampling.planner import (
    PlanResult,
    TrajectorySamplingPlanner,
    _bangbang_to_row,
    _CommittedTrajectoryObstacle,
    _flatten_obstacles,
    _flatten_query_trajectory,
    _priority_blocking_enabled,
)


@pytest.mark.parametrize("v_max,a_max", [(0.0, 1.0), (1.0, 0.0), (-1.0, 1.0)])
def test_bang_bang_rejects_non_positive_limits(v_max, a_max):
    with pytest.raises(ValueError, match="must be positive"):
        BangBang1D.compute(0.0, 0.0, 1.0, v_max, a_max)


@pytest.mark.parametrize(
    "p0,v0,p1,v_max,a_max",
    [
        (0.0, 0.0, 1.0, 3.0, 2.0),  # triangular profile
        (-2.0, 0.0, 5.0, 2.0, 1.0),  # trapezoid with cruise
        (2.0, -0.4, -1.0, 2.5, 1.5),  # negative travel direction
        (0.0, 1.0, 5.0, 2.0, 1.0),  # non-zero velocity toward target
    ],
)
def test_bang_bang_endpoint_and_kinematic_invariants(p0, v0, p1, v_max, a_max):
    trajectory = BangBang1D.compute(p0, v0, p1, v_max, a_max)

    assert 0.0 <= trajectory.t1 <= trajectory.t2 <= trajectory.t_end
    assert trajectory.state_at(-1.0) == (p0, v0)
    assert trajectory.state_at(trajectory.t_end) == (p1, 0.0)
    assert trajectory.state_at(trajectory.t_end + 1.0) == (p1, 0.0)

    times = np.linspace(0.0, trajectory.t_end, 1001)
    states = np.array([trajectory.state_at(float(t)) for t in times])
    positions = states[:, 0]
    velocities = states[:, 1]
    time_steps = np.diff(times)

    assert np.all(np.isfinite(states))
    assert np.max(np.abs(velocities)) <= v_max + 1e-10
    assert np.all(np.abs(np.diff(velocities)) <= a_max * time_steps + 1e-10)
    # Integrating a linearly-changing velocity over each sample interval
    # should recover the sampled positions for every constant-acceleration
    # phase. One grid interval may straddle a phase boundary, where the
    # trapezoidal approximation has an O(a*dt^2) error.
    integrated_steps = (velocities[:-1] + velocities[1:]) * 0.5 * time_steps
    integration_tolerance = a_max * float(np.max(time_steps)) ** 2
    assert np.allclose(np.diff(positions), integrated_steps, atol=integration_tolerance)


@pytest.mark.parametrize(
    "p0,v0,p1,v_max,a_max",
    [
        (0.0, -1.0, 2.0, 3.0, 2.0),
        (2.0, 0.4, -1.0, 2.5, 1.5),
    ],
)
def test_bang_bang_is_position_continuous_at_endpoint_after_opposing_velocity(p0, v0, p1, v_max, a_max):
    trajectory = BangBang1D.compute(p0, v0, p1, v_max, a_max)
    position_immediately_before_end, _ = trajectory.state_at(np.nextafter(trajectory.t_end, 0.0))

    assert position_immediately_before_end == pytest.approx(p1, abs=1e-9)


_BANG_BANG_FIX_BACKED_OUT = (
    "BangBang1D fix (b26a550) backed out: with physically correct trajectories the "
    "trajsample planner deadlocks mutually blocked robots at every kickoff. Re-apply "
    "together with a planner blocked-start fix; see docs/roadmap.md."
)


@pytest.mark.xfail(strict=True, reason=_BANG_BANG_FIX_BACKED_OUT)
def test_bang_bang_required_overshoot_produces_continuous_trajectory():
    """Regression for the required-overshoot defect: v0 points toward the
    target but its braking distance (v0^2/(2*a_max)) exceeds the remaining
    gap, so the robot cannot land on p1 without first passing it. Pre-fix,
    `compute` produced a negative `t1` and `state_at` jumped discontinuously
    in both position and velocity immediately after t=0 -- confirmed via the
    exact values below: pre-fix, `t.t1 == -0.5297` and `state_at(0.01)` gave
    `(2.904, -3.174)` against `state_at(0) == (2.611, -4.413)`, an implied
    acceleration far beyond `a_max`. The fix re-expresses this case as
    decelerate-past-target-to-v=0, then a fresh bang-bang straight back."""
    p0, v0, p1, v_max, a_max = (
        2.6108637010973705,
        -4.412899566309155,
        -1.4458847258394982,
        4.803012201215744,
        1.1582054489697327,
    )
    d = p1 - p0
    sign = 1.0 if d >= 0 else -1.0
    braking_distance = (v0 * sign) ** 2 / (2 * a_max)
    assert braking_distance > abs(d)  # sanity: this really is the required-overshoot region

    trajectory = BangBang1D.compute(p0, v0, p1, v_max, a_max)
    assert trajectory.t1 >= 0.0  # pre-fix: t1 == -0.5297...

    pos0, vel0 = trajectory.state_at(0.0)
    assert pos0 == pytest.approx(p0)
    assert vel0 == pytest.approx(v0)

    # Continuity at t=0+: the very next instant must be reachable from
    # (p0, v0) under |accel| <= a_max, not an unbounded jump.
    dt = 1e-4
    pos_next, vel_next = trajectory.state_at(dt)
    implied_accel = abs(vel_next - v0) / dt
    assert implied_accel <= a_max + 1e-6
    # Position changes smoothly too (v0 is large, so a generous but finite
    # bound -- not the ~30cm jump the pre-fix discontinuity produced).
    assert abs(pos_next - pos0) <= (abs(v0) + a_max * dt) * dt + 1e-9

    # |accel| <= a_max holds throughout the whole trajectory, including
    # across the direction-reversal at the overshoot point.
    times = np.linspace(0.0, trajectory.t_end, 2001)
    velocities = np.array([trajectory.state_at(float(t))[1] for t in times])
    accel = np.abs(np.diff(velocities)) / np.diff(times)
    assert np.max(accel) <= a_max * 1.01 + 1e-6

    end_pos, end_vel = trajectory.state_at(trajectory.t_end)
    assert end_pos == pytest.approx(p1, abs=1e-6)
    assert end_vel == pytest.approx(0.0, abs=1e-6)


@pytest.mark.xfail(strict=True, reason=_BANG_BANG_FIX_BACKED_OUT)
def test_bang_bang_same_direction_overspeed_decelerates_at_a_max():
    """Regression for the same-direction-overspeed defect: v0 already points
    toward p1 and exceeds v_max. Pre-fix, `compute`'s carry-over reset
    silently planned the phase schedule as if starting at v_max while
    `state_at(0)` still (correctly) reported the real, higher v0 -- an
    effectively instantaneous velocity change immediately after t=0 (t1 was
    exactly 0.0 for this repro). The fix decelerates from v0 down to v_max
    at exactly a_max first."""
    p0, v0, p1, v_max, a_max = 0.0, 5.0, 1.0, 2.0, 2.0
    trajectory = BangBang1D.compute(p0, v0, p1, v_max, a_max)
    assert trajectory.t1 > 0.0  # pre-fix: t1 == 0.0

    pos0, vel0 = trajectory.state_at(0.0)
    assert pos0 == pytest.approx(p0)
    assert vel0 == pytest.approx(v0)  # velocity continuous at t=0: the real v0, not v_max

    # Immediately after t=0 the robot must still be decelerating at exactly
    # a_max (not an implied ~3,000,000 m/s^2 as pre-fix), toward p1's
    # direction (sign=+1 here).
    dt = 1e-4
    _pos_next, vel_next = trajectory.state_at(dt)
    accel = (vel_next - vel0) / dt
    assert accel == pytest.approx(-a_max, abs=1e-2)

    end_pos, end_vel = trajectory.state_at(trajectory.t_end)
    assert end_pos == pytest.approx(p1, abs=1e-6)
    assert end_vel == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize(
    "p0,v0,p1",
    [
        ((-1.0, 2.0), (0.0, 0.0), (2.0, -2.0)),
        ((1.5, -0.5), (-0.4, 0.7), (-2.0, 1.5)),
        ((-2.0, -1.0), (0.8, 0.4), (2.0, 1.0)),
    ],
)
def test_trajectory_2d_endpoint_straightness_and_kinematic_limits(p0, v0, p1):
    v_max = 3.0
    a_max = 2.0
    trajectory = Trajectory2D.compute(p0, v0, p1, v_max, a_max)

    end_position, end_velocity = trajectory.state_at(trajectory.duration)
    assert end_position == pytest.approx(p1)
    assert end_velocity == pytest.approx((0.0, 0.0))
    position_immediately_before_end, _ = trajectory.state_at(np.nextafter(trajectory.duration, 0.0))
    assert position_immediately_before_end == pytest.approx(p1, abs=1e-9)

    times = np.linspace(0.0, trajectory.duration, 1001)
    states = [trajectory.state_at(float(t)) for t in times]
    positions = np.array([position for position, _ in states])
    velocities = np.array([velocity for _, velocity in states])
    displacement = np.asarray(p1) - np.asarray(p0)

    # Every position remains on the p0->p1 line and velocity has no
    # transverse component, which is Trajectory2D's defining invariant.
    relative_positions = positions - np.asarray(p0)
    position_cross_products = displacement[0] * relative_positions[:, 1] - displacement[1] * relative_positions[:, 0]
    velocity_cross_products = displacement[0] * velocities[:, 1] - displacement[1] * velocities[:, 0]
    assert np.allclose(position_cross_products, 0.0, atol=1e-10)
    assert np.allclose(velocity_cross_products, 0.0, atol=1e-10)
    assert np.max(np.linalg.norm(velocities, axis=1)) <= v_max + 1e-10

    dt = np.diff(times)
    delta_velocity = np.linalg.norm(np.diff(velocities, axis=0), axis=1)
    assert np.all(delta_velocity <= a_max * dt + 1e-10)


@pytest.mark.parametrize(
    "parameters",
    [
        (0.0, 0.0, 1.0, 3.0, 2.0),
        (2.0, 0.4, -1.0, 2.5, 1.5),
        (0.0, -1.0, 2.0, 3.0, 2.0),
    ],
)
def test_numba_bang_bang_state_matches_python(parameters):
    trajectory = BangBang1D.compute(*parameters)
    row = _bangbang_to_row(trajectory)

    for t in np.linspace(-0.1, trajectory.t_end + 0.1, 101):
        assert collision._bangbang_state_at(row, float(t)) == pytest.approx(trajectory.state_at(float(t)), abs=1e-12)


def test_numba_query_trajectory_state_matches_python_2d_trajectory():
    trajectory = Trajectory2D.compute((-1.0, 0.5), (0.4, -0.2), (2.0, -1.0), 3.0, 2.0)
    leg1, switch_t, leg2, _ = _flatten_query_trajectory(trajectory)

    for t in np.linspace(0.0, trajectory.duration, 101):
        expected_position, expected_velocity = trajectory.state_at(float(t))
        actual = collision._query_trajectory_state(
            leg1[0],
            leg1[1],
            leg1[2],
            leg1[3],
            leg1[4],
            switch_t,
            leg2[0],
            leg2[1],
            leg2[2],
            leg2[3],
            leg2[4],
            float(t),
        )
        assert actual == pytest.approx((*expected_position, *expected_velocity), abs=1e-12)


@pytest.mark.parametrize("t,point", [(0.0, (0.0, 0.0)), (0.3, (1.2, -0.4)), (0.8, (-0.5, 2.0))])
def test_numba_obstacle_distance_kernels_match_python(t, point):
    static = StaticSegmentObstacle(a=(-1.0, 0.5), b=(2.0, 0.5), radius=0.12)
    moving = ConstantVelocityObstacle(p0=(0.3, -0.2), v=(0.7, -0.4), radius=0.09)
    enemy = EnemyRobotObstacle(
        p0=(-0.4, 0.2),
        speed=0.8,
        direction=(0.6, 0.8),
        v_max=3.0,
        a_max=2.0,
        radius=0.09,
    )
    static_rows, _, moving_rows, enemy_rows = _flatten_obstacles([static, moving, enemy])

    assert collision._static_distance_at(static_rows[0], *point) == pytest.approx(
        static.distance_at(t, point), abs=1e-12
    )
    assert collision._cv_distance_at(moving_rows[0], t, *point) == pytest.approx(
        moving.distance_at(t, point), abs=1e-12
    )
    assert collision._enemy_distance_at(enemy_rows[0], t, *point) == pytest.approx(
        enemy.distance_at(t, point), abs=1e-12
    )


@pytest.mark.parametrize("t,point", [(0.0, (0.0, 0.0)), (0.4, (0.3, 0.8)), (1.2, (1.5, -0.2))])
def test_numba_committed_trajectory_obstacle_matches_python(t, point):
    trajectory = Trajectory2D.compute((-1.0, 0.0), (0.3, 0.0), (2.0, 0.5), 3.0, 2.0)
    obstacle = OwnRobotObstacle(trajectory=trajectory, radius=0.09)
    _, trajectory_rows, _, _ = _flatten_obstacles([obstacle])

    assert collision._traj_obstacle_distance_at(trajectory_rows[0], t, *point) == pytest.approx(
        obstacle.distance_at(t, point), abs=1e-12
    )


def test_trajectory_2d_initial_velocity_is_projection_onto_path():
    trajectory = Trajectory2D.compute((0.0, 0.0), (1.0, 2.0), (3.0, 0.0), 3.0, 2.0)

    _, velocity = trajectory.state_at(0.0)
    assert velocity == pytest.approx((1.0, 0.0))
    assert math.hypot(*velocity) <= trajectory.along.v_max


def test_trajectory_2d_stationary_at_target_has_zero_duration():
    trajectory = Trajectory2D.compute((1.0, -2.0), (0.0, 0.0), (1.0, -2.0), 3.0, 2.0)

    assert trajectory.duration == 0.0
    assert trajectory.state_at(0.0) == ((1.0, -2.0), (0.0, 0.0))


def test_trajectory_2d_degenerate_target_brakes_using_v0_direction_not_a_fixed_axis():
    """Pins fe6a07e ("Fix Trajectory2D degenerate zero-distance fallback").

    `move()`/`turn_on_spot()` call `Trajectory2D.compute(p0, v0, p1=p0, ...)`
    every tick while orienting in place (e.g. `GiveAndGoTactic`'s pre-kick aim
    step). The pre-fix `dist < 1e-9` branch projected `v0` onto a fixed
    arbitrary axis `(1.0, 0.0)` instead of `v0`'s own direction, so a robot
    moving purely laterally (v0 perpendicular to that fixed axis) had its
    entire real speed silently dropped: `duration` came out as 0.0 and the
    commanded velocity was exactly zero forever, never actually braking.
    """
    p0 = (1.0, 2.0)
    v0 = (0.0, 0.8)  # purely lateral -- fully perpendicular to the old (1,0) axis
    trajectory = Trajectory2D.compute(p0, v0, p0, v_max=3.0, a_max=2.0)

    assert trajectory.duration > 0.0
    _, commanded_velocity = trajectory.state_at(0.0)
    assert math.hypot(*commanded_velocity) == pytest.approx(math.hypot(*v0))
    # The trajectory must still come to rest exactly at p0.
    end_position, end_velocity = trajectory.state_at(trajectory.duration)
    assert end_position == pytest.approx(p0)
    assert end_velocity == pytest.approx((0.0, 0.0))


def test_first_collision_numba_grants_escape_grace_when_starting_inside_obstacle():
    """Pins 2e53f3e ("Fix trajsample planner deadlock when a robot starts
    inside an obstacle").

    A robot can begin a plan already inside another obstacle's clearance
    envelope (e.g. `GiveAndGoTactic`'s abandoned receiver, standing where it
    waited to catch a pass, the instant the ball stops being exempted as its
    own target). Pre-fix, `d < margin` fired at the very first sample
    (`t == start_t`) for every candidate trajectory regardless of direction --
    the robot's own starting position was "the collision" -- so a trajectory
    heading directly AWAY from the obstacle was still rejected at t=0.
    """
    from utama_core.motion_planning.src.trajsampling.config import (
        trajsamplingconfig as config,
    )

    # Ball-sized obstacle 0.05m away: closer than the combined robot+obstacle
    # clearance radius (ROBOT_RADIUS + 0.0215 = 0.1115m), so the robot starts
    # already penetrating it.
    obstacle = ConstantVelocityObstacle(p0=(0.05, 0.0), v=(0.0, 0.0), radius=0.0215)
    assert 0.05 - config.ROBOT_RADIUS - obstacle.radius < 0.0  # sanity: really penetrating at t=0

    # Trajectory heads straight away from the obstacle.
    trajectory = Trajectory2D.compute((0.0, 0.0), (0.0, 0.0), (-2.0, 0.0), v_max=2.0, a_max=2.0)
    leg1_args, switch_t, leg2_args, duration = _flatten_query_trajectory(trajectory)
    static_arr, traj_arr, cv_arr, enemy_arr = _flatten_obstacles([obstacle])

    t_col = collision.first_collision_numba(
        leg1_args[0],
        leg1_args[1],
        leg1_args[2],
        leg1_args[3],
        leg1_args[4],
        switch_t,
        leg2_args[0],
        leg2_args[1],
        leg2_args[2],
        leg2_args[3],
        leg2_args[4],
        duration,
        0.0,
        config.MAX_LOOKAHEAD_TIME,
        config.ROBOT_RADIUS,
        config.MARGIN_V_MAX,
        config.MARGIN_BASE,
        config.MAX_TIME_STEP,
        config.MIN_TIME_STEP,
        config.STEP_DISTANCE_RATIO,
        static_arr,
        traj_arr,
        cv_arr,
        enemy_arr,
    )

    assert t_col == -1.0  # no collision reported: escaping the obstacle it started inside


def _robot(rid: int, x: float, y: float, is_friendly: bool) -> Robot:
    return Robot(
        id=rid,
        is_friendly=is_friendly,
        has_ball=False,
        p=Vector2D(x, y),
        v=Vector2D(0, 0),
        a=Vector2D(0, 0),
        orientation=0.0,
    )


def _game(robot_xy: tuple, ball_xy: tuple) -> tuple:
    zero = Vector3D(0, 0, 0)
    frame = GameFrame(
        ts=0.0,
        my_team_is_yellow=True,
        my_team_is_right=False,
        friendly_robots={0: _robot(0, *robot_xy, True)},
        enemy_robots={},
        ball=Ball(p=Vector3D(ball_xy[0], ball_xy[1], 0), v=zero, a=zero),
    )
    field = Field(
        my_team_is_right=False, field_dims=STANDARD_FIELD_DIMS, field_bounds=STANDARD_FIELD_DIMS.full_field_bounds
    )
    return Game(past=GameHistory(10), current=frame, field=field), field


def test_try_reuse_tolerates_sub_millimetre_target_jitter():
    """Pins 5183ed1 ("Fix trajsample planner target-jitter stall on
    DIRECT_FREE restarts").

    `_try_reuse` used to compare the caller's target against the committed
    one with exact tuple equality. `DirectFreeOursStep`'s kicker-approach
    point is recomputed every tick from the ball's live position, so
    sub-millimetre physics-sim jitter alone counted as a "changed target" on
    every call, forcing a full replan from t=0 forever (`elapsed` pinned at
    0.0, the robot crawling instead of executing its committed trajectory).
    """
    planner = TrajectorySamplingPlanner(v_max=2.0, a_max=2.0)
    p0 = (0.0, 0.0)
    target = (2.0, 0.0)
    trajectory = Trajectory2D.compute(p0, (0.0, 0.0), target, planner.v_max, planner.a_max)
    planner._commit(0, ts=0.0, trajectory=trajectory, target_pos=target)

    obstacle_arrays = _flatten_obstacles([])
    jittered_target = (target[0] + 1e-4, target[1])  # ~0.1mm jitter, well under the 0.01m tolerance

    result = planner._try_reuse(
        robot_id=0,
        ts=0.1,
        p0=(0.05, 0.0),
        target_pos=jittered_target,
        obstacles=[],
        obstacle_arrays=obstacle_arrays,
    )

    assert result is not None
    assert result.elapsed == pytest.approx(0.1)  # executing the committed trajectory, not restarted from t=0


def test_try_reuse_notices_a_priority_obstacle_only_encroaching_later_in_the_trajectory():
    """Fix #3 from the item-15 Sumatra audit: `_try_reuse`'s priority
    re-check used to sample only two instants (`elapsed`,
    `elapsed + MIN_TIME_STEP`) rather than scanning ahead the way
    `_first_collision` itself does.

    Originally widened to scan the WHOLE remaining trajectory (capped by
    `MAX_LOOKAHEAD_TIME`, 1.5s) -- reverted to a much shorter dedicated cap,
    `_PRIORITY_RECHECK_LOOKAHEAD_TIME` (0.5s, see its docstring), after that
    full-length scan caused a 231/231 full-catalog stall: at a kickoff/
    formation restart every non-keeper teammate simultaneously replans a
    multi-second approach every tick, so within a 1.5s window there is
    almost always SOME higher-priority teammate's (itself about to be
    replaced) trajectory crossing somewhere, permanently invalidating a
    perfectly good plan. `_PRIORITY_RECHECK_LOOKAHEAD_TIME` still comfortably
    covers a genuinely imminent conflict while not reaching far enough to
    catch two mutually-unsettled robots' predictions of each other.

    Builds a stationary higher-priority teammate positioned exactly on the
    committed trajectory's path at t=0.3s (well within
    `_PRIORITY_RECHECK_LOOKAHEAD_TIME`, and still well past `MIN_TIME_STEP`
    so the OLD two-instant check would have missed it), offset in y so the
    gap sits between `_first_collision`'s dynamic margin at that speed
    (~0.089m, so `_first_collision` reports no collision at all) and the
    fixed `MARGIN_BASE` (0.2m) that `_blocked_by_priority_obstacle` checks --
    i.e. collision-free by the harder threshold, but priority-blocked by the
    softer one.
    """
    from utama_core.motion_planning.src.trajsampling.config import (
        trajsamplingconfig as config,
    )

    planner = TrajectorySamplingPlanner(v_max=2.0, a_max=2.0)
    p0 = (0.0, 0.0)
    target = (3.0, 0.0)
    trajectory = Trajectory2D.compute(p0, (0.0, 0.0), target, planner.v_max, planner.a_max)
    planner._commit(0, ts=0.0, trajectory=trajectory, target_pos=target)

    encroach_t = 0.3
    robot_pos, _ = trajectory.state_at(encroach_t)
    obstacle_pos = (robot_pos[0], robot_pos[1] + 0.15 + 2 * config.ROBOT_RADIUS)
    obstacle_trajectory = Trajectory2D.compute(obstacle_pos, (0.0, 0.0), obstacle_pos, 2.0, 2.0)
    # owner_id=99 outranks robot_id=0 under `_has_priority` (higher id wins).
    obstacle = _CommittedTrajectoryObstacle(
        trajectory=obstacle_trajectory, radius=config.ROBOT_RADIUS, time_offset=0.0, owner_id=99
    )
    obstacle_arrays = _flatten_obstacles([obstacle])

    result = planner._try_reuse(
        robot_id=0,
        ts=0.0,
        p0=p0,
        target_pos=target,
        obstacles=[obstacle],
        obstacle_arrays=obstacle_arrays,
    )

    assert result is None  # must trigger a fresh replan, not keep executing straight into the encroachment


def test_try_reuse_does_not_invalidate_on_a_priority_conflict_beyond_the_recheck_window():
    """Regression for the 231/231 full-catalog stall this session traced to
    `_try_reuse`'s priority re-check (see roadmap item 15/16): when it scanned
    the WHOLE remaining trajectory (up to the old `MAX_LOOKAHEAD_TIME`, 1.5s)
    instead of the current, much shorter `_PRIORITY_RECHECK_LOOKAHEAD_TIME`
    (0.5s), every `PREPARE_KICKOFF_YELLOW` restart stalled forever: all 5
    non-keeper teammates simultaneously replan multi-second approach
    trajectories every tick to converge on formation, so within a 1.5s window
    there was almost always some higher-priority teammate's (itself about to
    be replaced) trajectory crossing somewhere, permanently invalidating an
    otherwise perfectly good, collision-clear plan.

    Same construction as the sibling test above, but the obstacle only
    encroaches at t=1.0s -- inside the OLD 1.5s window (which would wrongly
    invalidate this reuse) but outside the current 0.5s window. A conflict
    this far out, against an obstacle that is itself a teammate's committed
    trajectory (likely to be replaced before t=1.0s arrives anyway), must not
    throw away an otherwise-valid reuse -- `_first_collision`'s own
    `MAX_LOOKAHEAD_TIME`-capped scan remains the backstop for anything
    genuinely dangerous further out.
    """
    from utama_core.motion_planning.src.trajsampling.config import (
        trajsamplingconfig as config,
    )

    planner = TrajectorySamplingPlanner(v_max=2.0, a_max=2.0)
    p0 = (0.0, 0.0)
    target = (3.0, 0.0)
    trajectory = Trajectory2D.compute(p0, (0.0, 0.0), target, planner.v_max, planner.a_max)
    planner._commit(0, ts=0.0, trajectory=trajectory, target_pos=target)

    encroach_t = 1.0
    robot_pos, _ = trajectory.state_at(encroach_t)
    obstacle_pos = (robot_pos[0], robot_pos[1] + 0.15 + 2 * config.ROBOT_RADIUS)
    obstacle_trajectory = Trajectory2D.compute(obstacle_pos, (0.0, 0.0), obstacle_pos, 2.0, 2.0)
    # owner_id=99 outranks robot_id=0 under `_has_priority` (higher id wins).
    obstacle = _CommittedTrajectoryObstacle(
        trajectory=obstacle_trajectory, radius=config.ROBOT_RADIUS, time_offset=0.0, owner_id=99
    )
    obstacle_arrays = _flatten_obstacles([obstacle])

    result = planner._try_reuse(
        robot_id=0,
        ts=0.0,
        p0=p0,
        target_pos=target,
        obstacles=[obstacle],
        obstacle_arrays=obstacle_arrays,
    )

    assert result is not None  # the t=1.0s conflict is beyond the recheck window -- reuse must survive


def test_try_reuse_ignores_priority_conflict_when_priority_blocking_disabled():
    """Pins the restart-formation fix (roadmap item 15/16, session following
    the recheck-window fix above): even after shrinking the recheck window to
    0.5s, a real kickoff still stalled 100% of the time in a live
    `debug_match.py` run, traced to a DIFFERENT mechanism than the recheck
    window -- priority-blocking itself, not just how far ahead it looks.

    At a `PREPARE_KICKOFF_YELLOW`/similar restart, every non-keeper teammate
    commits a fresh multi-second approach trajectory simultaneously, every
    tick, to converge on formation; none of them settle for more than a
    fraction of a second. The kicker is always the lowest non-keeper robot
    ID, so by `_has_priority`'s fixed "higher ID wins" ordering it is also
    the LOWEST-priority outfield robot -- every other teammate's transient,
    about-to-be-replaced path outranks it. Confirmed live: the kicker's
    direct line to the ball was priority-blocked on nearly every tick,
    forcing a `_two_segment_candidates` fallback through a randomly-sampled
    intermediate waypoint each time; since that intermediate target kept
    getting invalidated before its own switch point was ever reached, the
    kicker executed repeated short first-leg bursts in place and
    `PREPARE_KICKOFF_YELLOW` never auto-advanced (traced: held for the full
    match duration in every run tested, including a 231/231 full-catalog
    stall). Disabling priority-blocking for the restart window (via
    `priority_enabled=False`, threaded from `_priority_blocking_enabled`)
    resolved it in every run tested, without touching `_first_collision`,
    `_collision_leniency_accepts`, or the emergency-brake layer, so ordinary
    (non-priority) collision avoidance during the restart is unaffected.

    Same construction as the very first priority test above (a stationary
    higher-priority obstacle sitting on the trajectory at t=0.3s, safely
    within `_PRIORITY_RECHECK_LOOKAHEAD_TIME`, so `priority_enabled=True`
    would still reject the reuse) -- but passes `priority_enabled=False` and
    asserts the reuse now survives instead.
    """
    from utama_core.motion_planning.src.trajsampling.config import (
        trajsamplingconfig as config,
    )

    planner = TrajectorySamplingPlanner(v_max=2.0, a_max=2.0)
    p0 = (0.0, 0.0)
    target = (3.0, 0.0)
    trajectory = Trajectory2D.compute(p0, (0.0, 0.0), target, planner.v_max, planner.a_max)
    planner._commit(0, ts=0.0, trajectory=trajectory, target_pos=target)

    encroach_t = 0.3
    robot_pos, _ = trajectory.state_at(encroach_t)
    obstacle_pos = (robot_pos[0], robot_pos[1] + 0.15 + 2 * config.ROBOT_RADIUS)
    obstacle_trajectory = Trajectory2D.compute(obstacle_pos, (0.0, 0.0), obstacle_pos, 2.0, 2.0)
    # owner_id=99 outranks robot_id=0 under `_has_priority` (higher id wins) --
    # would block reuse if priority_enabled were True (see the sibling test).
    obstacle = _CommittedTrajectoryObstacle(
        trajectory=obstacle_trajectory, radius=config.ROBOT_RADIUS, time_offset=0.0, owner_id=99
    )
    obstacle_arrays = _flatten_obstacles([obstacle])

    result = planner._try_reuse(
        robot_id=0,
        ts=0.0,
        p0=p0,
        target_pos=target,
        obstacles=[obstacle],
        obstacle_arrays=obstacle_arrays,
        priority_enabled=False,
    )

    assert result is not None  # priority-blocking disabled -- the conflict must not invalidate reuse


@pytest.mark.parametrize(
    "referee_command,expected",
    [
        (RefereeCommand.NORMAL_START, True),
        (RefereeCommand.FORCE_START, True),
        (RefereeCommand.PREPARE_KICKOFF_YELLOW, False),
        (RefereeCommand.PREPARE_KICKOFF_BLUE, False),
        (RefereeCommand.PREPARE_PENALTY_YELLOW, False),
        (RefereeCommand.DIRECT_FREE_YELLOW, False),
        (RefereeCommand.STOP, False),
        (RefereeCommand.HALT, False),
        (None, True),  # no referee data at all -- default to the safer (priority-blocking-on) behaviour
    ],
)
def test_priority_blocking_enabled_only_during_live_play(referee_command, expected):
    """Live play (`NORMAL_START`/`FORCE_START`, mirroring
    `utama_core.engine.match_stats`'s own `_LIVE_PLAY_COMMANDS`) keeps
    teammate priority-blocking active -- that's the one-or-two-robots-moving
    steady state the mechanism was built and validated for (mirror_swap, see
    the module docstring). Every other command is a restart/formation phase
    where a whole side's non-keeper robots replan simultaneously every tick
    -- see `_priority_blocking_enabled`'s docstring for why priority-blocking
    is actively harmful there instead.
    """

    class _StubReferee:
        def __init__(self, command):
            self.referee_command = command

    class _StubGame:
        def __init__(self, command):
            self.referee = _StubReferee(command) if command is not None else None

    assert _priority_blocking_enabled(_StubGame(referee_command)) is expected


def test_plan_exempts_ball_as_obstacle_when_targeting_it():
    """Pins the `planner.py` half of c4ad99c ("Fix two bugs blocking
    trajsample from playing a real match").

    `go_to_ball` deliberately targets a point past the ball's own centre so
    the robot's controller drives through to actual contact. Pre-fix, the
    ball was added as an unconditional collision obstacle for every robot's
    obstacle set -- including the robot whose own target IS the ball -- so
    every candidate collided with the ball itself before reaching the
    target and `plan()` never returned a clean approach.
    """
    game, field = _game(robot_xy=(-1.0, 0.0), ball_xy=(0.0, 0.0))
    planner = TrajectorySamplingPlanner(v_max=2.0, a_max=2.0)

    # Instrument `_first_collision` to capture the CV-obstacle row count on
    # the very first call (the direct trajectory, checked before any
    # fallback candidate search) without altering planner behaviour.
    first_call_cv_row_count = []
    original_first_collision = planner._first_collision

    def _traced_first_collision(trajectory, obstacle_arrays, start_t=0.0):
        if not first_call_cv_row_count:
            first_call_cv_row_count.append(obstacle_arrays[2].shape[0])
        return original_first_collision(trajectory, obstacle_arrays, start_t)

    planner._first_collision = _traced_first_collision

    overshoot_target = (0.03, 0.0)  # just past the ball's centre, like go_to_ball's real target
    planner.plan(game, robot_id=0, target_pos=overshoot_target, field_bounds=field.full_field_bounds)

    assert first_call_cv_row_count == [0]  # the ball must not appear as a CV obstacle for its own fetcher


def _own_obstacle_after(planner, robot_xy, ts):
    """Obstacle the planner would present for friendly robot 4, really at `robot_xy`, on tick `ts`."""
    robot = _robot(4, robot_xy[0], robot_xy[1], True)
    return planner._own_robot_obstacle(robot, 0.09, ts)


def test_own_robot_obstacle_falls_back_to_real_state_once_robot_leaves_its_stale_plan():
    """Pins the DIRECT_FREE ghost-obstacle deadlock (roadmap item 15).

    Robot 4 committed a plan ending next to the ball, then stopped being
    planned (`empty_command()` during a restart) and was later found 2m
    away. Its committed trajectory, clamped at its endpoint, must no longer
    be used as its obstacle: otherwise the endpoint sits on the ball as a
    priority-carrying ghost and the lower-ranked kicker can never approach.
    """
    planner = TrajectorySamplingPlanner(v_max=2.0, a_max=2.0)
    endpoint = (0.0, 1.0)
    trajectory = Trajectory2D.compute((-2.0, 1.0), (0.0, 0.0), endpoint, planner.v_max, planner.a_max)
    planner._commit(4, ts=0.0, trajectory=trajectory, target_pos=endpoint)

    real = (-1.1, -1.4)
    obstacle = _own_obstacle_after(planner, real, ts=trajectory.duration + 4.0)

    assert isinstance(obstacle, ConstantVelocityObstacle)
    assert obstacle.distance_at(0.0, real) == pytest.approx(-0.09)  # centred on the real position
    assert obstacle.distance_at(0.0, endpoint) > 1.0  # the old endpoint is free again


def test_own_robot_obstacle_keeps_priority_for_robot_resting_on_its_completed_plan():
    """The fallback must be gated on divergence, not on elapsed time: a
    robot that finished its plan and is sitting on the endpoint is exactly
    where the plan says, and must keep its committed-trajectory obstacle
    (and so its `owner_id` priority) even long after `duration` -- an
    elapsed-time-only rule demotes every robot between plans and froze
    kickoffs when tried.
    """
    planner = TrajectorySamplingPlanner(v_max=2.0, a_max=2.0)
    endpoint = (0.0, 1.0)
    trajectory = Trajectory2D.compute((-2.0, 1.0), (0.0, 0.0), endpoint, planner.v_max, planner.a_max)
    planner._commit(4, ts=0.0, trajectory=trajectory, target_pos=endpoint)

    for ts in (trajectory.duration * 0.5, trajectory.duration + 0.02, trajectory.duration + 4.0):
        (ex, ey), _ = trajectory.state_at(min(ts, trajectory.duration))
        obstacle = _own_obstacle_after(planner, (ex + 0.03, ey), ts)  # within tracking tolerance
        assert getattr(obstacle, "owner_id", None) == 4, f"lost priority at ts={ts}"


def test_intermediate_targets_drops_a_stale_backward_pointing_last_target():
    """Regression for the DIRECT_FREE_BLUE congestion stall live-traced on
    `clear_danger_vs_shadow_switch` (roadmap item 15). `_intermediate_targets`
    always retried the previous tick's winning detour target FIRST, and
    `plan()` commits to the first collision-free candidate it finds without
    ever comparing it against the freshly-sorted, toward-target candidates
    later in the list. A `last` target sitting in open space behind the
    robot stays collision-free indefinitely, so once picked (for whatever
    now-irrelevant earlier situation) it kept winning forever: every replan
    committed a short first-leg burst backward, then a second leg that had
    to kill that backward velocity before making any real progress -- net
    near-zero displacement, repeating for the rest of the restart, with
    `has_collision=False` on every single call (traced: `last` 166 degrees
    off the goal direction). Fixed by dropping `last` outright, not just
    de-prioritizing it, once it's more than 90 degrees off the current
    final-target direction.
    """
    planner = TrajectorySamplingPlanner(v_max=2.0, a_max=2.0)
    p0 = (0.4, -1.6)
    final_target = (4.37, -0.57)  # far ahead in +x, matching the traced case

    # A stale winner from some earlier situation, now almost directly
    # behind the robot relative to the current final target (~166 degrees
    # off-axis in the traced case -- use the same shape here).
    planner._last_intermediate_target[0] = (-2.09, -1.55)
    stale_target = planner._last_intermediate_target[0]

    candidates = planner._intermediate_targets(robot_id=0, p0=p0, final_target=final_target)

    from utama_core.motion_planning.src.trajsampling.config import (
        trajsamplingconfig as config,
    )

    # The stale target must be dropped outright, not merely reordered --
    # config.N_INTERMEDIATE_TARGETS fresh candidates only, none of them the
    # excluded stale point.
    assert stale_target not in candidates
    assert len(candidates) == config.N_INTERMEDIATE_TARGETS

    # The surviving (fresh) candidates are still sorted toward the goal --
    # the first one's angular distance must be no worse than the last's.
    fx, fy = final_target[0] - p0[0], final_target[1] - p0[1]
    final_angle = math.atan2(fy, fx)

    def _angular_distance(t):
        cand_angle = math.atan2(t[1] - p0[1], t[0] - p0[0])
        diff = abs(cand_angle - final_angle)
        return min(diff, 2 * math.pi - diff)

    distances = [_angular_distance(c) for c in candidates]
    assert distances == sorted(distances)


def test_intermediate_targets_keeps_a_last_target_that_is_still_a_reasonable_detour():
    """Companion to the stale-target regression above: a cached `last`
    target that's merely angled (a genuine sidestep, not a reversal) must
    still be tried first -- this is the actual stabilizing behaviour
    `_intermediate_targets` exists for, and the stale-target fix must not
    remove it for ordinary detours.
    """
    planner = TrajectorySamplingPlanner(v_max=2.0, a_max=2.0)
    p0 = (0.0, 0.0)
    final_target = (4.0, 0.0)

    # 45 degrees off-axis -- a plausible sidestep around an obstacle, well
    # under the 90-degree staleness threshold.
    planner._last_intermediate_target[0] = (1.0, 1.0)

    candidates = planner._intermediate_targets(robot_id=0, p0=p0, final_target=final_target)

    assert candidates[0] == (1.0, 1.0)


def test_collision_leniency_accepts_a_slow_collision_near_the_destination():
    """Ports the leniency half of TIGERs' `MovingObstacleResultAcceptor.
    accept` (confirmed against the real Sumatra source, not just the paper):
    a non-priority collision near the final destination, at a speed under
    `COLLISION_SPEED_THRESHOLD_MPS`, must be accepted outright -- "be a bit
    aggressive to non-priority obstacles." Without this, every ordinary slow
    final approach near another robot was ranked down by raw survival time
    exactly like a genuine head-on collision, and the planner had no way to
    ever accept a completely normal, low-speed arrival.
    """
    planner = TrajectorySamplingPlanner(v_max=2.0, a_max=2.0)
    target = (2.0, 0.0)
    # Trajectory arriving at the target with near-zero terminal velocity --
    # a normal bang-bang arrival, collision registered right at the end.
    trajectory = Trajectory2D.compute((0.0, 0.0), (0.0, 0.0), target, planner.v_max, planner.a_max)
    collision_time = trajectory.duration  # at the very end: at rest, at the destination

    assert planner._collision_leniency_accepts(trajectory, collision_time, v0=(0.0, 0.0), target_pos=target)


def test_collision_leniency_rejects_a_fast_collision_far_from_the_destination():
    """Companion: a collision far from the final destination, or one at
    speed with plenty of braking distance remaining, must NOT be excused --
    only near-arrival, low-speed (or already-past-brake-time) collisions get
    Sumatra's leniency.
    """
    planner = TrajectorySamplingPlanner(v_max=2.0, a_max=2.0)
    target = (10.0, 0.0)
    # Fast mid-path collision, nowhere near the (distant) final target.
    trajectory = Trajectory2D.compute((0.0, 0.0), (0.0, 0.0), target, planner.v_max, planner.a_max)
    collision_time = 0.5  # early in a long trajectory -- far from target, still near top speed

    assert not planner._collision_leniency_accepts(trajectory, collision_time, v0=(0.0, 0.0), target_pos=target)


def test_plan_accepts_direct_trajectory_through_a_slow_obstacle_parked_on_target():
    """End-to-end: a non-priority teammate parked exactly on the target must
    not force the planner into `best_fallback`'s survival-time ranking
    forever -- a normal, collision-registering-at-arrival direct trajectory
    must be accepted outright once no priority obstacle blocks it.

    Robot 1 (planned robot) outranks robot 0 (stationary obstacle) per
    `_has_priority` (higher id wins), so robot 0 is a NON-priority obstacle
    from robot 1's perspective -- exactly the case Sumatra's leniency
    applies to (a priority obstacle stays an unconditional reject; see
    `_blocked_by_priority_obstacle`).
    """
    zero = Vector3D(0, 0, 0)
    target = (0.0, 0.0)
    frame = GameFrame(
        ts=0.0,
        my_team_is_yellow=True,
        my_team_is_right=False,
        friendly_robots={
            0: _robot(0, target[0], target[1], True),  # parked on the target -- non-priority obstacle
            1: _robot(1, -1.0, 0.0, True),  # the robot being planned -- close enough that the
            # collision near arrival falls within MAX_LOOKAHEAD_TIME (1.5s)
        },
        enemy_robots={},
        ball=Ball(p=Vector3D(5.0, 5.0, 0), v=zero, a=zero),  # far away, irrelevant to this approach
    )
    field = Field(
        my_team_is_right=False, field_dims=STANDARD_FIELD_DIMS, field_bounds=STANDARD_FIELD_DIMS.full_field_bounds
    )
    game = Game(past=GameHistory(10), current=frame, field=field)
    planner = TrajectorySamplingPlanner(v_max=2.0, a_max=2.0)

    result = planner.plan(game, robot_id=1, target_pos=target, field_bounds=field.full_field_bounds)

    # Direct trajectory to the target, accepted via leniency (registers as
    # having a collision at/near the very end, at the destination) rather
    # than falling through to a `_two_segment_candidates` detour.
    assert result.has_collision is True
    assert result.trajectory.duration == pytest.approx(
        Trajectory2D.compute((-1.0, 0.0), (0.0, 0.0), target, planner.v_max, planner.a_max).duration
    )


def test_with_current_clearance_excludes_an_obstacle_the_robot_is_still_escaping():
    """Regression for the escaping-grace / clearance-report mismatch
    (roadmap item 15): `_first_collision`'s numba scan grants a one-time
    "still escaping" grace to any obstacle already penetrated at t == the
    query start (see its own docstring -- built for GiveAndGoTactic's
    abandoned-receiver robot, replanning a fresh route the instant the ball
    becomes a real obstacle again). `_with_current_clearance` computes
    `nearest_obstacle_distance`/`closing_speed` -- fed straight into
    `TrajectorySamplingController`'s emergency-brake check -- via a
    completely separate distance computation that didn't know about that
    grace at all: it reported the still-penetrated obstacle as "the nearest
    danger" unconditionally, which could clamp the robot's escape velocity
    down right as it was legitimately getting clear, undercutting the
    numba-level fix via this separate code path.

    Set up: the robot's OWN position (`p0`) is already inside an obstacle's
    dynamic margin (mirrors "plan() already accepted this trajectory because
    the obstacle is being escaped, not approached" -- exactly what a
    collision-free `PlanResult` from `plan()` would look like in that
    situation). A second, genuinely-clear obstacle is also present. The
    penetrated obstacle must NOT be reported as nearest -- the clear one
    must win instead.
    """
    from utama_core.motion_planning.src.trajsampling.config import (
        trajsamplingconfig as config,
    )

    planner = TrajectorySamplingPlanner(v_max=2.0, a_max=2.0)
    p0 = (0.0, 0.0)
    # Trajectory moving away from p0 at speed -- a real "escaping" motion.
    trajectory = Trajectory2D.compute(p0, (1.0, 0.0), (2.0, 0.0), planner.v_max, planner.a_max)
    result = PlanResult(trajectory=trajectory, has_collision=False, collision_time=None, elapsed=0.0)

    # Penetrating obstacle: centred exactly on p0, well inside any margin.
    penetrating = ConstantVelocityObstacle(p0=p0, v=(0.0, 0.0), radius=config.ROBOT_RADIUS)
    # Genuinely clear obstacle, far away.
    clear = ConstantVelocityObstacle(p0=(5.0, 5.0), v=(0.0, 0.0), radius=config.ROBOT_RADIUS)

    updated = planner._with_current_clearance(result, p0, [penetrating, clear])

    expected_clear_distance = clear.distance_at(0.0, p0) - config.ROBOT_RADIUS
    assert updated.nearest_obstacle_distance == pytest.approx(expected_clear_distance)
    assert updated.nearest_obstacle_distance > 1.0  # nowhere near a brake-triggering distance


def test_with_current_clearance_still_reports_a_genuinely_approaching_obstacle():
    """Companion: an obstacle NOT penetrated at the current position must
    still be reported normally -- the escaping-grace exclusion above must
    only ever exclude an obstacle the robot is already inside, never a
    genuinely approaching one that the emergency brake still needs to see.
    """
    from utama_core.motion_planning.src.trajsampling.config import (
        trajsamplingconfig as config,
    )

    planner = TrajectorySamplingPlanner(v_max=2.0, a_max=2.0)
    p0 = (0.0, 0.0)
    trajectory = Trajectory2D.compute(p0, (1.0, 0.0), (2.0, 0.0), planner.v_max, planner.a_max)
    result = PlanResult(trajectory=trajectory, has_collision=False, collision_time=None, elapsed=0.0)

    approaching = ConstantVelocityObstacle(p0=(0.5, 0.0), v=(0.0, 0.0), radius=config.ROBOT_RADIUS)

    updated = planner._with_current_clearance(result, p0, [approaching])

    expected = approaching.distance_at(0.0, p0) - config.ROBOT_RADIUS
    assert updated.nearest_obstacle_distance == pytest.approx(expected)
