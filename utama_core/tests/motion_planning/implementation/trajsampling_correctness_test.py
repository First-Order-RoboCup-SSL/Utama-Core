"""Internal correctness tests for trajectory-sampling motion primitives."""

import math

import numpy as np
import pytest

from utama_core.config.field_params import STANDARD_FIELD_DIMS
from utama_core.entities.data.vector import Vector2D, Vector3D
from utama_core.entities.game import Ball, Field, Game, GameFrame, GameHistory, Robot
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
    TrajectorySamplingPlanner,
    _bangbang_to_row,
    _flatten_obstacles,
    _flatten_query_trajectory,
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
