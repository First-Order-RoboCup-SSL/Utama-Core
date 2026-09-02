"""Internal correctness tests for trajectory-sampling motion primitives."""

import math

import numpy as np
import pytest

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
