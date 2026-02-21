"""
Correctness tests for KalmanFilter and KalmanFilterBall.

Key properties checked:
- Filtered output is a weighted blend of prediction and measurement.
- Vanished frames fall back to velocity-based prediction.
- Repeated identical measurements converge to the true value.
- Orientation wrapping is handled correctly near the ±π boundary.
- Return types and shapes are always correct.
"""

import math

import numpy as np
import pytest

from utama_core.data_processing.refiners.filters.kalman import (
    KalmanFilter,
    KalmanFilterBall,
)
from utama_core.entities.data.vector import Vector2D, Vector3D
from utama_core.entities.data.vision import VisionRobotData
from utama_core.entities.game import Ball
from utama_core.entities.game.robot import Robot

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_robot(
    x: float = 0.0,
    y: float = 0.0,
    vx: float = 0.0,
    vy: float = 0.0,
    orientation: float = 0.0,
) -> Robot:
    return Robot(
        id=0,
        is_friendly=True,
        has_ball=False,
        p=Vector2D(x, y),
        v=Vector2D(vx, vy),
        a=Vector2D(0, 0),
        orientation=orientation,
    )


def make_ball(
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    vx: float = 0.0,
    vy: float = 0.0,
    vz: float = 0.0,
) -> Ball:
    return Ball(
        p=Vector3D(x, y, z),
        v=Vector3D(vx, vy, vz),
        a=Vector3D(0, 0, 0),
    )


def make_vision(x: float | None, y: float | None, orientation: float | None, robot_id: int = 0) -> VisionRobotData:
    return VisionRobotData(id=robot_id, x=x, y=y, orientation=orientation)


# ---------------------------------------------------------------------------
# KalmanFilter – construction guards
# ---------------------------------------------------------------------------


class TestKalmanFilterInit:
    def test_default_construction(self):
        kf = KalmanFilter()
        assert kf.id == 0

    def test_custom_id(self):
        kf = KalmanFilter(id=3)
        assert kf.id == 3

    def test_zero_noise_xy_raises(self):
        with pytest.raises(AssertionError):
            KalmanFilter(noise_xy_sd=0)

    def test_negative_noise_xy_raises(self):
        with pytest.raises(AssertionError):
            KalmanFilter(noise_xy_sd=-0.5)

    def test_zero_noise_th_raises(self):
        with pytest.raises(AssertionError):
            KalmanFilter(noise_th_sd_deg=0)

    def test_negative_noise_th_raises(self):
        with pytest.raises(AssertionError):
            KalmanFilter(noise_th_sd_deg=-1)


# ---------------------------------------------------------------------------
# KalmanFilter – _step_xy
# ---------------------------------------------------------------------------


class TestKalmanFilterStepXY:
    def _make_last_frame(self, robot: Robot) -> dict[int, Robot]:
        return {robot.id: robot}

    def test_returns_tuple_of_two_floats(self):
        kf = KalmanFilter()
        robot = make_robot(x=1.0, y=2.0)
        result = kf._step_xy((1.0, 2.0), robot, time_elapsed=0.1)
        assert isinstance(result, tuple) and len(result) == 2

    def test_initialises_state_on_first_call(self):
        kf = KalmanFilter()
        assert kf.state_xy is None
        robot = make_robot(x=3.0, y=4.0)
        kf._step_xy((3.0, 4.0), robot, time_elapsed=0.1)
        assert kf.state_xy is not None

    def test_exact_repeated_measurement_converges(self):
        """After many steps with zero velocity and the same measurement the filter should
        converge very close to the true position."""
        kf = KalmanFilter(noise_xy_sd=0.01)
        robot = make_robot(x=2.0, y=5.0)
        for _ in range(50):
            result = kf._step_xy((2.0, 5.0), robot, time_elapsed=0.1)
        assert abs(result[0] - 2.0) < 1e-3
        assert abs(result[1] - 5.0) < 1e-3

    def test_output_between_prediction_and_measurement(self):
        """When prediction and measurement differ, filtered output must lie strictly
        between them (weighted blend)."""
        kf = KalmanFilter()
        robot = make_robot(x=0.0, y=0.0)
        # Initialise filter state
        kf._step_xy((0.0, 0.0), robot, time_elapsed=0.1)

        # Now send a measurement far from the prediction
        robot_stationary = make_robot(x=0.0, y=0.0, vx=0.0, vy=0.0)
        result = kf._step_xy((3.0, 3.0), robot_stationary, time_elapsed=0.1)
        # Should be pulled toward measurement but not all the way
        assert 0.0 < result[0] < 3.0 + 1e-9
        assert 0.0 < result[1] < 3.0 + 1e-9

    def test_vanished_frame_uses_velocity_prediction(self):
        """Consecutive vanished steps must each advance position by exactly v * dt."""
        kf = KalmanFilter()
        vx, vy = 1.0, 0.5
        robot = make_robot(x=0.0, y=0.0, vx=vx, vy=vy)
        kf._step_xy(None, robot, time_elapsed=0.1)  # init (no Kalman update)

        dt = 0.1
        result1 = kf._step_xy(None, robot, time_elapsed=dt)
        result2 = kf._step_xy(None, robot, time_elapsed=dt)
        # Each vanished step should advance by v * dt
        assert abs((result2[0] - result1[0]) - vx * dt) < 1e-6
        assert abs((result2[1] - result1[1]) - vy * dt) < 1e-6

    def test_state_advances_with_velocity_over_multiple_vanished_steps(self):
        kf = KalmanFilter()
        vx = 2.0
        robot = make_robot(x=0.0, y=0.0, vx=vx, vy=0.0)
        kf._step_xy((0.0, 0.0), robot, time_elapsed=0.1)

        dt = 0.1
        steps = 5
        for _ in range(steps):
            result = kf._step_xy(None, robot, time_elapsed=dt)

        # After n vanished steps starting from 0 the position grows monotonically
        assert result[0] > 0.0


# ---------------------------------------------------------------------------
# KalmanFilter – _step_th
# ---------------------------------------------------------------------------


class TestKalmanFilterStepTH:
    def test_returns_float(self):
        kf = KalmanFilter()
        result = kf._step_th(0.5, last_th=0.0)
        assert isinstance(result, float)

    def test_initialises_state_on_first_call(self):
        kf = KalmanFilter()
        assert kf.state_th is None
        kf._step_th(1.0, last_th=1.0)
        assert kf.state_th is not None

    def test_exact_repeated_measurement_converges(self):
        kf = KalmanFilter(noise_th_sd_deg=5)
        th = math.pi / 4
        for _ in range(50):
            result = kf._step_th(th, last_th=th)
        assert abs(result - th) < 1e-3

    def test_vanished_frame_preserves_state(self):
        kf = KalmanFilter()
        kf._step_th(1.0, last_th=1.0)  # init
        state_before = kf.state_th
        kf._step_th(None, last_th=1.0)
        assert kf.state_th == state_before

    def test_output_wrapped_to_minus_pi_plus_pi(self):
        kf = KalmanFilter()
        # Initialise near +π
        kf._step_th(math.pi - 0.1, last_th=math.pi - 0.1)
        # Measurement just past +π (wraps to ≈ -π)
        result = kf._step_th(-math.pi + 0.1, last_th=math.pi)
        assert -math.pi <= result <= math.pi

    def test_circular_average_across_pi_boundary(self):
        """Averaging π-0.1 and -π+0.1 should stay near ±π, not collapse to 0."""
        kf = KalmanFilter()
        th1 = math.pi - 0.1
        th2 = -math.pi + 0.1
        kf._step_th(th1, last_th=th1)
        result = kf._step_th(th2, last_th=th1)
        # Result must be close to ±π, not near 0
        assert abs(result) > math.pi / 2


# ---------------------------------------------------------------------------
# KalmanFilter – filter_data (public API)
# ---------------------------------------------------------------------------


class TestKalmanFilterFilterData:
    def _last_frame(self, robot: Robot) -> dict[int, Robot]:
        return {robot.id: robot}

    def test_returns_vision_robot_data(self):
        kf = KalmanFilter()
        robot = make_robot()
        result = kf.filter_data(make_vision(1.0, 2.0, 0.5), self._last_frame(robot)[robot.id], 0.1)
        assert isinstance(result, VisionRobotData)

    def test_id_preserved(self):
        kf = KalmanFilter(id=2)
        robot = Robot(
            id=2,
            is_friendly=True,
            has_ball=False,
            p=Vector2D(0, 0),
            v=Vector2D(0, 0),
            a=Vector2D(0, 0),
            orientation=0.0,
        )
        result = kf.filter_data(make_vision(1.0, 2.0, 0.3, robot_id=2), robot, 0.1)
        assert result.id == 2

    def test_valid_data_returns_finite_values(self):
        kf = KalmanFilter()
        robot = make_robot(x=0.0, y=0.0)
        result = kf.filter_data(make_vision(1.0, 1.0, 0.2), self._last_frame(robot)[robot.id], 0.1)
        assert math.isfinite(result.x)
        assert math.isfinite(result.y)
        assert math.isfinite(result.orientation)

    def test_vanished_data_returns_finite_values(self):
        kf = KalmanFilter()
        robot = make_robot(x=1.0, y=1.0)
        # First call to initialise
        kf.filter_data(make_vision(1.0, 1.0, 0.0), self._last_frame(robot)[robot.id], 0.1)
        # Second call with vanished data
        result = kf.filter_data(None, self._last_frame(robot)[robot.id], 0.1)
        assert math.isfinite(result.x)
        assert math.isfinite(result.y)
        assert math.isfinite(result.orientation)

    def test_convergence_with_repeated_measurement(self):
        kf = KalmanFilter()
        robot = make_robot(x=5.0, y=3.0)
        for _ in range(50):
            result = kf.filter_data(make_vision(5.0, 3.0, 1.0), self._last_frame(robot)[robot.id], 0.1)
        assert abs(result.x - 5.0) < 1e-3
        assert abs(result.y - 3.0) < 1e-3
        assert abs(result.orientation - 1.0) < 1e-2


# ---------------------------------------------------------------------------
# KalmanFilterBall – construction guards
# ---------------------------------------------------------------------------


class TestKalmanFilterBallInit:
    def test_default_construction(self):
        kf = KalmanFilterBall()
        assert kf.state is None

    def test_zero_noise_raises(self):
        with pytest.raises(AssertionError):
            KalmanFilterBall(noise_sd=0)

    def test_negative_noise_raises(self):
        with pytest.raises(AssertionError):
            KalmanFilterBall(noise_sd=-1)


# ---------------------------------------------------------------------------
# KalmanFilterBall – _step
# ---------------------------------------------------------------------------


class TestKalmanFilterBallStep:
    def test_returns_tuple_of_three_floats(self):
        kf = KalmanFilterBall()
        ball = make_ball()
        result = kf._step((0.0, 0.0, 0.0), ball, time_elapsed=0.1)
        assert isinstance(result, tuple) and len(result) == 3

    def test_initialises_state_on_first_call(self):
        kf = KalmanFilterBall()
        assert kf.state is None
        kf._step((1.0, 2.0, 0.0), make_ball(1.0, 2.0), time_elapsed=0.1)
        assert kf.state is not None

    def test_exact_repeated_measurement_converges(self):
        kf = KalmanFilterBall(noise_sd=0.01)
        ball = make_ball(x=1.0, y=-2.0, z=0.0)
        for _ in range(50):
            result = kf._step((1.0, -2.0, 0.0), ball, time_elapsed=0.1)
        assert abs(result[0] - 1.0) < 1e-3
        assert abs(result[1] - (-2.0)) < 1e-3
        assert abs(result[2] - 0.0) < 1e-3

    def test_vanished_frame_uses_velocity_prediction(self):
        """Consecutive vanished steps must each advance position by exactly v * dt."""
        kf = KalmanFilterBall()
        vx, vy, vz = 1.0, 2.0, 0.5
        ball = make_ball(x=0.0, y=0.0, z=0.0, vx=vx, vy=vy, vz=vz)
        kf._step(None, ball, time_elapsed=0.1)  # init (no Kalman update)

        dt = 0.1
        result1 = kf._step(None, ball, time_elapsed=dt)
        result2 = kf._step(None, ball, time_elapsed=dt)
        # Each vanished step should advance by v * dt
        assert abs((result2[0] - result1[0]) - vx * dt) < 1e-6
        assert abs((result2[1] - result1[1]) - vy * dt) < 1e-6
        assert abs((result2[2] - result1[2]) - vz * dt) < 1e-6

    def test_output_between_prediction_and_measurement(self):
        kf = KalmanFilterBall()
        ball = make_ball()
        kf._step((0.0, 0.0, 0.0), ball, time_elapsed=0.1)

        stationary_ball = make_ball(x=0.0, y=0.0, z=0.0)
        result = kf._step((4.0, 4.0, 1.0), stationary_ball, time_elapsed=0.1)
        assert 0.0 < result[0] < 4.0 + 1e-9
        assert 0.0 < result[1] < 4.0 + 1e-9


# ---------------------------------------------------------------------------
# KalmanFilterBall – filter_data (public API)
# ---------------------------------------------------------------------------


class TestKalmanFilterBallFilterData:
    def test_returns_ball(self):
        kf = KalmanFilterBall()
        ball = make_ball(1.0, 2.0, 0.0)
        result = kf.filter_data(ball, ball, time_elapsed=0.1)
        assert isinstance(result, Ball)

    def test_valid_data_returns_finite_position(self):
        kf = KalmanFilterBall()
        ball = make_ball(1.0, 2.0, 0.0)
        result = kf.filter_data(ball, ball, time_elapsed=0.1)
        assert math.isfinite(result.p.x)
        assert math.isfinite(result.p.y)
        assert math.isfinite(result.p.z)

    def test_none_data_returns_finite_position(self):
        """A None ball (not detected) should produce a predicted ball, not crash."""
        kf = KalmanFilterBall()
        last = make_ball(1.0, 1.0, 0.0)
        kf.filter_data(last, last, time_elapsed=0.1)  # init
        result = kf.filter_data(None, last, time_elapsed=0.1)
        assert isinstance(result, Ball)
        assert math.isfinite(result.p.x)
        assert math.isfinite(result.p.y)
        assert math.isfinite(result.p.z)

    def test_none_data_velocity_zeroed(self):
        """When the ball has vanished, velocity is set to zero in the returned Ball."""
        kf = KalmanFilterBall()
        last = make_ball(0.0, 0.0, 0.0)
        kf.filter_data(last, last, time_elapsed=0.1)
        result = kf.filter_data(None, last, time_elapsed=0.1)
        assert result.v.x == 0.0
        assert result.v.y == 0.0
        assert result.v.z == 0.0

    def test_velocity_passed_through_from_measurement(self):
        """Velocity from a valid Ball measurement should be passed through unchanged."""
        kf = KalmanFilterBall()
        ball = make_ball(1.0, 2.0, 0.0, vx=3.0, vy=4.0, vz=0.5)
        result = kf.filter_data(ball, ball, time_elapsed=0.1)
        assert result.v.x == 3.0
        assert result.v.y == 4.0
        assert result.v.z == 0.5

    def test_convergence_with_repeated_measurement(self):
        kf = KalmanFilterBall(noise_sd=0.01)
        ball = make_ball(x=3.0, y=-1.0, z=0.0)
        for _ in range(50):
            result = kf.filter_data(ball, ball, time_elapsed=0.1)
        assert abs(result.p.x - 3.0) < 1e-3
        assert abs(result.p.y - (-1.0)) < 1e-3
        assert abs(result.p.z - 0.0) < 1e-3

    def test_prediction_tracks_moving_ball_after_vanish(self):
        """After the ball vanishes, successive None steps should accumulate displacement."""
        kf = KalmanFilterBall()
        vx = 1.0
        ball = make_ball(x=0.0, y=0.0, z=0.0, vx=vx)
        kf.filter_data(ball, ball, time_elapsed=0.1)  # init

        dt = 0.1
        steps = 5
        prev_x = 0.0
        for _ in range(steps):
            result = kf.filter_data(None, ball, time_elapsed=dt)
            assert result.p.x > prev_x
            prev_x = result.p.x

    def test_ball_covariance_shrinks_with_repeated_measurement(self):
        """After many identical measurements, the covariance should shrink (filter gains confidence)."""
        kf = KalmanFilterBall(noise_sd=0.001)
        ball = make_ball(x=1.0, y=2.0)
        initial_cov = kf.covariance_mat.copy()
        for _ in range(50):
            kf._step((1.0, 2.0, 0.0), ball, 0.1)

        # Diagonal elements (variances) should shrink
        assert np.all(np.diag(kf.covariance_mat) < np.diag(initial_cov))

    def test_robot_xy_covariance_shrinks_with_repeated_measurement(self):
        kf = KalmanFilter(id=1, noise_xy_sd=0.001, noise_th_sd_deg=5)

        robot = make_robot(x=1.0, y=2.0, orientation=0.0)

        # First call initializes state
        kf._step_xy((1.0, 2.0), robot, time_elapsed=0.1)

        initial_cov = kf.covariance_mat_xy.copy()

        for _ in range(50):
            kf._step_xy((1.0, 2.0), robot, time_elapsed=0.1)

        assert np.all(np.diag(kf.covariance_mat_xy) < np.diag(initial_cov))
