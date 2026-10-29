import math
from typing import Optional

import numpy as np

from utama_core.config.settings import BALL_ROLL_DECEL, EXACT_MATH
from utama_core.entities.data.vector import Vector3D
from utama_core.entities.data.vision import VisionRobotData
from utama_core.entities.game import Ball, Robot
from utama_core.global_utils.math_utils import deg_to_rad, normalise_heading

# An unseen robot is carried on its velocity this long, then held where it is: one seen a few
# times a second on the 2026-10-27 field test, carried on, ran off to 10^7 m at 7 m/s.
ROBOT_COAST_S = 0.3
# The filter's process noise: robots' acceleration as white noise of this standard deviation
# (m/s^2). Well above the real robots' ~5 m/s^2 (2026-10-27 calibration), for starts, stops and
# knocks: at 5 the filter lagged a defender braking at the keep-out 1.3 cm inside it.
ROBOT_ACCEL_SD = 20.0


def _weighted_circular_mean_numpy(weight: float, a: float, b: float) -> float:
    """Circular mean of angles `a` and `b` weighted `weight` and `1 - weight`."""
    weights = (weight, 1 - weight)
    values = (a, b)
    sines = np.dot(weights, np.sin(values))
    cosines = np.dot(weights, np.cos(values))
    return float(np.arctan2(sines, cosines))


def _weighted_circular_mean_fast(weight: float, a: float, b: float) -> float:
    """`_weighted_circular_mean_numpy` with `math` on plain floats (can differ
    from numpy's vectorised trig and 2-element dot in the last bits)."""
    other = 1 - weight
    sines = weight * math.sin(a) + other * math.sin(b)
    cosines = weight * math.cos(a) + other * math.cos(b)
    return math.atan2(sines, cosines)


# See `EXACT_MATH` in settings.py.
_weighted_circular_mean = _weighted_circular_mean_numpy if EXACT_MATH else _weighted_circular_mean_fast


class KalmanFilter:
    """
    Kalman filter for 2D position and orientation of robots.

    It works in 2 phases:
    1. Prediction: The object's last known velocity and kinematics formulae are
    used to estimate its current position. (Orientation is assumed to be constant
    due to a lack of data on angular velocity.)

    2. Updating: New vision data is used to update the filter's estimate. The
    information is weighed using the Kalman gain, a constant that depends on
    the system's noise level.

    If no data is received, the filter's prediction is used.

    Note: The filter's parameters are the standard deviations of random noise.
    The rsim noise generator also uses standard deviation, so the optimal parameter
    should be the same as the argument to it. Do not use variance (standard deviation squared).

    More about the methodology and formulae used can be found at https://kalmanfilter.net/.

    Args:
        id (int): The associated robot's ID, used for associating the filter
            with the robot. Defaults to 0.

        noise_xy_sd (float): A hyper-parameter, used to weigh the filter's
            predictions and the vision data received during the "update" phase.
            Unit is metres.
            Defaults to 0.01 (In simulation, this should match the argument
            passed to the rsim noise generator, but should be adjusted based on
            real-world conditions when live robots are used).

        noise_th_sd_deg (float): A hyper-parameter, used to weigh the filter's
            predictions and the vision data received during the "update" phase.
            Unit is degrees. Conversion to radians is done internally.
            Defaults to 5 (In simulation, this should match the argument
            passed to the rsim noise generator, but should be adjusted based on
            real-world conditions when live robots are used).
    """

    def __init__(self, id: int = 0, noise_xy_sd: float = 0.01, noise_th_sd_deg: float = 5):
        assert noise_xy_sd > 0, "The standard deviation must be greater than 0"
        assert noise_th_sd_deg > 0, "The standard deviation must be greater than 0"

        self.id = id

        # For position: per axis a constant-velocity filter, state (position, velocity).
        # s; to be initialised by strategy runner with 1st GameFrame
        self.state_x = None
        self.state_y = None
        self.velocity_x = 0.0
        self.velocity_y = 0.0
        self._unseen_s = 0.0  # since vision last saw the robot
        self.accel_var = ROBOT_ACCEL_SD**2

        # sigma squared x, sigma squared y. x and y errors are assumed uncorrelated (the
        # measurement/process covariance matrices below are always diagonal), which means the
        # 2D xy filter is mathematically equivalent to two independent scalar filters — same
        # closed-form update as _step_th below, just applied twice. Plain scalar arithmetic,
        # not numpy: _step_xy is called ~43k times per match, where np.matmul/np.linalg.solve's
        # dispatch overhead on a 2x2 matrix dwarfs the actual arithmetic at this call volume.
        noise_xy_var = pow(noise_xy_sd, 2)
        self.measurement_var_x = noise_xy_var
        self.measurement_var_y = noise_xy_var
        # P_n,n per axis as (position var, position-velocity cov, velocity var); initialised with
        # the measurement's uncertainty and a robot's speed unknown to ~1 m/s.
        self.covariance_x = (noise_xy_var, 0.0, 1.0)
        self.covariance_y = (noise_xy_var, 0.0, 1.0)

        # Observation matrix H and state transition matrix F are just the identity matrix.
        # Multiplications with them are omitted.

        # For orientation
        # s; to be initialised by strategy runner with 1st GameFrame
        self.state_th = None

        # sigma squared th
        noise_th_var = pow(deg_to_rad(noise_th_sd_deg), 2)

        # r_n
        self.measurement_cov_th = noise_th_var
        # p_n,n; initialised with uncertainty in 1st frame
        self.covariance_th = noise_th_var
        # q
        self.process_noise_th = noise_th_var

    def _step_xy(
        self,
        new_data: Optional[tuple[float, float]],
        last_robot: Robot,
        time_elapsed: float,
    ) -> tuple[float, float]:
        """
        A single iteration of the filter for x and y coordinates.

        Each axis is a constant-velocity Kalman filter: its own velocity is part of its state,
        corrected by each detection with its own (small) gain. It used to predict with the
        velocity differenced from its own output; a robot vision saw ~20 times a second then
        swung +-0.4 m round where it stood: each snap back onto a detection read as up to
        3 m/s, which it coasted on through the next gap (2026-10-28 field run).

        Args:
            new_data (tuple[float, float]): New vision data received (x coordinates in metres, y coordinates in metres),
                passed by filter_data.
            last_robot (Robot): The robot's last known state; its position and velocity start the filter.
            time_elapsed (float): Time since the previous step.

        Returns:
            tuple[float, float]: Filtered vision data (x coordinates, y coordinates),
                returned to filter_data for packaging.

        """
        if self.state_x is None:
            self.state_x, self.state_y = last_robot.p.x, last_robot.p.y
            self.velocity_x, self.velocity_y = last_robot.v.x, last_robot.v.y

        new_x, new_y = (None, None) if new_data is None else new_data
        if new_data is None:
            # Only the part of this step still within ROBOT_COAST_S moves the robot.
            moving_s = max(0.0, min(time_elapsed, ROBOT_COAST_S - self._unseen_s))
            self._unseen_s += max(0.0, time_elapsed)
        else:
            moving_s = time_elapsed
            self._unseen_s = 0.0

        self.state_x, self.velocity_x, self.covariance_x = self._step_axis(
            new_x, self.state_x, self.velocity_x, moving_s, time_elapsed, self.covariance_x, self.measurement_var_x
        )
        self.state_y, self.velocity_y, self.covariance_y = self._step_axis(
            new_y, self.state_y, self.velocity_y, moving_s, time_elapsed, self.covariance_y, self.measurement_var_y
        )
        return (self.state_x, self.state_y)

    def _step_axis(
        self,
        new_data: Optional[float],
        position: float,
        velocity: float,
        moving_s: float,
        time_elapsed: float,
        covariance: tuple[float, float, float],
        measurement_var: float,
    ) -> tuple[float, float, tuple[float, float, float]]:
        """One prediction-update cycle of one axis's constant-velocity filter: (position,
        velocity, covariance). It moves on its velocity for `moving_s` of the `time_elapsed`."""
        p, c, v = covariance
        dt = moving_s
        # Phase 1: predict. F = [[1, dt], [0, 1]]; Q from white acceleration over the whole step.
        position = position + velocity * dt
        q = self.accel_var
        t = max(0.0, time_elapsed)
        p, c, v = (
            p + 2 * dt * c + dt * dt * v + q * t**4 / 4,
            c + dt * v + q * t**3 / 2,
            v + q * t * t,
        )
        if new_data is None:  # Vanished frame: the prediction stands.
            return position, velocity, (p, c, v)
        # Phase 2: update with the measured position (H = [1, 0]).
        s = p + measurement_var
        k_p, k_v = p / s, c / s
        residual = new_data - position
        position += k_p * residual
        velocity += k_v * residual
        p, c, v = (1 - k_p) * p, (1 - k_p) * c, v - k_v * c
        return position, velocity, (p, c, v)

    def _step_th(self, new_data: Optional[float], last_th: float) -> float:
        """
        A single iteration of the filter for orientation.

        Args:
            new_data (float): New vision data received (orientation in radians),
                passed by the externally callable function filter_data.
            last_th (float): The robot's last known orientation

        Returns:
            float: Filtered vision data orientation,
                returned to filter_data for packaging.

        """

        # Phase 0: Initialised with the 1st valid GameFrame (only on initialisation)
        if self.state_th is None:
            self.state_th = last_th

        # Phase 1: Predicting the current state given the last state.
        # s_n,n-1 = s_n-1,n-1 (Assuming constant velocity)
        # P_n,n-1
        pred_cov_th = self.covariance_th + self.process_noise_th

        # Phase 2: Adjust this prediction based on new data
        if new_data is not None:  # Received frame.
            # z
            measurement_th = normalise_heading(new_data)

            # K_n
            kalman_gain_th = pred_cov_th / (pred_cov_th + self.measurement_cov_th)

            # Taking a circular weighted average
            # s_n,n; already wrapped to (-pi, pi] as we're taking a circular average
            self.state_th = _weighted_circular_mean(kalman_gain_th, measurement_th, self.state_th)

            # P_n,n
            self.covariance_th = (1 - kalman_gain_th) * pred_cov_th

        # We can rely on the invariant that vanished frames have null x values
        # as they are imputed with a null VisionRobotData in the Position Refiner.
        else:  # Vanished frame: use predicted values.
            # self.state_th is unchanged
            self.covariance_th = pred_cov_th

        return self.state_th

    def filter_data(
        self,
        data: Optional[VisionRobotData],
        last_frame: Robot,
        time_elapsed: float,
    ) -> VisionRobotData:
        """
        Performs one prediction–update cycle of the Kalman filter for the
        associated robot.

        The robot's state is first predicted using its last known velocity and
        the elapsed time. If new vision data is available, the prediction is
        corrected using the Kalman gain. If the vision frame is missing, the
        predicted state is used directly.

        Args:
            data (VisionRobotData): New vision measurement containing position
                (x, y) in metres and orientation in radians.
            last_frame (dict[int, Robot]): Mapping of robot IDs to their last
                known state (position, velocity, orientation), used for motion
                prediction.
            time_elapsed (float): Time in seconds since the previous update.

        Returns:
            VisionRobotData: Filtered estimate of the robot's position (x, y)
                and orientation, packaged as a VisionRobotData object.
        """

        xy_tuple = (data.x, data.y) if data is not None else None
        x_f, y_f = self._step_xy(xy_tuple, last_frame, time_elapsed)
        th_f = self._step_th(
            data.orientation if data is not None else None,
            last_frame.orientation,
        )

        return VisionRobotData(self.id, x_f, y_f, th_f)


class KalmanFilterBall:
    """
    Kalman filter for 3D position of ball.

    See above for details about the methodology.

    Args:
        noise_sd (float): A hyper-parameter, used to weigh the filter's
            predictions and the vision data received during the "update" phase.
            Unit is metres.
            Defaults to 0.01 (In simulation, this should match the argument
            passed to the rsim noise generator, but should be adjusted based on
            real-world conditions when live robots are used).
        bounds (Optional[tuple[float, float, float, float]]): (x_min, x_max, y_min, y_max) an
            unseen ball is kept within (m); None for no limit.

    While the ball is unseen it rolls on from where it was lost, slowing at `BALL_ROLL_DECEL`
    until it stops, and never leaves `bounds`. (Carrying its last velocity unchanged, a ball out
    of sight flew off for good.)
    """

    def __init__(self, noise_sd: float = 0.01, bounds: Optional[tuple[float, float, float, float]] = None):
        assert noise_sd > 0, "The standard deviation must be greater than 0"

        # s; to be initialised by strategy runner with 1st GameFrame
        self.state = None
        self.bounds = bounds
        # While unseen: (position, xy velocity) when it was lost, and the time since.
        self._lost: Optional[tuple[np.ndarray, np.ndarray]] = None
        self._unseen_s = 0.0

        # sigma squared x, y, z
        noise_var = pow(noise_sd, 2)
        var_x, var_y, var_z = noise_var, noise_var, noise_var

        # sigma xy, xz, yz
        noise_covariance = 0  # assume their errors are uncorrelated
        covariance_xy, covariance_xz, covariance_yz = (
            noise_covariance,
            noise_covariance,
            noise_covariance,
        )

        dimensions = 3
        self.identity = np.identity(dimensions)

        # R_n
        self.measurement_cov = np.array(
            [
                [var_x, covariance_xy, covariance_xz],
                [covariance_xy, var_y, covariance_yz],
                [covariance_xz, covariance_yz, var_z],
            ]
        )
        # P_n,n; initialised with uncertainty in 1st frame
        self.covariance_mat = self.measurement_cov
        # Q
        self.process_noise = (2 * noise_var) * self.identity

        # Observation matrix H and state transition matrix F are just the identity matrix.
        # Multiplications with them are omitted.

    def _step(
        self,
        new_data: Optional[tuple[float, float, float]],
        last_ball: Ball,
        time_elapsed: float,
    ) -> tuple[float, float, float]:
        """
        A single iteration of the filter.

        Args:
            new_data (Optional[tuple[float, float, float]]): New vision data received (xyz coordinates in metres),
                passed by the filter_data function. None if the ball is not detected.
            last_ball (Ball): An object storing the ball's last known position and velocity.
            time_elapsed (float): Time since last vision data was received.

        Returns:
            tuple[float, float, float]: Filtered vision data (xyz coordinates),
                returned to the filter_data function for packaging.
        """

        # Phase 0: Initialised with the 1st valid GameFrame (only on initialisation)
        if self.state is None:
            self.state = np.array((last_ball.p.x, last_ball.p.y, last_ball.p.z))

        # Phase 1: Predicting the current state given the last state.
        # u
        control_velocities = np.array((last_ball.v.x, last_ball.v.y, last_ball.v.z))
        # G
        control_mat = time_elapsed * self.identity
        # s_n,n-1
        pred_state = self.state + np.matmul(control_mat, control_velocities)
        # P_n,n-1
        pred_cov = self.covariance_mat + self.process_noise

        # Phase 2: Adjust this prediction based on new data
        if new_data is not None:  # Received frame.
            # z
            measurement = np.array(new_data)

            # K_n
            kalman_gain = np.linalg.solve((pred_cov + self.measurement_cov).T, pred_cov.T).T

            # s_n,n
            self.state = pred_state + np.matmul(kalman_gain, (measurement - pred_state))
            self._lost = None

            ident_less_kalman = self.identity - kalman_gain
            measurement_uncertainty = np.matmul(kalman_gain, np.matmul(self.measurement_cov, kalman_gain.T))

            # P_n,n
            self.covariance_mat = (
                np.matmul(ident_less_kalman, np.matmul(pred_cov, ident_less_kalman.T)) + measurement_uncertainty
            )

        # We can rely on the invariant that vanished frames have null x values
        # as they are imputed with None by filter_data
        else:  # Vanished frame: the ball rolls on from where it was lost.
            if self._lost is None:
                self._lost = (self.state.copy(), np.array((last_ball.v.x, last_ball.v.y)))
                self._unseen_s = 0.0
            self._unseen_s += time_elapsed
            self.state = self._rolled(*self._lost, self._unseen_s)
            self.covariance_mat = pred_cov

        return tuple(self.state)

    def _rolled(self, p0: np.ndarray, v0: np.ndarray, t: float) -> np.ndarray:
        """Where a ball at `p0` rolling at `v0` is after `t` s of `BALL_ROLL_DECEL`, within `bounds`."""
        speed = math.hypot(v0[0], v0[1])
        xy = p0[:2].copy()
        if speed > 0:
            t = min(t, speed / BALL_ROLL_DECEL)
            xy += v0 / speed * (speed * t - 0.5 * BALL_ROLL_DECEL * t * t)
        if self.bounds is not None:
            x_min, x_max, y_min, y_max = self.bounds
            xy = np.clip(xy, (x_min, y_min), (x_max, y_max))
        return np.array((xy[0], xy[1], p0[2]))

    def filter_data(self, data: Optional[Ball], last_frame: Ball, time_elapsed: float) -> Ball:
        """
        Performs one prediction–update cycle of the Kalman filter for the ball.

        The ball's position is first predicted using its last known velocity
        and the elapsed time. If new vision data is available, the prediction
        is corrected using the Kalman gain. If the vision frame is missing,
        the ball rolls on from where it was lost (see the class docstring).

        Args:
            data (Optional[Ball]): New vision measurement containing the ball's position
                (x, y, z) in metres. May be None if the ball is not detected.
            last_frame (Ball): The ball's last known state, including position,
                velocity, and acceleration, used for motion prediction.
            time_elapsed (float): Time in seconds since the previous update.

        Returns:
            Ball: Filtered estimate of the ball's position, returned as a Ball
                object with updated position and preserved velocity and
                acceleration.
        """

        if data is not None:
            new_data = (data.p.x, data.p.y, data.p.z)
            velocity, acceleration = data.v, data.a
        else:
            # If the ball data vanished, PositionRefiner._get_most_confident_ball returns null.
            new_data = None
            zero_vector = Vector3D(0, 0, 0)
            velocity, acceleration = zero_vector, zero_vector

        filtered_data = self._step(new_data, last_frame, time_elapsed)

        return Ball(Vector3D(*filtered_data), velocity, acceleration)
