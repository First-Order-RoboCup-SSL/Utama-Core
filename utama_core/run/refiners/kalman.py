from typing import Optional

import numpy as np

from utama_core.entities.data.vector import Vector3D
from utama_core.entities.data.vision import VisionRobotData
from utama_core.entities.game import Ball, Robot
from utama_core.global_utils.math_utils import deg_to_rad, normalise_heading


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

    def __init__(self, noise_xy_sd: float = 0.01, noise_th_sd_deg: float = 5):
        assert noise_xy_sd > 0, "The standard deviation must be greater than 0"
        assert noise_th_sd_deg > 0, "The standard deviation must be greater than 0"

        # For position
        # s; to be initialised by strategy runner with 1st GameFrame
        self.state_xy = None

        # sigma squared x, sigma squared y
        noise_xy_var = pow(noise_xy_sd, 2)
        var_x, var_y = noise_xy_var, noise_xy_var

        # sigma xy; assume their errors are uncorrelated
        covariance_xy = 0

        dimensions_xy = 2
        self.identity_xy = np.identity(dimensions_xy)

        # R_n
        self.measurement_cov_xy = np.array([[var_x, covariance_xy], [covariance_xy, var_y]])
        # P_n,n; initialised with uncertainty in 1st frame
        self.covariance_mat_xy = self.measurement_cov_xy
        # Q
        self.process_noise_xy = (2 * noise_xy_var) * self.identity_xy

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

        Args:
            new_data (tuple[float, float]): New vision data received (x coordinates in metres, y coordinates in metres),
                passed by filter_data.
            last_robot (Robot): An object storing the robot's last known position and velocity, among others.
            time_elapsed (float): Time since last vision data was received.

        Returns:
            tuple[float, float]: Filtered vision data (x coordinates, y coordinates),
                returned to filter_data for packaging.

        """

        # class Robot: id: int; is_friendly: bool; has_ball: bool
        # p: Vector2D; v: Vector2D; a: Vector2D; orientation: float

        # Phase 0: Initialised with the 1st valid GameFrame (only on initialisation)
        if self.state_xy is None:
            self.state_xy = np.array((last_robot.p.x, last_robot.p.y))

        # Phase 1: Predicting the current state given the last state.
        # u
        control_velocities_xy = np.array((last_robot.v.x, last_robot.v.y))
        # G
        control_mat_xy = time_elapsed * self.identity_xy

        # s_n,n-1
        pred_state_xy = self.state_xy + np.matmul(control_mat_xy, control_velocities_xy)
        # P_n,n-1
        pred_cov_xy = self.covariance_mat_xy + self.process_noise_xy

        # Phase 2: Adjust this prediction based on new data
        if new_data is not None:  # Received frame.
            # z
            measurement_xy = np.array(new_data)

            # K_n
            kalman_gain_xy = np.linalg.solve((pred_cov_xy + self.measurement_cov_xy).T, pred_cov_xy.T).T

            # s_n,n
            self.state_xy = pred_state_xy + np.matmul(kalman_gain_xy, (measurement_xy - pred_state_xy))

            ident_less_kalman_xy = self.identity_xy - kalman_gain_xy
            measurement_uncertainty_xy = np.matmul(
                kalman_gain_xy,
                np.matmul(self.measurement_cov_xy, kalman_gain_xy.T),
            )

            # P_n,n
            self.covariance_mat_xy = (
                np.matmul(ident_less_kalman_xy, np.matmul(pred_cov_xy, ident_less_kalman_xy.T))
                + measurement_uncertainty_xy
            )

        # We can rely on the invariant that vanished frames have null x values
        # as they are imputed with a null VisionRobotData in the Position Refiner.
        else:  # Vanished frame: use predicted values.
            self.state_xy = pred_state_xy
            self.covariance_mat_xy = pred_cov_xy

        return tuple(self.state_xy)

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
            weights_th = (kalman_gain_th, 1 - kalman_gain_th)
            values_th = (measurement_th, self.state_th)
            sines_th = np.dot(weights_th, np.sin(values_th))
            cosines_th = np.dot(weights_th, np.cos(values_th))
            # s_n,n; already wrapped to (-pi, pi] as we're taking a circular average
            self.state_th = float(np.arctan2(sines_th, cosines_th))

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

        # class VisionRobotData: id: int; x: float; y: float; orientation: float
        xy_tuple = (data.x, data.y) if data is not None else None
        x_f, y_f = self._step_xy(xy_tuple, last_frame, time_elapsed)
        th_f = self._step_th(
            data.orientation if data is not None else None,
            last_frame.orientation,
        )

        return VisionRobotData(last_frame.id, x_f, y_f, th_f)


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
    """

    def __init__(self, noise_sd: float = 0.01):
        assert noise_sd > 0, "The standard deviation must be greater than 0"

        # s; to be initialised by strategy runner with 1st GameFrame
        self.state = None

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

        # class Ball: p: Vector3D; v: Vector3D; a: Vector3D

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

            ident_less_kalman = self.identity - kalman_gain
            measurement_uncertainty = np.matmul(kalman_gain, np.matmul(self.measurement_cov, kalman_gain.T))

            # P_n,n
            self.covariance_mat = (
                np.matmul(ident_less_kalman, np.matmul(pred_cov, ident_less_kalman.T)) + measurement_uncertainty
            )

        # We can rely on the invariant that vanished frames have null x values
        # as they are imputed with None by filter_data
        else:  # Vanished frame: use predicted values.
            self.state = pred_state
            self.covariance_mat = pred_cov

        return tuple(self.state)

    def filter_data(self, data: Optional[Ball], last_frame: Ball, time_elapsed: float) -> Ball:
        """
        Performs one prediction–update cycle of the Kalman filter for the ball.

        The ball's position is first predicted using its last known velocity
        and the elapsed time. If new vision data is available, the prediction
        is corrected using the Kalman gain. If the vision frame is missing,
        the predicted state is used directly.

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

        # class Ball: p: Vector3D, v: Vector3D, a: Vector3D
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
