import numpy as np

from utama_core.entities.data.vision import VisionRobotData
from utama_core.entities.game import Robot, Ball
from utama_core.entities.data.vector import Vector3D
from utama_core.global_utils.math_utils import deg_to_rad, normalise_heading
    

class Kalman_filter:
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
            Defaults to 0.01 (but should be adjusted based on real-world conditions).
            
        noise_th_sd_deg (float): A hyper-parameter, used to weigh the filter's
            predictions and the vision data received during the "update" phase.
            Unit is degrees. Conversion to radians is done internally.
            Defaults to 5 (but should be adjusted based on real-world conditions).
    """

    def __init__(self, id: int=0, noise_xy_sd: float=0.01, noise_th_sd_deg: float=5):
        assert noise_xy_sd > 0, "The standard deviation must be greater than 0"
        assert noise_th_sd_deg > 0, "The standard deviation must be greater than 0"
        
        self.id = id
        
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
        self.measurement_cov_xy = np.array([[var_x, covariance_xy],
                                            [covariance_xy, var_y]])
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
        

    def _step_xy(self, new_data: tuple[float], last_robot: Robot, time_elapsed: float) -> tuple[float]:
        """
        A single iteration of the filter for x and y coordinates.
        
        Args:
            new_data (tuple[float]): New vision data received (x coordinates in metres, y coordinates in metres),
                passed by the externally callable function filter_data.
            last_robot (Robot): An object storing the robot's last known position and velocity, among others.
            time_elapsed (float): Time since last vision data was received.
        
        Returns:
            tuple[float]: Filtered vision data (x coordinates, y coordinates),
                returned to the externally callable function filter_data for packaging.
        
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
        if new_data[0] is not None:  # Received frame.
            # z
            measurement_xy = np.array(new_data)
            
            # K_n
            kalman_gain_xy = np.matmul(pred_cov_xy, np.linalg.inv(pred_cov_xy + self.measurement_cov_xy))
            
            # s_n,n
            self.state_xy = pred_state_xy + np.matmul(kalman_gain_xy, (measurement_xy - pred_state_xy))
            
            ident_less_kalman_xy = self.identity_xy - kalman_gain_xy
            ident_less_kalman_xy_T = np.transpose(ident_less_kalman_xy)
            measurement_uncertainty_xy = np.matmul(
                kalman_gain_xy, np.matmul(self.measurement_cov_xy, np.transpose(kalman_gain_xy))
            )
            
            # P_n,n
            self.covariance_mat_xy = np.matmul(
                ident_less_kalman_xy, np.matmul(pred_cov_xy, ident_less_kalman_xy_T)
            ) + measurement_uncertainty_xy
        
        # We can rely on the invariant that vanished frames have null x values
        # as they are imputed with a null VisionRobotData in the Position Refiner.
        else:  # Vanished frame: use predicted values.
            self.state_xy = pred_state_xy
            self.covariance_mat_xy = pred_cov_xy

        return tuple(self.state_xy)


    def _step_th(self, new_data: float, last_th: float) -> float:
        """
        A single iteration of the filter for orientation.
        
        Args:
            new_data (float): New vision data received (orientation in radians),
                passed by the externally callable function filter_data.
            last_th (float): The robot's last known orientation
        
        Returns:
            float: Filtered vision data orientation,
                returned to the externally callable function filter_data for packaging.
        
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
            measurement = normalise_heading(new_data)
            
            # K_n
            kalman_gain = pred_cov_th / (pred_cov_th + self.measurement_cov_th)
            
            # Taking a circular weighted average
            weights = (kalman_gain, 1 - kalman_gain)
            values  = (measurement, self.state_th)
            sines   = np.dot(weights, np.sin(values))
            cosines = np.dot(weights, np.cos(values))
            # s_n,n; already wrapped to (-pi, pi] as we're taking a circular average
            self.state_th = float(np.arctan2(sines, cosines))
            
            # P_n,n
            self.covariance_th = (1 - kalman_gain) * pred_cov_th
        
        # We can rely on the invariant that vanished frames have null x values
        # as they are imputed with a null VisionRobotData in the Position Refiner.
        else:  # Vanished frame: use predicted values.
            # self.state_th is unchanged
            self.covariance_th = pred_cov_th

        return self.state_th
    

    @staticmethod
    def filter_data(filter, data: VisionRobotData, last_frame: dict[int, Robot], time_elapsed: float) -> VisionRobotData:
        """
        An externally-callable function for the position refiner to pass data into the filter.
        
        Args:
            data (VisionRobotData): An object representing the new vision data
                received (x coordinates in metres, y coordinates in metres, orientation in radians).
            last_frame (Dict[int, Robot]): An object storing the last known
                position and velocity of all robots, among others.
            time_elapsed (float): Time since last vision data was received.
        
        Returns:
            VisionRobotData: Filtered vision data x coordinates, y coordinates, orientation),
                re-packaged into the object representing vision data.
        
        """
        
        # class VisionRobotData: id: int; x: float; y: float; orientation: float
        x_f, y_f = filter._step_xy((data.x, data.y), last_frame[filter.id], time_elapsed)
        th_f = filter._step_th(data.orientation, last_frame[filter.id].orientation)

        return VisionRobotData(data.id, x_f, y_f, th_f)
    
    
class Kalman_filter_ball():
    """
    Kalman filter for 3D position of ball.
    
    See above for details about the methodology.

    Args:
        noise_sd (float): A hyper-parameter, used to weigh the filter's
            predictions and the vision data received during the "update" phase.
            Unit is metres.
            Defaults to 0.01 (but should be adjusted based on real-world conditions).
    """

    def __init__(self, noise_sd: float=0.01):
        assert noise_sd > 0, "The standard deviation must be greater than 0"
        
        # s; to be initialised by strategy runner with 1st GameFrame
        self.state = None
        
        # sigma squared x, y, z
        noise_var = pow(noise_sd, 2)
        var_x, var_y, var_z = noise_var, noise_var, noise_var
        
        # sigma xy, xz, yz
        noise_covariance = 0  # assume their errors are uncorrelated
        covariance_xy, covariance_xz, covariance_yz = noise_covariance, noise_covariance, noise_covariance
        
        dimensions = 3
        self.identity = np.identity(dimensions)
        
        # R_n
        self.measurement_cov = np.array([[var_x, covariance_xy, covariance_xz],
                                         [covariance_xy, var_y, covariance_yz],
                                         [covariance_xz, covariance_yz, var_z]])
        # P_n,n; initialised with uncertainty in 1st frame
        self.covariance_mat = self.measurement_cov
        # Q
        self.process_noise = (2 * noise_var) * self.identity
        
        # Observation matrix H and state transition matrix F are just the identity matrix.
        # Multiplications with them are omitted.


    def _step(self, new_data: tuple[float], last_ball: Ball, time_elapsed: float) -> tuple[float]:
        """
        A single iteration of the filter.
        
        Args:
            new_data (tuple[float]): New vision data received (xyz coordinates in metres),
                passed by the externally callable function filter_data.
            last_ball (Ball): An object storing the ball's last known position and velocity.
            time_elapsed (float): Time since last vision data was received.
        
        Returns:
            tuple[float]: Filtered vision data (xyz coordinates),
                returned to the externally callable function filter_data for packaging.
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
            kalman_gain = np.matmul(pred_cov, np.linalg.inv(pred_cov + self.measurement_cov))
            
            # s_n,n
            self.state = pred_state + np.matmul(kalman_gain, (measurement - pred_state))
            
            ident_less_kalman = self.identity - kalman_gain
            ident_less_kalman_T = np.transpose(ident_less_kalman)
            measurement_uncertainty = np.matmul(
                kalman_gain, np.matmul(self.measurement_cov, np.transpose(kalman_gain))
            )
            
            # P_n,n
            self.covariance_mat_xy = np.matmul(
                ident_less_kalman, np.matmul(pred_cov, ident_less_kalman_T)
            ) + measurement_uncertainty
        
        # We can rely on the invariant that vanished frames have null x values
        # as they are imputed with None by filter_data
        else:  # Vanished frame: use predicted values.
            self.state = pred_state
            self.covariance_mat = pred_cov

        return tuple(self.state)


    @staticmethod
    def filter_data(filter, data: Ball, last_frame: Ball, time_elapsed: float) -> Ball:
        """
        An externally-callable function for the position refiner to pass data into the filter.
        
        Args:
            data (Ball): An object representing the new vision data received (xyz coordinates in metres).
            last_frame (Ball): An object storing the last known positon and velocity of the ball, among others.
            time_elapsed (float): Time since last vision data was received.
        
        Returns:
            VisionRobotData: Filtered vision data x coordinates, y coordinates, orientation),
                re-packaged into the object representing vision data.
        
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
                    
        filtered_data = filter._step(new_data, last_frame, time_elapsed)

        return Ball(Vector3D(*filtered_data), velocity, acceleration)