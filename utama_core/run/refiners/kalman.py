from collections import deque
import numpy as np
import sys
from typing import Dict

# For running analytics from Jupyter notebook
try:
    from utama_core.entities.data.vision import VisionRobotData
    from utama_core.entities.game import GameFrame, Robot
except ModuleNotFoundError:
    sys.path.append("../utama_core/entities/")
    from data.vision import VisionRobotData
    from game import GameFrame, Robot
    

class Kalman_filter:
    """
    Kalman filter for 2D position + orientation.
    - Streams [x, y, theta] measurements at a fixed sampling rate.
    - Position is filtered with linear FIR taps.
    - Orientation is filtered via circular (vector) averaging using the same taps:
        theta_hat = atan2(sum(w_i*sin(theta_i)), sum(w_i*cos(theta_i)))

    Parameters
    ----------
    fs : float
        Sampling rate (Hz). Default 60.0.
    """

    def __init__(self, id, noise_sd=0.01):
        self.id = id
        
        self.state = None  # s; to be initialised by strategy runner with 1st GameFrame
        self.var_x = noise_sd  # sigma squared x
        self.var_y = self.var_x  # sigma squared y
        self.covariance_xy = 0  # sigma xy; assume their errors are uncorrelated
        self.dimensions = 2
        self.identity = np.identity(2)
        
        self.measurement_cov = np.array([[self.var_x, self.covariance_xy],
                                         [self.covariance_xy, self.var_y]])  # R_n
        self.covariance_mat = self.measurement_cov  # P_n,n; initialised with uncertainty in 1st frame
        self.process_noise = (2 * self.var_x) * self.identity  # Q
        # Observation matrix H and state transition matrix F are just the identity matrix.
        # Multiplications with them are omitted.


    def step(self, new_data: tuple[float], last_robot: Robot, time_elapsed: float) -> tuple[float]:
        """
        Push a new measurement and return filtered output.
        """
        # class Robot: id: int; is_friendly: bool; has_ball: bool
        # p: Vector2D; v: Vector2D; a: Vector2D; orientation: float
        
        # Phase 0: Initialised with the 1st valid GameFrame (only on initialisation)
        if self.state is None:
            self.state = np.array([last_robot.p.x, last_robot.p.y])
            
        # Phase 1: Predicting the current state given the last state.
        control_velocities = np.array([last_robot.v.x, last_robot.v.y])  # u
        control_mat = time_elapsed * self.identity  # G
        
        pred_state = self.state + np.matmul(control_mat, control_velocities)  # s_n,n-1
        pred_cov = self.covariance_mat + self.process_noise  # P_n,n-1
        
        # Phase 2: Adjust this prediction based on new data
        
        x, y, theta = new_data
        # theta = normalise_heading(theta)
        
        if x is not None:  # Received frame.
            measurement = np.array([x, y])  # z
            
            kalman_gain = np.matmul(
                pred_cov, np.linalg.inv(pred_cov + self.measurement_cov)
            )  # K_n
            
            self.state = pred_state + np.matmul(
                kalman_gain, (measurement - pred_state)
            )  # s_n,n
            
            ident_less_kalman = self.identity - kalman_gain
            ident_less_kalman_T = np.transpose(ident_less_kalman)
            measurement_uncertainty = np.matmul(
                kalman_gain, np.matmul(
                    self.measurement_cov,
                    np.transpose(kalman_gain)
                )
            )
            
            self.covariance_mat = np.matmul(
                ident_less_kalman,
                np.matmul(pred_cov, ident_less_kalman_T)
            ) + measurement_uncertainty  # P_n,n
        
        # We can rely on the invariant that vanished frames have null x values
        # as they are imputed with a null VisionRobotData
        else:  # Vanished frame: use predicted values.
            self.state = pred_state
            self.covariance_mat = pred_cov

        return self.state[0], self.state[1], theta


    @staticmethod
    def filter_robot(
        filter,
        data: VisionRobotData,
        last_frame: Dict[int, Robot],
        time_elapsed: float,
    ) -> VisionRobotData:
        
        # class VisionRobotData: id: int; x: float; y: float; orientation: float
        (x_f, y_f, th_f) = filter.step(
            (data.x, data.y, data.orientation),
            last_frame[filter.id],
            time_elapsed
        )

        return VisionRobotData(id=data.id, x=x_f, y=y_f, orientation=th_f)