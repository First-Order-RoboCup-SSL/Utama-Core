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

    def __init__(self, id, noise=5):
        self.id = id
        
        self.state = None  # s; to be initialised by strategy runner with 1st GameFrame
        self.var_x = noise  # sigma squared x; assume standard deviation of 10 cm
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

    def step(self, new_data: list[float], last_robot: Robot, time_elapsed: float) -> tuple[float]:
        """
        Push a new measurement and return filtered output.
        """
        # class Robot: id: int; is_friendly: bool; has_ball: bool
        # p: Vector2D; v: Vector2D; a: Vector2D; orientation: float
        
        # TODO: Vanishing
        x, y, theta = new_data
        # theta = normalise_heading(theta)

        if self.state is None:  # Initialised with the 1st GameFrame
            self.state = np.array([last_robot.p.x, last_robot.p.y])
            
        measurement = np.array([x, y])  # z
        control_velocities = np.array([last_robot.v.x, last_robot.v.y])  # u
        control_mat = time_elapsed * self.identity  # G
        
        # Phase 1: Predicting the current state given the last state.
        pred_state = self.state + np.matmul(control_mat, control_velocities)  # s_n,n-1
        pred_cov = self.covariance_mat + self.process_noise  # P_n,n-1
        
        # Phase 2: Adjust this prediction based on new data
        kalman_gain = np.matmul(pred_cov, np.linalg.inv(pred_cov + self.measurement_cov))  # K_n
        
        self.state = pred_state + np.matmul(kalman_gain, (measurement - pred_state))  # s_n,n
        
        ident_less_kalman = self.identity - kalman_gain
        ident_less_kalman_T = np.transpose(ident_less_kalman)
        measurement_uncertainty = np.matmul(kalman_gain,
                                            np.matmul(self.measurement_cov, np.transpose(kalman_gain)))
        self.covariance_mat = np.matmul(ident_less_kalman,
                                        np.matmul(pred_cov, ident_less_kalman_T)) + measurement_uncertainty  # P_n,n

        return self.state[0], self.state[1], theta


    @staticmethod
    def filter_robot(
        filter,
        data: VisionRobotData,
        last_frame: Dict[int, Robot],
        time_elapsed: float,
    ) -> VisionRobotData:
        
        # class VisionRobotData: id: int; x: float; y: float; orientation: float
        (x_f, y_f, th_f) = filter.step([data.x, data.y, data.orientation],
                                       last_frame[filter.id],
                                       time_elapsed)

        return VisionRobotData(id=data.id, x=x_f, y=y_f, orientation=th_f)


class Kalman_filter_2:
    def __init__(self, dt=1/60):
        self.dt = dt

        # [x, y, theta, v, omega]
        self.x = np.zeros((5, 1))

        self.P = np.eye(5) * 5.0

        # [x, y, theta, v, omega]
        self.Q = np.diag([1e-2, 1e-2, 1, 1e-2, 1e-1])

        self.R = np.diag([0.1, 0.1, 0.001])

        self.I = np.eye(5)

    def predict(self):
        theta = self.x[2, 0]
        v = self.x[3, 0]
        omega = self.x[4, 0]

        self.x[0, 0] += v * np.cos(theta) * self.dt
        self.x[1, 0] += v * np.sin(theta) * self.dt
        self.x[2, 0] += omega * self.dt
        self.x[2, 0] = self._wrap_angle(self.x[2, 0])

        F = np.eye(5)
        F[0, 2] = -v * np.sin(theta) * self.dt
        F[0, 3] = np.cos(theta) * self.dt
        F[1, 2] = v * np.cos(theta) * self.dt
        F[1, 3] = np.sin(theta) * self.dt
        F[2, 4] = self.dt

        self.P = F @ self.P @ F.T + self.Q

    def update(self, z_x, z_y, z_theta):
        z = np.array([[z_x], [z_y], [z_theta]])
        h_x = self.x[:3]
        y = z - h_x
        y[2, 0] = self._wrap_angle(y[2, 0])

        H = np.zeros((3, 5))
        H[0, 0] = 1.0 # x measurement
        H[1, 1] = 1.0 # y measurement
        H[2, 2] = 1.0 # theta measurement

        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + (K @ y)
        self.P = (self.I - (K @ H)) @ self.P

        self.x[2, 0] = self._wrap_angle(self.x[2, 0])

    def _wrap_angle(self, angle):
        """Helper to keep angles between -pi and pi"""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def step(self, dfSeries):
        z_x, z_y, z_theta = dfSeries
        self.predict()
        self.update(z_x, z_y, z_theta)
        return self.x[:3, 0]
    

    @staticmethod
    def filter_robot(
        filter,
        data: VisionRobotData,
    ) -> VisionRobotData:
        
        # class VisionRobotData: id: int; x: float; y: float; orientation: float
        (x_f, y_f, th_f) = filter.step([data.x, data.y, data.orientation])

        return VisionRobotData(id=data.id, x=x_f, y=y_f, orientation=th_f)


class Kalman_filter_3:
    def __init__(self, dt=1/60):
        self.dt = dt
        self.x = np.array([[0.0],
                           [0.0]])
        self.P = np.eye(2)
        self.F = np.array([[1, self.dt],
                           [0,1]])
        self.H = np.array([[1.0, 0.0]])
        self.Q = np.array([[1e-2, 0.0],
                           [0.0, 1e-2]])

        self.R = np.array([[0.05]])
        self.I = np.eye(2)

    def predict(self):
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, z): # z observing value
        self.S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        self.K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(self.S))
        self.x = self.x + np.dot(self.K, z - np.dot(self.H, self.x))
        self.P = np.dot(self.I - np.dot(self.K, self.H), self.P)

    def step(self, z):
        self.predict()
        self.update(z)
        return self.x[0, 0]
    

class Kalman_filter_2D:
    def __init__(self, dt=1/60):
        self.kalmanY = Kalman_filter_3(dt)
        self.kalmanX = Kalman_filter_3(dt)

    def step(self, x, y):
        return self.kalmanX.step(x), self.kalmanY.step(y)

    @staticmethod
    def filter_robot(
        filter,
        data: VisionRobotData,
    ) -> VisionRobotData:
        
        # class VisionRobotData: id: int; x: float; y: float; orientation: float
        x_f, y_f = filter.step(data.x, data.y)

        return VisionRobotData(id=data.id, x=x_f, y=y_f, orientation=data.orientation)