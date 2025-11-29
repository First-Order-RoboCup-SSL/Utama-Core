import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque

# class KalmanCVOrientation:
#     """
#     Real-time Kalman filter for 2D position + orientation.
#     State: [x, y, vx, vy, theta, omega]
#     Measurement: [x, y, theta]
#     Model: constant velocity for x,y and constant angular velocity for theta.
#     """
#
#     def __init__(self, dt=1/60,
#                  sigma_a=1.5,           # linear acceleration noise (units/s^2)
#                  sigma_alpha=0.8,       # angular acceleration noise (rad/s^2)
#                  sigma_x=0.02, sigma_y=0.02, sigma_theta=0.05,  # measurement noise (units, units, rad)
#                  x0=None, P0=None,
#                  mahalanobis_thresh=None # e.g., 9.21 ~ chi2(3, 0.99) for 3D measurement
#                  ):
#         self.dt = float(dt)
#
#         # System matrices
#         self.F = np.array([
#             [1, 0, dt, 0, 0, 0],
#             [0, 1, 0, dt, 0, 0],
#             [0, 0, 1,  0, 0, 0],
#             [0, 0, 0,  1, 0, 0],
#             [0, 0, 0,  0, 1, dt],
#             [0, 0, 0,  0, 0, 1]
#         ], dtype=float)
#
#         self.H = np.array([
#             [1, 0, 0, 0, 0, 0],  # x
#             [0, 1, 0, 0, 0, 0],  # y
#             [0, 0, 0, 0, 1, 0]   # theta
#         ], dtype=float)
#
#         # Process noise Q (block-diagonal: translation + orientation)
#         Q_xy = (sigma_a**2) * np.array([
#             [dt**4/4, 0,        dt**3/2, 0       ],
#             [0,        dt**4/4, 0,        dt**3/2],
#             [dt**3/2,  0,        dt**2,   0       ],
#             [0,        dt**3/2,  0,        dt**2  ]
#         ], dtype=float)
#
#         Q_th = (sigma_alpha**2) * np.array([
#             [dt**4/4, dt**3/2],
#             [dt**3/2, dt**2  ]
#         ], dtype=float)
#
#         self.Q = np.zeros((6, 6), dtype=float)
#         self.Q[:4, :4] = Q_xy
#         self.Q[4:, 4:] = Q_th
#
#         # Measurement noise R
#         self.R = np.diag([sigma_x**2, sigma_y**2, sigma_theta**2]).astype(float)
#
#         # State and covariance initialization
#         if x0 is None:
#             self.x = np.zeros((6, 1), dtype=float)  # [x, y, vx, vy, theta, omega]
#         else:
#             x0 = np.asarray(x0, dtype=float).reshape(6, 1)
#             self.x = x0
#
#         if P0 is None:
#             self.P = np.eye(6, dtype=float) * 1e2
#         else:
#             P0 = np.asarray(P0, dtype=float).reshape(6, 6)
#             self.P = P0
#
#         # Optional gating threshold on Mahalanobis distance squared
#         self.mahalanobis_thresh = mahalanobis_thresh
#
#     @staticmethod
#     def wrap_angle(a):
#         """Wrap angle to (-pi, pi]."""
#         return (a + np.pi) % (2 * np.pi) - np.pi
#
#     @staticmethod
#     def angle_diff(a, b):
#         """Shortest angle difference from b to a."""
#         return ((a - b + np.pi) % (2 * np.pi)) - np.pi
#
#     def predict(self):
#         """Kalman predict step."""
#         # Predict state
#         self.x = self.F @ self.x
#         # Wrap theta
#         self.x[4, 0] = self.wrap_angle(self.x[4, 0])
#         # Predict covariance
#         self.P = self.F @ self.P @ self.F.T + self.Q
#
#     def update(self, z):
#         """
#         Kalman update step with measurement z = [x, y, theta].
#         Returns:
#             accepted (bool): whether the update was applied (gating might reject).
#             innov (3x1), S (3x3), K (6x3): innovation, innovation covariance, and Kalman gain.
#         """
#         z = np.asarray(z, dtype=float).reshape(3, 1)
#
#         # Predicted measurement
#         z_pred = self.H @ self.x
#
#         # Innovation (with angle wrapping for theta)
#         innov = z.copy()
#         innov[0:2] -= z_pred[0:2]
#         innov[2, 0] = self.angle_diff(z[2, 0], z_pred[2, 0])
#
#         # Innovation covariance
#         S = self.H @ self.P @ self.H.T + self.R
#
#         # Optional innovation gating (Mahalanobis distance)
#         accepted = True
#         if self.mahalanobis_thresh is not None:
#             try:
#                 S_inv = np.linalg.inv(S)
#             except np.linalg.LinAlgError:
#                 # Regularize S if singular
#                 S_inv = np.linalg.inv(S + 1e-9 * np.eye(S.shape[0]))
#             d2 = float(innov.T @ S_inv @ innov)
#             if d2 > self.mahalanobis_thresh:
#                 accepted = False
#
#         if accepted:
#             # Kalman gain
#             K = self.P @ self.H.T @ np.linalg.inv(S)
#             # State update
#             self.x = self.x + K @ innov
#             self.x[4, 0] = self.wrap_angle(self.x[4, 0])
#             # Covariance update (Joseph form can be used for numerical stability)
#             I = np.eye(self.P.shape[0])
#             self.P = (I - K @ self.H) @ self.P
#         else:
#             K = np.zeros((6, 3), dtype=float)
#
#         return accepted, innov, S, K
#
#     def step(self, z=None):
#         """
#         Convenience method: predict and (optionally) update.
#         If z is None, performs only predict (useful for dropouts).
#         Returns:
#             state (np.ndarray): current state as (6,)
#         """
#         self.predict()
#         if z is not None:
#             self.update(z)
#         return self.state_vector()
#
#     def state_vector(self):
#         """Return state as a flat array: [x, y, vx, vy, theta, omega]."""
#         return self.x.ravel().copy()
#
#     def state_dict(self):
#         """Return state as a dictionary."""
#         s = self.state_vector()
#         return {
#             "x": s[0], "y": s[1],
#             "vx": s[2], "vy": s[3],
#             "theta": s[4], "omega": s[5]
#         }
#
#     def set_measurement_noise(self, sigma_x=None, sigma_y=None, sigma_theta=None):
#         """Update measurement noise R."""
#         sx2 = self.R[0, 0] if sigma_x is None else sigma_x**2
#         sy2 = self.R[1, 1] if sigma_y is None else sigma_y**2
#         sth2 = self.R[2, 2] if sigma_theta is None else sigma_theta**2
#         self.R = np.diag([sx2, sy2, sth2]).astype(float)
#
#     def scale_R_by_confidence(self, conf, min_conf=0.2, max_scale=10.0):
#         """
#         Inflate R when measurement confidence is low.
#         conf in [0,1]. When conf=min_conf, scale ~ max_scale; when conf=1, scale ~ 1.
#         """
#         conf = float(np.clip(conf, min_conf, 1.0))
#         scale = 1.0 + (1.0 - conf) * (max_scale - 1.0) / (1.0 - min_conf)
#         self.R = self.R * scale
#
#     def reset(self, x0=None, P0=None):
#         """Reset state and covariance."""
#         if x0 is not None:
#             x0 = np.asarray(x0, dtype=float).reshape(6, 1)
#             self.x = x0
#         else:
#             self.x = np.zeros((6, 1), dtype=float)
#         if P0 is not None:
#             P0 = np.asarray(P0, dtype=float).reshape(6, 6)
#             self.P = P0
#         else:
#             self.P = np.eye(6, dtype=float) * 1e2
#         self.x[4, 0] = self.wrap_angle(self.x[4, 0])
#
#     def set_mahalanobis_threshold(self, d2_thresh):
#         """Set gating threshold on Mahalanobis distance squared."""
#         self.mahalanobis_thresh = float(d2_thresh)

class FIRPosOrientation:
    """
    Finite Impulse Response (FIR) filter for 2D position + orientation.
    - Streams [x, y, theta] measurements at a fixed sampling rate.
    - Position is filtered with linear FIR taps.
    - Orientation is filtered via circular (vector) averaging using the same taps:
        theta_hat = atan2(sum(w_i*sin(theta_i)), sum(w_i*cos(theta_i)))

    Parameters
    ----------
    fs : float
        Sampling rate (Hz). Default 60.0.
    taps : array-like or None
        FIR taps. If None, a boxcar of length `window_len` is used.
    window_len : int
        Length for boxcar if `taps` is None. Default 7.
    """

    def __init__(self, fs=60.0, taps=None, window_len=7):
        self.fs = float(fs)

        if taps is None:
            assert window_len >= 1, "window_len must be >= 1"
            self.taps = np.ones(window_len, dtype=float) / float(window_len)
        else:
            t = np.asarray(taps, dtype=float).ravel()
            assert t.size >= 1, "taps must have at least 1 element"
            # Normalize taps to sum to 1 for unity DC gain
            self.taps = t / np.sum(t)

        self.N = self.taps.size
        self.buf_x = deque(maxlen=self.N)
        self.buf_y = deque(maxlen=self.N)
        self.buf_th = deque(maxlen=self.N)

    @property
    def group_delay_frames(self):
        """Group delay in frames: D = (N-1)/2."""
        return (self.N - 1) / 2.0

    @property
    def group_delay_seconds(self):
        """Group delay in seconds: D/fs."""
        return self.group_delay_frames / self.fs

    @property
    def approx_boxcar_fc(self):
        """
        Approximate -3 dB cutoff for a boxcar of length N:
        fc ≈ 0.443 * fs / N.
        For non-boxcar taps, this is just a reference.
        """
        return 0.443 * self.fs / self.N

    @staticmethod
    def wrap_angle(a):
        """Wrap angle to (-pi, pi]."""
        return (a + np.pi) % (2 * np.pi) - np.pi

    def reset(self):
        """Clear buffers."""
        self.buf_x.clear()
        self.buf_y.clear()
        self.buf_th.clear()

    def set_taps(self, taps):
        """Update FIR taps (normalized to sum to 1) and reset buffers."""
        t = np.asarray(taps, dtype=float).ravel()
        assert t.size >= 1, "taps must have at least 1 element"
        self.taps = t / np.sum(t)
        self.N = self.taps.size
        self.buf_x = deque(maxlen=self.N)
        self.buf_y = deque(maxlen=self.N)
        self.buf_th = deque(maxlen=self.N)

    def set_boxcar(self, window_len):
        """Switch to a boxcar (moving average) of given length."""
        assert window_len >= 1, "window_len must be >= 1"
        self.set_taps(np.ones(window_len, dtype=float))

    def step(self, z):
        """
        Push a new measurement and return filtered output.
        z = [x, y, theta] (theta in radians).
        Returns: (x_filt, y_filt, theta_filt)
        """
        x, y, theta = map(float, z)
        theta = self.wrap_angle(theta)

        self.buf_x.append(x)
        self.buf_y.append(y)
        self.buf_th.append(theta)

        # Use only the available samples during warm-up
        k = len(self.buf_x)  # same as len(buf_y) and len(buf_th)
        taps_eff = self.taps[-k:]
        taps_eff = taps_eff / np.sum(taps_eff)  # renormalize

        # Position FIR
        x_arr = np.asarray(self.buf_x, dtype=float)
        y_arr = np.asarray(self.buf_y, dtype=float)
        x_f = float(np.dot(taps_eff, x_arr))
        y_f = float(np.dot(taps_eff, y_arr))

        # Orientation FIR via circular averaging
        th_arr = np.asarray(self.buf_th, dtype=float)
        s = np.dot(taps_eff, np.sin(th_arr))
        c = np.dot(taps_eff, np.cos(th_arr))
        th_f = float(np.arctan2(s, c))  # already wrapped to (-pi, pi]

        return x_f, y_f, th_f

def createStreams(filename): #for robot zero
    streams = []
    data = pd.read_csv(filename)
    data = data[data["camera_id"] == 1]
    data = data[data["id"] == 0]
    data = data[["x", "y", "orientation"]]
    return data.to_numpy()

def toPlot(noise_data, real_data):
    df1 = pd.DataFrame(data=noise_data, columns=["x", "y", "orientation"])
    df1 = df1.assign(TYPE="noise")
    df3 = pd.DataFrame(data=real_data, columns=["x", "y", "orientation"])
    df3 = df3.assign(TYPE="FIR")
    df4 = pd.concat([df1, df3])
    sns.lineplot(x="x", y="y", hue="TYPE", data=df4)
    plt.show()



# kf = KalmanCVOrientation(
#     dt=1/60,
#     sigma_a=1.5,
#     sigma_alpha=0.8,
#     sigma_x=0.02, sigma_y=0.02, sigma_theta=0.05,
#     mahalanobis_thresh=9.21  # Chi-square(3) 99% for [x,y,theta]
# )

fir = FIRPosOrientation(fs=60.0, window_len=7)

measurements_stream = createStreams("noise_all_5.csv")
print(measurements_stream.dtype)

# Streaming measurements: z = [x_meas, y_meas, theta_meas]
# kalman_stream = []
fir_stream = []
for z in measurements_stream:
    # Optional: scale measurement noise by confidence
    # kf.scale_R_by_confidence(confidence)

    # state = kf.step(z)  # predict + update
    # # If a frame is missing: state = kf.step(None)  # predict only
    #
    # # Use state: [x, y, vx, vy, theta, omega]
    # x, y, vx, vy, theta, omega = state
    # kalman_stream.append([x, y, vx, vy, theta, omega])

    x, y, theta = fir.step(z)
    fir_stream.append([x, y, theta])



# kalman_stream = np.array(kalman_stream)
fir_stream = np.array(fir_stream)
toPlot(measurements_stream, fir_stream)