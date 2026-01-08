from collections import deque

import numpy as np
from scipy.signal import firwin

from utama_core.entities.data.vision import VisionRobotData


class FIR_filter:
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
        Length for boxcar if `taps` is None. Default 20.
    """

    def __init__(self, fs=60.0, taps=None, window_len=5, cutoff=None):
        self._fs = float(fs)
        self._N = window_len
        self._nyquist = 0.4 * self._fs

        if cutoff and cutoff < self._nyquist:
            self._cutoff = cutoff
        else:
            """
            Sets cutoff frequency according to the maximum acceleration and
            velocity of robots, below the limits dictated by Nyquist's theorem.
            """
            a_max = 50
            v_max = 5
            fc = a_max / (2 * np.pi * v_max)

            self._cutoff = min(self._nyquist, fc)

        if taps is None:
            assert window_len >= 1, "window_len must be >= 1"
            self._taps = firwin(window_len, self._cutoff, fs=fs)

        else:
            t = np.asarray(taps, dtype=float).ravel()
            assert t.size >= 1, "taps must have at least 1 element"
            # Normalize taps to sum to 1 for unity DC gain
            self._taps = t / np.sum(t)
            self._N = self._taps.size

        self._buf_x = deque(maxlen=self._N)
        self._buf_y = deque(maxlen=self._N)
        self._buf_th = deque(maxlen=self._N)

    def step(self, z):
        """
        Push a new measurement and return filtered output.
        z = [x, y, theta] (theta in radians).
        Returns: (x_filt, y_filt, theta_filt)
        """
        x, y, theta = map(float, z)
        # theta = normalise_heading(theta)

        self._buf_x.append(x)
        self._buf_y.append(y)
        self._buf_th.append(theta)

        # Use only the available samples during warm-up
        k = len(self._buf_x)  # same as len(buf_y) and len(buf_th)
        taps_eff = self._taps[-k:]
        taps_eff = taps_eff / np.sum(taps_eff)  # renormalize

        # Position FIR
        x_arr = np.asarray(self._buf_x, dtype=float)
        y_arr = np.asarray(self._buf_y, dtype=float)
        x_f = float(np.dot(taps_eff, x_arr))
        y_f = float(np.dot(taps_eff, y_arr))

        # Orientation FIR via circular averaging - currently disabled
        # th_arr = np.asarray(self._buf_th, dtype=float)
        # s = np.dot(taps_eff, np.sin(th_arr))
        # c = np.dot(taps_eff, np.cos(th_arr))
        # th_f = float(np.arctan2(s, c))  # already wrapped to (-pi, pi]

        return x_f, y_f, theta

    @staticmethod
    def filter_robot(filter, data: VisionRobotData) -> VisionRobotData:
        # class VisionRobotData: id: int; x: float; y: float; orientation: float
        (x_f, y_f, th_f) = filter.step([data.x, data.y, data.orientation])

        return VisionRobotData(id=data.id, x=x_f, y=y_f, orientation=th_f)
