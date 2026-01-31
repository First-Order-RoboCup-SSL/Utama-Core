from collections import deque
import numpy as np
from scipy.signal import firwin
from typing import Union

from utama_core.entities.data.vision import VisionRobotData


class FIR_filter:
    """
    Finite Impulse Response (FIR) filter for 2D position and orientation.
    This is essentially a weighted average of the past n data points.
    
    More about the methodology:
    https://youtube.com/playlist?list=PLbqhA-NKGP6Afr_KbPUuy_yIBpPR4jzWo&si=7l2BVnsN_jSKq2JL.

    Args:
        fs (float): Sampling rate (Hz). Defaults to 60.0.
        taps (array-like or None): FIR taps. If None, a series of taps of length `window_len` is created.
        window_len (int): Number of taps to be created if `taps` is None. Default 20.
        cutoff (float): Cutoff frequency of the filter.
    """

    def __init__(
        self,
        fs: float=60.0,
        taps: Union[np.array, None]=None,
        window_len: int=5,
        cutoff: Union[float, None]=None
    ):
        self._fs = float(fs)
        self._N = window_len
        self._nyquist = 0.4 * self._fs

        if cutoff and cutoff < self._nyquist:
            self._cutoff = cutoff
        else:
            # Sets cutoff frequency according to the maximum acceleration and
            # velocity of robots, below the limits dictated by Nyquist's theorem.
            
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

    def step(self, data: tuple[float]):
        """
        A single iteration of the filter
        
        Args: 
            data (tuple[float]): The new vision data received (x and y coordinates
                in metres, orientation in radians).
        
        Returns:
            tuple[float]: The filtered data
        """
        
        x, y, th = map(float, data)
        # theta = normalise_heading(theta)

        self._buf_x.append(x)
        self._buf_y.append(y)
        self._buf_th.append(th)

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

        return x_f, y_f, th

    @staticmethod
    def filter_robot(filter, data: VisionRobotData) -> VisionRobotData:
        """
        Externally callable function for Position Refiner to pass data
        to be filtered.

        Args:
            filter (FIR_filter): The filter associated with a robot
            data (VisionRobotData): The new vision data received (x and y
                coordinates in metres, orientation in radians).

        Returns:
            VisionRobotData: The filtered vision data.
        """
        
        # class VisionRobotData: id: int; x: float; y: float; orientation: float
        (x_f, y_f, th_f) = filter.step((data.x, data.y, data.orientation))

        return VisionRobotData(id=data.id, x=x_f, y=y_f, orientation=th_f)
