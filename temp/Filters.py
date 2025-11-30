import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque

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

fir = FIRPosOrientation(fs=60.0, window_len=7)

measurements_stream = createStreams("noise_all_5.csv")
print(measurements_stream.dtype)

fir_stream = []
for z in measurements_stream:
    x, y, theta = fir.step(z)
    fir_stream.append([x, y, theta])

fir_stream = np.array(fir_stream)
toPlot(measurements_stream, fir_stream)