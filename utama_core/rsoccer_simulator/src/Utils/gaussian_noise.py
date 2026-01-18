from dataclasses import dataclass


@dataclass(frozen=True)
class RsimGaussianNoise:
    x_stddev: float = 0
    y_stddev: float = 0
    th_stddev_deg: float = 0