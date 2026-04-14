from dataclasses import dataclass


@dataclass(frozen=True)
class RsimGaussianNoise:
    """
    When running in rsim, add Gaussian noise to balls and robots with the given standard deviation.

    Args:
        x_stddev (float): Gaussian noise standard deviation for x values (in m). Defaults to 0.
        y_stddev (float): Gaussian noise standard deviation for y values (in m). Defaults to 0.
        th_stddev_deg (float): Gaussian noise standard deviation for orientation values (in degrees). Defaults to 0.
    """

    x_stddev: float = 0
    y_stddev: float = 0
    th_stddev_deg: float = 0

    def __post_init__(self):
        assert self.x_stddev >= 0
        assert self.y_stddev >= 0
        assert self.th_stddev_deg >= 0
