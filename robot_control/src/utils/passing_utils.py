from entities.data.vision import RobotData, BallData
from typing import Tuple, List
import numpy as np


def calculate_ttr():
    # TODO: calculate time to ready to prevent unnecessary waiting of the robots
    pass


def calculate_adjusted_receiver_pos(
    receiver_data: RobotData,
    ball_traj_points: List[Tuple[float, float]],
) -> Tuple[float, float]:
    """
    Returns the adjusted receive position of the receiver based on ball trajectory

    where ball trajectory is plotted with ax + by + c = 0

    Uses the formula of x = x1 - a * (a * x1 + b * y1 + c) / (a^2 + b^2)
                    and y = y1 - b * (a * x1 + b * y1 + c) / (a^2 + b^2)
    """

    def get_ball_movement_line(
        ball_traj_points: List[Tuple[float, float]]
    ) -> Tuple[float, float, float]:
        """
        Returns the line equation ax + by + c = 0 of the ball's movement line
        """
        points = np.array(ball_traj_points)
        x = points[:, 0]
        y = points[:, 1]
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]

        # Convert to standard form ax + by + c = 0
        a = m
        b = -1
        c = c

        return a, b, c

    x1 = receiver_data.x
    y1 = receiver_data.y

    a, b, c = get_ball_movement_line(ball_traj_points)

    denominator = a**2 + b**2
    assert denominator != 0, "Denominator is zero"

    numerator = a * x1 + b * y1 + c

    x = x1 - a * numerator / denominator
    y = y1 - b * numerator / denominator

    return (x, y)
