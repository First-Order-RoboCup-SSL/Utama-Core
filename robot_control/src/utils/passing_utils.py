from entities.data.vision import RobotData, BallData
from typing import Tuple


def calculate_ttr():
    # TODO: calculate time to ready to prevent unnecessary waiting of the robots
    pass


def calculate_adjusted_receiver_pos(
    ball_initial_pos: Tuple[float, float],
    receiver_data: RobotData,
    ball_data: BallData,
) -> Tuple[float, float]:
    """
    Returns the adjusted receive position of the receiver based on ball trajectory

    where ball trajectory is plotted with ax + by + c = 0

    Uses the formula of x = x1 - a * (a * x1 + b * y1 + c) / (a^2 + b^2)
                    and y = y1 - b * (a * x1 + b * y1 + c) / (a^2 + b^2)
    """

    def get_ball_movement_line(
        ball_ini_pos: Tuple[float, float], ball_pos: Tuple[float, float]
    ) -> Tuple[float, float, float]:
        """
        Returns the line equation ax + by + c = 0 of the ball's movement line
        """
        vx = ball_pos[0] - ball_ini_pos[0]
        vy = ball_pos[1] - ball_ini_pos[1]
        b = -vx
        a = vy
        c = -a * ball_pos[0] - b * ball_pos[1]

        return a, b, c

    x1 = receiver_data.x
    y1 = receiver_data.y

    a, b, c = get_ball_movement_line(ball_initial_pos, (ball_data.x, ball_data.y))

    denominator = a**2 + b**2
    assert denominator != 0, "Denominator is zero"

    numerator = a * x1 + b * y1 + c

    x = x1 - a * numerator / denominator
    y = y1 - b * numerator / denominator

    return (x, y)
