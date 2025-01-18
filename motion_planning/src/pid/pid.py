import numpy as np
from typing import Optional, Union, Tuple

from team_controller.src.config.settings import TIMESTEP


def get_pids(n_robots: int):
    pid_oren = PID(TIMESTEP, 4, -4, 4.5, 0.1, 0.045, num_robots=n_robots)
    pid_trans = TwoDPID(TIMESTEP, 2.5, -2.5, 4.5, 0.1, 0.0, num_robots=n_robots)
    return pid_oren, pid_trans


class PID:
    """
    A Proportional-Integral-Derivative (PID) controller for managing error corrections over time.

    Args:
        dt (float): Time step for each PID update; must be positive.
        max_output (float): Maximum output value that the controller can produce.
        min_output (float): Minimum output value that the controller can produce.
        Kp (float): Proportional gain.
        Kd (float): Derivative gain.
        Ki (float): Integral gain.
        num_robots (int): Number of robots being controlled; each has separate error tracking.

    Raises:
        ValueError: If `dt` is not greater than zero.
    """

    def __init__(
        self,
        dt: float,
        max_output: float,
        min_output: float,
        Kp: float,
        Kd: float,
        Ki: float,
        num_robots: int,
    ):
        if dt <= 0:
            raise ValueError("dt should be greater than zero")
        self.dt = dt

        self.max_output = max_output
        self.min_output = min_output

        self.Kp = Kp
        self.Kd = Kd
        self.Ki = Ki

        self.pre_errors = {i: 0.0 for i in range(num_robots)}
        self.integrals = {i: 0.0 for i in range(num_robots)}

    def calculate(
        self,
        target: float,
        current: float,
        robot_id: int,
        oren: bool = False,
        normalize_range: Optional[float] = None,
    ) -> float:
        """
        Compute the PID output to move a robot towards a target position.

        Args:
            target (float): Desired target value (e.g., target position).
            current (float): Current observed value (e.g., current position).
            robot_id (int): Unique identifier for the robot to apply this control to.
            oren (bool): Whether to adjust for angular orientation. Default is False.
            normalize_range (Optional[float]): If provided, scales the output by this range.

        Returns:
            float: The PID control output value, limited between `min_output` and `max_output`.

        This function calculates the PID output by computing proportional, integral, and derivative terms,
        with optional normalization in order to try and keep all the output values within a range. When
        `oren` is True, the error calculation is modified to handle angular differences. Clamping is
        applied to prevent integral wind-up and to limit the total output.
        """

        # Calculate error
        error = target - current

        # Adjust error if orientation (oren) is considered
        if oren:
            error = np.atan2(np.sin(error), np.cos(error))

        # Calculate PID output
        Pout = self.Kp * error if self.Kp != 0 else 0.0

        # Integral term with clamping
        if self.Ki != 0:
            self.integrals[robot_id] += error * self.dt
            self.integrals[robot_id] = max(
                min(self.integrals[robot_id], self.max_output / self.Ki),
                self.min_output / self.Ki,
            )
            Iout = self.Ki * self.integrals[robot_id]
        else:
            Iout = 0.0

        # Derivative term
        if self.Kd != 0:
            derivative = (error - self.pre_errors[robot_id]) / self.dt
            Dout = self.Kd * derivative
        else:
            Dout = 0.0

        # Total output with clamping
        output = Pout + Iout + Dout

        # Apply optional normalization
        if normalize_range is not None and normalize_range != 0:
            output /= normalize_range

        # apply clamping
        output = max(self.min_output, min(self.max_output, output))

        # Save error for next calculation
        self.pre_errors[robot_id] = error
        return output

    def reset(self, robot_id: int):
        self.pre_errors[robot_id] = 0.0
        self.integrals[robot_id] = 0.0


class TwoDPID:
    def __init__(
        self,
        dt: float,
        max_output: float,
        min_output: float,
        Kp: float,
        Kd: float,
        Ki: float,
        num_robots: int,
    ):
        self.dimX = PID(dt, max_output, min_output, Kp, Kd, Ki, num_robots)
        self.dimY = PID(dt, max_output, min_output, Kp, Kd, Ki, num_robots)

    def calculate(
        self, target: Tuple[float, float], current: Tuple[float, float], robot_id
    ):
        return self.dimX.calculate(
            target[0], current[0], robot_id, False, normalize_range=4.5
        ), self.dimY.calculate(target[1], current[1], robot_id, False, normalize_range=3)

    def reset(self, robot_id: int):
        self.dimX.reset(robot_id)
        self.dimY.reset(robot_id)
