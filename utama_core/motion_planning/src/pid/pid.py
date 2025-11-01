import math
import time
from typing import Optional

from utama_core.config.settings import SENDING_DELAY
from utama_core.entities.data.vector import Vector2D
from utama_core.global_utils.math_utils import normalise_heading
from utama_core.motion_planning.src.pid.pid_abstract import AbstractPID


class PID(AbstractPID[float]):
    """A PID controller that control the Orientation of the robot.

    Args:
        dt (float): Time step for each update.
        max_output (Optional[float]): Maximum output value (None for no limit).
        min_output (Optional[float]): Minimum output value (None for no limit).
        Kp (float): Proportional gain.
        Kd (float): Derivative gain.
        Ki (float): Integral gain.
        integral_min (Optional[float]): Minimum allowed integral value.
        integral_max (Optional[float]): Maximum allowed integral value.

    Note:
        The delay used by the Smith predictor is internally set using the
        :data:`SENDING_DELAY` configuration value.
    """

    def __init__(
        self,
        dt: float,
        max_output: Optional[float],
        min_output: Optional[float],
        Kp: float,
        Kd: float,
        Ki: float,
        integral_min: Optional[float] = None,
        integral_max: Optional[float] = None,
    ):
        if dt <= 0:
            raise ValueError("dt should be greater than zero")
        self.dt = dt
        self.delay = SENDING_DELAY / 1000  # Convert to seconds

        self.max_output = max_output
        self.min_output = min_output

        self.Kp = Kp
        self.Kd = Kd
        self.Ki = Ki

        self.pre_errors = {i: 0.0 for i in range(6)}
        self.integrals = {i: 0.0 for i in range(6)}

        # Anti-windup limits
        self.integral_min = integral_min
        self.integral_max = integral_max

        self.prev_times = {i: 0.0 for i in range(6)}

        self.first_pass = {i: True for i in range(6)}

    def calculate(
        self,
        target: float,
        current: float,
        robot_id: int,
    ) -> float:
        """Compute the PID output to move a robot towards a target with delay compensation.

        The delay is compensated by predicting the current value using the derivative.
        """
        call_func_time = time.time()
        # Compute the basic (instantaneous) error
        raw_error = target - current
        # For angular measurements adjust error
        error = normalise_heading(raw_error)
        # For very small errors, return zero
        if abs(error) < 0.001:
            self.prev_times[robot_id] = call_func_time
            return 0.0

        # Compute time difference
        dt = self.dt  # Default
        if self.prev_times[robot_id] != 0:
            measured_dt = call_func_time - self.prev_times[robot_id]
            dt = measured_dt if measured_dt > 0 else self.dt

        # Compute derivative term using the previous stored error
        if not self.first_pass[robot_id]:
            derivative = (error - self.pre_errors[robot_id]) / dt
        else:
            derivative = 0.0
            self.first_pass[robot_id] = False

        # --- Delay Compensation (Smith predictor approach) ---
        # Predict the "current" value (what it will be after the delay)
        # Here we assume a simple linear extrapolation: x_predicted = current + derivative * delay
        if self.delay > 0:
            predicted_current = current + derivative * self.delay
            # Then the predicted error is:
            predicted_error = normalise_heading(target - predicted_current)
            # Optionally, you might replace 'error' with 'predicted_error' in the PID computation.
            effective_error = predicted_error
        else:
            effective_error = error

        # Proportional term
        Pout = self.Kp * effective_error if self.Kp != 0 else 0.0

        # Integral term with anti-windup using effective_error
        if self.Ki != 0:
            self.integrals[robot_id] += effective_error * self.dt
            if self.integral_max is not None:
                self.integrals[robot_id] = min(self.integrals[robot_id], self.integral_max)
            if self.integral_min is not None:
                self.integrals[robot_id] = max(self.integrals[robot_id], self.integral_min)
            Iout = self.Ki * self.integrals[robot_id]
        else:
            Iout = 0.0

        # Derivative term based on effective error (already computed above)
        Dout = self.Kd * derivative

        # Combine the PID outputs
        output = Pout + Iout + Dout

        # Clamp the output for consistency
        if self.max_output is not None:
            output = min(self.max_output, output)
        if self.min_output is not None:
            output = max(self.min_output, output)

        # Store the error and update the time for the next iteration
        self.pre_errors[robot_id] = error
        self.prev_times[robot_id] = call_func_time
        # print(f"oren PID: {robot_id}, current:{current}, target: {target}, error: {error}, output: {output}")
        return output

    def reset(self, robot_id: int):
        """Reset the error and integral for the specified robot."""
        self.pre_errors[robot_id] = 0.0
        self.integrals[robot_id] = 0.0
        self.first_pass[robot_id] = True


class TwoDPID(AbstractPID[Vector2D]):
    """A 2D PID controller that controls the X and Y dimensions and scales the resulting velocity vector to a maximum
    speed if needed."""

    def __init__(
        self,
        dt: float,
        max_velocity: float,
        Kp: float,
        Kd: float,
        Ki: float,
        integral_min: Optional[float] = None,
        integral_max: Optional[float] = None,
    ):
        if dt <= 0:
            raise ValueError("dt should be greater than zero")
        self.dt = dt
        self.delay = SENDING_DELAY / 1000  # Delay in seconds

        self.max_velocity = max_velocity

        self.Kp = Kp
        self.Kd = Kd
        self.Ki = Ki

        self.pre_errors = {i: 0.0 for i in range(6)}
        self.integrals = {i: 0.0 for i in range(6)}

        # Anti-windup limits
        self.integral_min = integral_min
        self.integral_max = integral_max

        self.prev_times = {i: 0.0 for i in range(6)}

        self.first_pass = {i: True for i in range(6)}

    def calculate(self, target: Vector2D, current: Vector2D, robot_id: int) -> Vector2D:
        call_func_time = time.time()

        dx = target[0] - current[0]
        dy = target[1] - current[1]

        error = math.hypot(dx, dy)

        if abs(error) < 3 / 1000:
            self.prev_times[robot_id] = call_func_time
            return Vector2D(0.0, 0.0)

        # Compute time difference
        dt = self.dt
        if self.prev_times[robot_id] != 0:
            measured_dt = call_func_time - self.prev_times[robot_id]
            # Use the measured dt if nonzero; otherwise fall back to self.dt.
            dt = measured_dt if measured_dt > 0 else self.dt

        # Compute derivative term using the previous stored error
        if not self.first_pass[robot_id]:
            derivative = (error - self.pre_errors[robot_id]) / dt
        else:
            derivative = 0.0
            self.first_pass[robot_id] = False

        # --- Delay Compensation (Smith predictor approach) ---
        # Then the predicted error is:
        if self.delay > 0:
            predicted_error = error + derivative * self.delay
            # Optionally, you might replace 'error' with 'predicted_error' in the PID computation.
            effective_error = predicted_error
        else:
            effective_error = error

        # Proportional term
        Pout = self.Kp * effective_error if self.Kp != 0 else 0.0

        # Integral term with anti-windup using effective_error
        if self.Ki != 0:
            self.integrals[robot_id] += effective_error * self.dt
            if self.integral_max is not None:
                self.integrals[robot_id] = min(self.integrals[robot_id], self.integral_max)
            if self.integral_min is not None:
                self.integrals[robot_id] = max(self.integrals[robot_id], self.integral_min)
            Iout = self.Ki * self.integrals[robot_id]
        else:
            Iout = 0.0

        # Derivative term based on effective error (already computed above)
        Dout = self.Kd * derivative

        # Combine the PID outputs
        output = Pout + Iout + Dout

        # Store the error and update the time for the next iteration
        self.pre_errors[robot_id] = error
        self.prev_times[robot_id] = call_func_time

        # print(f"x-y PID: {robot_id}, current:{current}, target: {target}, error: {error}, output: {output}")
        if error == 0.0:
            return Vector2D(0.0, 0.0)
        else:
            x_vel = output * (dx / error)
            y_vel = output * (dy / error)
            return self._apply_speed_limits(x_vel, y_vel, self.max_velocity)

    def _apply_speed_limits(self, x_vel: float, y_vel: float, max_vel: float) -> Vector2D:
        current_vel = math.hypot(x_vel, y_vel)
        if current_vel > max_vel:
            scaling_factor = max_vel / current_vel
            x_vel *= scaling_factor
            y_vel *= scaling_factor
        return Vector2D(x_vel, y_vel)

    def reset(self, robot_id: int):
        """Reset the error and integral for the specified robot."""
        self.pre_errors[robot_id] = 0.0
        self.integrals[robot_id] = 0.0
        self.first_pass[robot_id] = True
