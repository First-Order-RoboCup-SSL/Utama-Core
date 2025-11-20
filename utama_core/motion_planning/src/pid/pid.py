import math
import time
from typing import Optional

from utama_core.config.enums import Mode
from utama_core.config.settings import SENDING_DELAY
from utama_core.entities.data.vector import Vector2D
from utama_core.global_utils.math_utils import normalise_heading
from utama_core.motion_planning.src.pid.configs import (
    OrientationPIDConfigs,
    PIDConfigs,
    TranslationPIDConfigs,
    get_pid_configs,
)
from utama_core.motion_planning.src.pid.pid_abstract import AbstractPID
from utama_core.motion_planning.src.pid.pid_acceleration_limiter import (
    PIDAccelerationLimiterWrapper,
)


def get_pids(
    mode: Mode,
) -> tuple[PIDAccelerationLimiterWrapper, PIDAccelerationLimiterWrapper]:
    """Instantiate PID controllers from a configuration."""
    config = get_pid_configs(mode)

    pid_oren = PID(config.orientation)
    pid_trans = TwoDPID(config.translation)
    limited_oren = PIDAccelerationLimiterWrapper(
        pid_oren,
        max_acceleration=config.orientation.max_acceleration,
        dt=config.orientation.dt,
    )
    limited_trans = PIDAccelerationLimiterWrapper(
        pid_trans,
        max_acceleration=config.translation.max_acceleration,
        dt=config.translation.dt,
    )
    return limited_oren, limited_trans


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
        config: OrientationPIDConfigs,
    ):
        if config.dt <= 0:
            raise ValueError("dt should be greater than zero")
        self.dt = config.dt
        self.delay = SENDING_DELAY / 1000  # Convert to seconds

        self.max_output = config.max_output
        self.min_output = config.min_output

        self.Kp = config.kp
        self.Kd = config.kd
        self.Ki = config.ki

        self.pre_errors = {i: 0.0 for i in range(6)}
        self.integrals = {i: 0.0 for i in range(6)}

        # Anti-windup limits
        self.integral_min = config.integral_min
        self.integral_max = config.integral_max

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
        # Compute the basic (instantaneous) error
        raw_error = target - current
        # For angular measurements adjust error
        error = normalise_heading(raw_error)
        # For very small errors, return zero
        if abs(error) < 0.001:
            return 0.0

        # Compute derivative term using the previous stored error
        if not self.first_pass[robot_id]:
            derivative = (error - self.pre_errors[robot_id]) / self.dt
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
        config: TranslationPIDConfigs,
    ):
        if config.dt <= 0:
            raise ValueError("dt should be greater than zero")
        self.dt = config.dt
        self.delay = SENDING_DELAY / 1000  # Delay in seconds

        self.max_velocity = config.max_velocity

        self.Kp = config.kp
        self.Kd = config.kd
        self.Ki = config.ki

        self.pre_errors = {i: 0.0 for i in range(6)}
        self.integrals = {i: 0.0 for i in range(6)}

        # Anti-windup limits
        self.integral_min = config.integral_min
        self.integral_max = config.integral_max

        self.first_pass = {i: True for i in range(6)}

    def calculate(self, target: Vector2D, current: Vector2D, robot_id: int) -> Vector2D:
        dx = target[0] - current[0]
        dy = target[1] - current[1]

        error = math.hypot(dx, dy)

        if abs(error) < 3 / 1000:
            return Vector2D(0.0, 0.0)

        # Compute derivative term using the previous stored error
        if not self.first_pass[robot_id]:
            derivative = (error - self.pre_errors[robot_id]) / self.dt
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
