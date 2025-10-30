import math
import time
from typing import Optional, Tuple

from utama_core.config.settings import (
    MAX_ACCELERATION,
    MAX_ANGULAR_VEL,
    MAX_VEL,
    REAL_MAX_ANGULAR_VEL,
    REAL_MAX_VEL,
    SENDING_DELAY,
    TIMESTEP,
)
from utama_core.entities.data.vector import Vector2D
from utama_core.global_utils.math_utils import normalise_heading
from utama_core.motion_planning.src.pid.pid_abstract import AbstractPID


# Helper functions to create PID controllers.
def get_real_pids():
    pid_oren = PID(
        TIMESTEP,
        REAL_MAX_ANGULAR_VEL,
        -REAL_MAX_ANGULAR_VEL,
        0.5,
        0.075,
        0,
    )
    pid_trans = TwoDPID(
        TIMESTEP,
        REAL_MAX_VEL,
        0,
        0,
        0.0,
    )
    return PIDAccelerationLimiterWrapper(pid_oren, max_acceleration=0.2), PIDAccelerationLimiterWrapper(
        pid_trans, max_acceleration=0.05
    )


def get_real_pids_goalie():
    pid_oren = PID(
        TIMESTEP,
        REAL_MAX_ANGULAR_VEL,
        -REAL_MAX_ANGULAR_VEL,
        1.5,
        0,
        0,
    )
    pid_trans = TwoDPID(TIMESTEP, 2, 8.5, 0.025, 1)
    return PIDAccelerationLimiterWrapper(pid_oren, max_acceleration=2), PIDAccelerationLimiterWrapper(
        pid_trans, max_acceleration=1
    )


def get_grsim_pids():
    pid_oren = PID(
        TIMESTEP,
        MAX_ANGULAR_VEL,
        -MAX_ANGULAR_VEL,
        4.5,
        0.02,
        0,
        integral_min=-10,
        integral_max=10,
    )
    pid_trans = TwoDPID(
        TIMESTEP,
        MAX_VEL,
        1.8,
        0.025,
        0.0,
        integral_min=-5,
        integral_max=5,
    )
    return PIDAccelerationLimiterWrapper(pid_oren, max_acceleration=50, dt=TIMESTEP), PIDAccelerationLimiterWrapper(
        pid_trans, max_acceleration=2, dt=TIMESTEP
    )


def get_rsim_pids():
    pid_oren = PID(
        TIMESTEP,
        MAX_ANGULAR_VEL,
        -MAX_ANGULAR_VEL,
        4.5,
        0.02,
        0,
        integral_min=-10,
        integral_max=10,
    )
    pid_trans = TwoDPID(
        TIMESTEP,
        MAX_VEL,
        1.8,
        0.025,
        0.0,
        integral_min=-5,
        integral_max=5,
    )
    return PIDAccelerationLimiterWrapper(pid_oren, max_acceleration=50, dt=TIMESTEP), PIDAccelerationLimiterWrapper(
        pid_trans, max_acceleration=2, dt=TIMESTEP
    )


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
            dt = measured_dt if measured_dt > 0 else TIMESTEP

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
            # Use the measured dt if nonzero; otherwise fall back to TIMESTEP.
            dt = measured_dt if measured_dt > 0 else TIMESTEP

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
            return self.scale_velocity(x_vel, y_vel, self.max_velocity)

    def scale_velocity(self, x_vel: float, y_vel: float, max_vel: float) -> Vector2D:
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


class PIDAccelerationLimiterWrapper:
    """Wraps a PID controller and limits the acceleration using a fixed time step (dt).

    Maintains separate state for each robot to prevent interference.
    """

    def __init__(self, internal_pid: AbstractPID, max_acceleration: float = MAX_ACCELERATION, dt: float = TIMESTEP):
        self._internal_pid = internal_pid
        self._last_results = {}  # Key: robot_id
        self._max_acceleration = max_acceleration
        self.dt = dt

    def calculate(self, *args, **kwargs):
        result = self._internal_pid.calculate(*args, **kwargs)

        # Extract robot_id from arguments (3rd positional arg or kwargs)
        robot_id = args[2] if len(args) >= 3 else kwargs.get("robot_id", 0)

        # Get last result for this robot
        last_result = self._last_results.get(robot_id, None)

        # Calculate maximum allowed change per timestep
        dv_allowed = self._max_acceleration * self.dt

        if isinstance(result, (float, int)):
            # Handle scalar outputs
            last_val = 0.0 if last_result is None else last_result
            diff = result - last_val
            diff = max(-dv_allowed, min(dv_allowed, diff))
            limited_result = last_val + diff
        elif isinstance(result, Vector2D):
            # Handle 2D vector outputs
            if last_result is None:
                last_vec = Vector2D(0.0, 0.0)
            elif isinstance(last_result, Vector2D):
                last_vec = last_result
            else:
                last_vec = Vector2D(last_result[0], last_result[1])

            dx = result.x - last_vec.x
            dy = result.y - last_vec.y
            norm_diff = math.hypot(dx, dy)

            if norm_diff <= dv_allowed:
                limited_result = result
            else:
                scale = dv_allowed / norm_diff
                limited_result = Vector2D(last_vec.x + dx * scale, last_vec.y + dy * scale)
                # assert isinstance(limited_result, float)
        else:
            raise NotImplementedError(f"Unsupported output type: {type(result)}")

        # Update stored state
        self._last_results[robot_id] = limited_result
        # print(f"Result: {result}, Limited Result: {limited_result}, last_result: {last_result}")
        return limited_result

    def reset(self, robot_id: int):
        """Reset both the internal PID and acceleration state for this robot."""
        self._internal_pid.reset(robot_id)
        if robot_id in self._last_results:
            del self._last_results[robot_id]


if __name__ == "__main__":
    # Example usage for testing purposes.
    pid_trans = TwoDPID(TIMESTEP, MAX_VEL, 8.5, 0.025, 0.0, num_robots=6)
    pid_trans2 = TwoDPID(TIMESTEP, MAX_VEL, 8.5, 0.025, 0.0, num_robots=6)

    pid_trans_limited = PIDAccelerationLimiterWrapper(pid_trans2, max_acceleration=0.05)
    for pid in [pid_trans, pid_trans_limited]:
        result = pid.calculate((100, 100), (0, 0), 0)
        print(result)
        print(math.hypot(result[0], result[1]))
