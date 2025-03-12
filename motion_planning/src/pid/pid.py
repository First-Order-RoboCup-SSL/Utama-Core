import numpy as np
from typing import Optional, Tuple
from motion_planning.src.pid.pid_abstract import AbstractPID
import time
import math
from config.settings import (
    TIMESTEP,
    MAX_ANGULAR_VEL,
    MAX_VEL,
    REAL_MAX_ANGULAR_VEL,
    REAL_MAX_VEL,
)


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
    return pid_oren, PIDAccelerationLimiterWrapper(pid_trans, max_acceleration=0.05)


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
    return PIDAccelerationLimiterWrapper(
        pid_oren, max_acceleration=2
    ), PIDAccelerationLimiterWrapper(pid_trans, max_acceleration=1)


def get_grsim_pids():
    pid_oren = PID(
        TIMESTEP,
        MAX_ANGULAR_VEL,
        -MAX_ANGULAR_VEL,
        20.5,
        0.075,
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
    return pid_oren, PIDAccelerationLimiterWrapper(
        pid_trans, max_acceleration=2, dt=TIMESTEP
    )


def get_rsim_pids():
    pid_oren = PID(
        TIMESTEP,
        MAX_ANGULAR_VEL,
        -MAX_ANGULAR_VEL,
        20.5,
        0.075,
        0,
        integral_min=-10,
        integral_max=10,
    )
    pid_trans = TwoDPID(
        TIMESTEP,
        MAX_VEL,
        1.8,
        0.025,
        0,
        integral_min=-5,
        integral_max=5,
    )
    return pid_oren, PIDAccelerationLimiterWrapper(
        pid_trans, max_acceleration=2, dt=TIMESTEP
    )


class PID(AbstractPID[float]):
    """
    A Proportional-Integral-Derivative (PID) controller for managing error corrections.

    Args:
        dt (float): Time step for each update.
        max_output (Optional[float]): Maximum output value (None for no limit).
        min_output (Optional[float]): Minimum output value (None for no limit).
        Kp (float): Proportional gain.
        Kd (float): Derivative gain.
        Ki (float): Integral gain.
        num_robots (int): Number of robots (each maintains its own error tracking).
        integral_min (Optional[float]): Minimum allowed integral value.
        integral_max (Optional[float]): Maximum allowed integral value.
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

        self.prev_time = 0

        self.first_pass = {i: True for i in range(6)}

        self.errors = []

    def calculate(
        self,
        target: float,
        current: float,
        robot_id: int,
        oren: bool = False,
        normalize_range: Optional[float] = None,
    ) -> float:
        """
        Compute the PID output to move a robot towards a target.

        Args:
            target (float): Desired value.
            current (float): Current value.
            robot_id (int): Unique robot identifier.
            oren (bool): If True, treats error as an angular difference.
            normalize_range (Optional[float]): If provided, scales the output.

        Returns:
            float: Clamped PID output.
        """
        call_func_time = time.time()
        error = target - current

        # Adjust error for angular measurements
        if oren:
            error = np.arctan2(np.sin(error), np.cos(error))

        # Proportional term
        Pout = self.Kp * error if self.Kp != 0 else 0.0

        # Integral term with anti-windup
        if self.Ki != 0:
            self.integrals[robot_id] += error * self.dt
            if self.integral_max is not None:
                self.integrals[robot_id] = min(
                    self.integrals[robot_id], self.integral_max
                )
            if self.integral_min is not None:
                self.integrals[robot_id] = max(
                    self.integrals[robot_id], self.integral_min
                )
            Iout = self.Ki * self.integrals[robot_id]
        else:
            Iout = 0.0

        # Derivative term
        if self.Kd != 0 and not self.first_pass[robot_id]:
            if round(call_func_time - self.prev_time, 4) > 0:
                dt = round(call_func_time - self.prev_time, 4)
            else:
                dt = 0.016

            derivative = (error - self.pre_errors[robot_id]) / dt
            Dout = self.Kd * derivative
        else:
            Dout = 0.0
            self.first_pass[robot_id] = False

        output = Pout + Iout + Dout

        # Apply optional normalization
        if normalize_range is not None and normalize_range != 0:
            output /= normalize_range

        # Consistent output clamping
        if self.max_output is not None:
            output = min(self.max_output, output)
        if self.min_output is not None:
            output = max(self.min_output, output)

        self.pre_errors[robot_id] = error
        self.prev_time = time.time()
        # self.errors.append(error)
        # print(f"Error: {error}, Avg Error: {np.mean(self.errors)}")
        return output

    def reset(self, robot_id: int):
        """Reset the error and integral for the specified robot."""
        self.pre_errors[robot_id] = 0.0
        self.integrals[robot_id] = 0.0
        self.first_pass[robot_id] = True


class TwoDPID(AbstractPID[Tuple[float, float]]):
    """
    A 2D PID controller that independently controls the X and Y dimensions and scales
    the resulting velocity vector to a maximum speed if needed.
    """

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

        self.max_velocity = max_velocity

        self.Kp = Kp
        self.Kd = Kd
        self.Ki = Ki

        self.pre_errors = {i: 0.0 for i in range(6)}
        self.integrals = {i: 0.0 for i in range(6)}

        # Anti-windup limits
        self.integral_min = integral_min
        self.integral_max = integral_max

        self.prev_time = 0

        self.first_pass = {i: True for i in range(6)}

    def calculate(
        self, target: Tuple[float, float], current: Tuple[float, float], robot_id: int
    ) -> Tuple[float, float]:
        call_func_time = time.time()

        dx = target[0] - current[0]
        dy = target[1] - current[1]

        error = math.hypot(dx, dy)

        # Proportional term
        Pout = self.Kp * error if self.Kp != 0 else 0.0

        # Integral term with anti-windup
        if self.Ki != 0:
            self.integrals[robot_id] += error * self.dt
            if self.integral_max is not None:
                self.integrals[robot_id] = min(
                    self.integrals[robot_id], self.integral_max
                )
            if self.integral_min is not None:
                self.integrals[robot_id] = max(
                    self.integrals[robot_id], self.integral_min
                )
            Iout = self.Ki * self.integrals[robot_id]
        else:
            Iout = 0.0

        # Derivative term
        if self.Kd != 0 and not self.first_pass[robot_id]:
            if round(call_func_time - self.prev_time, 4) > 0:
                dt = round(call_func_time - self.prev_time, 4)
            else:
                dt = 0.016

            derivative = (error - self.pre_errors[robot_id]) / dt
            Dout = self.Kd * derivative
        else:
            Dout = 0.0
            self.first_pass[robot_id] = False

        output = Pout + Iout + Dout

        self.pre_errors[robot_id] = error
        self.prev_time = time.time()

        # Convert output to directional velocities
        if error == 0.0:
            return 0.0, 0.0
        else:
            x_vel = output * (dx / error)
            y_vel = output * (dy / error)
            return self.scale_velocity(x_vel, y_vel, self.max_velocity)

    def scale_velocity(
        self, x_vel: float, y_vel: float, max_vel: float
    ) -> Tuple[float, float]:
        current_vel = math.hypot(x_vel, y_vel)
        if current_vel > max_vel:
            scaling_factor = max_vel / current_vel
            x_vel *= scaling_factor
            y_vel *= scaling_factor
        return x_vel, y_vel

    def reset(self, robot_id: int):
        """Reset the error and integral for the specified robot."""
        self.pre_errors[robot_id] = 0.0
        self.integrals[robot_id] = 0.0
        self.first_pass[robot_id] = True


class PIDAccelerationLimiterWrapper:
    """
    Wraps a PID controller and limits the acceleration using a fixed time step (dt).
    Maintains separate state for each robot to prevent interference.
    """

    def __init__(
        self, internal_pid: AbstractPID, max_acceleration: float, dt: float = TIMESTEP
    ):
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
        elif isinstance(result, tuple):
            # Handle 2D vector outputs
            last_val = (0.0, 0.0) if last_result is None else last_result
            dx = result[0] - last_val[0]
            dy = result[1] - last_val[1]
            norm_diff = math.hypot(dx, dy)

            if norm_diff <= dv_allowed:
                limited_result = result
            else:
                scale = dv_allowed / norm_diff
                limited_result = (last_val[0] + dx * scale, last_val[1] + dy * scale)
        else:
            raise NotImplementedError(f"Unsupported output type: {type(result)}")

        # Update stored state
        self._last_results[robot_id] = limited_result
        # print(f"Result: {result}, Limited Result: {limited_result}, last_result: {last_result}")
        return limited_result

    def reset(self, robot_id: int):
        """Reset both the internal PID and acceleration state for this robot"""
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
