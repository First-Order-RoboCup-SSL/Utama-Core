import math

from utama_core.config.settings import TIMESTEP
from utama_core.entities.data.vector import Vector2D
from utama_core.motion_planning.src.pid.pid_abstract import AbstractPID


class PIDAccelerationLimiterWrapper:
    """Wraps a PID controller and limits the acceleration using a fixed time step (dt).

    Maintains separate state for each robot to prevent interference.
    """

    def __init__(self, internal_pid: AbstractPID, max_acceleration: float, dt: float = TIMESTEP):
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
