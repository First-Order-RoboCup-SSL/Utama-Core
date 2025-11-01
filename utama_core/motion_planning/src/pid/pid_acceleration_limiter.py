from utama_core.config.settings import TIMESTEP
from utama_core.motion_planning.src.common.acceleration_limiter import (
    AccelerationLimiter,
)
from utama_core.motion_planning.src.pid.pid_abstract import AbstractPID


class PIDAccelerationLimiterWrapper:
    """Wraps a PID controller and limits the acceleration using a fixed time step (dt).

    Maintains separate state for each robot to prevent interference.
    """

    def __init__(self, internal_pid: AbstractPID, max_acceleration: float, dt: float = TIMESTEP):
        self._internal_pid = internal_pid
        self._limiter = AccelerationLimiter(max_acceleration=max_acceleration, dt=dt)

    def calculate(self, *args, **kwargs):
        result = self._internal_pid.calculate(*args, **kwargs)

        # Extract robot_id from arguments (3rd positional arg or kwargs)
        robot_id = args[2] if len(args) >= 3 else kwargs.get("robot_id", 0)

        return self._limiter.limit(robot_id, result)

    def reset(self, robot_id: int):
        """Reset both the internal PID and acceleration state for this robot."""
        self._internal_pid.reset(robot_id)
        self._limiter.reset(robot_id)
