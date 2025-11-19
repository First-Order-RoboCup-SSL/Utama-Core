from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from utama_core.config.settings import SENDING_DELAY
from utama_core.motion_planning.src.common.acceleration_limiter import (
    AccelerationLimiter,
)
from utama_core.motion_planning.src.pid.configs import (
    OrientationPIDConfigs,
    TranslationPIDConfigs,
)

T = TypeVar("T")


class AbstractPID(ABC, Generic[T]):
    def __init__(self, config: OrientationPIDConfigs | TranslationPIDConfigs):
        if config.dt <= 0:
            raise ValueError("dt should be greater than zero")

        self.dt = config.dt
        self.delay = SENDING_DELAY / 1000  # seconds

        # PID gains
        self.Kp = config.kp
        self.Kd = config.kd
        self.Ki = config.ki

        # Error tracking for 6 robots
        self.pre_errors = {i: 0.0 for i in range(6)}
        self.integrals = {i: 0.0 for i in range(6)}
        self.prev_times = {i: 0.0 for i in range(6)}
        self.first_pass = {i: True for i in range(6)}

        # Anti-windup
        self.integral_min = config.integral_min
        self.integral_max = config.integral_max

        # Acceleration limiting
        self.accel_limiter = AccelerationLimiter(
            max_acceleration=config.max_acceleration,
            dt=config.dt,
        )

    @abstractmethod
    def _calculate(self, target: T, current: T, robot_id: int) -> T:
        """Perform a PID calculation without acceleration limiting."""
        ...

    def calculate(self, target: T, current: T, robot_id: int) -> T:
        """Perform a PID calculation."""
        res = self._calculate(target, current, robot_id)
        return self.accel_limiter.limit(robot_id, res)

    def reset(self, robot_id: int):
        """Reset the PID controller state for a given robot."""
        self.pre_errors[robot_id] = 0.0
        self.integrals[robot_id] = 0.0
        self.first_pass[robot_id] = True
        self.accel_limiter.reset(robot_id)
