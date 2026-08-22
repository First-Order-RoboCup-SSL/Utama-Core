from abc import ABC, abstractmethod
from typing import Generic, Optional, TypeVar

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
        self.first_pass = {i: True for i in range(6)}

        # Anti-windup
        self.integral_min = config.integral_min
        self.integral_max = config.integral_max

        # Acceleration limiting
        self.accel_limiter = AccelerationLimiter(
            max_acceleration=config.max_acceleration,
            dt=config.dt,
        )

        # Last target passed to `calculate()`, used by `_target_jumped` to
        # tell a genuine discontinuity (a tactic phase change, hand-off, or
        # reassignment re-aiming this robot at an unrelated target) apart
        # from ordinary tick-to-tick tracking drift (e.g. chasing a moving
        # ball). `pre_errors`/`integrals` are only meaningful as a
        # continuation of tracking the *same* target; on a jump they're
        # reset here automatically instead of relying on every call site to
        # notice and call `reset()` itself — see git history for the
        # tactic-level bugs this replaces.
        self._last_targets: dict[int, T] = {}

    @abstractmethod
    def _calculate(self, target: T, current: T, robot_id: int) -> T:
        """Perform a PID calculation without acceleration limiting."""
        ...

    @abstractmethod
    def _target_jumped(self, target: T, last_target: T) -> bool:
        """Whether `target` differs enough from `last_target` to count as a
        new task rather than a continuation of tracking the same one."""
        ...

    def calculate(self, target: T, current: T, robot_id: int) -> T:
        """Perform a PID calculation."""
        last_target = self._last_targets.get(robot_id)
        if last_target is not None and self._target_jumped(target, last_target):
            self.reset(robot_id)
        self._last_targets[robot_id] = target
        res = self._calculate(target, current, robot_id)
        return self.accel_limiter.limit(robot_id, res)

    def reset(self, robot_id: int):
        """Reset the PID controller state for a given robot."""
        self.pre_errors[robot_id] = 0.0
        self.integrals[robot_id] = 0.0
        self.first_pass[robot_id] = True
        self.accel_limiter.reset(robot_id)
        self._last_targets.pop(robot_id, None)
