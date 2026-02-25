"""Velocity-based holonomic dynamics for sim-to-real SSL robots.

Unlike HolonomicWithRotation (which maps actions to forces/torques),
this sets agent velocity directly — matching real SSL robots that accept
velocity commands (vx, vy, omega) with onboard PID controllers.
"""

import torch
from vmas.simulator.dynamics.common import Dynamics


class VelocityHolonomic(Dynamics):
    """Holonomic dynamics that interpret actions as target velocities.

    Action dims consumed: 3 (vx, vy, omega).
    - vx, vy are clamped by norm to max_speed (preserving direction).
    - omega is clamped to [-max_angular_vel, max_angular_vel].
    - Forces and torques are zeroed so VMAS physics preserves set velocity.
    """

    def __init__(self, max_speed: float = 2.0, max_angular_vel: float = 4.0):
        super().__init__()
        self._max_speed = max_speed
        self._max_angular_vel = max_angular_vel

    @property
    def needed_action_size(self) -> int:
        return 3

    def process_action(self):
        u = self.agent.action.u

        # Linear velocity: clamp by norm to preserve direction
        vel = u[:, :2]
        speed = torch.norm(vel, dim=-1, keepdim=True)
        scale = torch.where(
            speed > self._max_speed,
            self._max_speed / speed.clamp(min=1e-8),
            torch.ones_like(speed),
        )
        vel = vel * scale
        self.agent.state.vel = vel

        # Angular velocity: scalar clamp
        ang_vel = u[:, 2:3].clamp(-self._max_angular_vel, self._max_angular_vel)
        self.agent.state.ang_vel = ang_vel

        # Zero forces so physics doesn't override our velocities
        # (F=0 → a=0 → v unchanged in VMAS semi-implicit Euler)
        self.agent.state.force = torch.zeros_like(self.agent.state.force)
        self.agent.state.torque = torch.zeros_like(self.agent.state.torque)
