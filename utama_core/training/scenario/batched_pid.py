"""Batched PyTorch PID controllers for VMAS training scenarios.

Mirrors the algorithm in ``utama_core.motion_planning.src.pid.pid`` (PID and
TwoDPID) but operates on ``(batch_dim, ...)`` tensors for GPU-vectorized
environments.  Reads gains from the shared ``get_pid_configs()`` so both
the strategy layer and training layer share one source of truth.

Features ported from the scalar PID:
- Smith-predictor delay compensation (active when ``SENDING_DELAY > 0``)
- Integral term with anti-windup (active when ``ki > 0``)
- Acceleration limiting on output
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor

from utama_core.config.enums import Mode
from utama_core.config.settings import SENDING_DELAY
from utama_core.motion_planning.src.pid.configs import get_pid_configs


def _angle_wrap(angle: Tensor) -> Tensor:
    """Wrap angle to [-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


class BatchedTranslationPID:
    """Batched 2-D translation PID controller (mirrors ``TwoDPID``).

    State is tracked per agent name in ``(batch_dim,)`` tensors.
    """

    def __init__(self, mode: Mode = Mode.VMAS):
        cfg = get_pid_configs(mode).translation
        self.kp = cfg.kp
        self.kd = cfg.kd
        self.ki = cfg.ki
        self.dt = cfg.dt
        self.max_velocity = cfg.max_velocity
        self.max_acceleration = cfg.max_acceleration
        self.integral_min = cfg.integral_min
        self.integral_max = cfg.integral_max
        self.delay = SENDING_DELAY / 1000.0  # seconds

        # Per-agent state
        self._prev_error: dict[str, Tensor] = {}
        self._integral: dict[str, Tensor] = {}
        self._first_pass: dict[str, Tensor] = {}
        self._prev_output: dict[str, Tensor] = {}  # for acceleration limiting

    def _ensure_state(self, name: str, batch_dim: int, device: torch.device):
        if name not in self._prev_error:
            self._prev_error[name] = torch.zeros(batch_dim, 1, device=device)
            self._integral[name] = torch.zeros(batch_dim, 1, device=device)
            self._first_pass[name] = torch.ones(batch_dim, dtype=torch.bool, device=device)
            self._prev_output[name] = torch.zeros(batch_dim, 2, device=device)

    def calculate(
        self,
        current: Tensor,
        target: Tensor,
        name: str,
    ) -> Tensor:
        """Compute velocity command.  Returns ``(batch_dim, 2)``."""
        batch_dim = current.shape[0]
        device = current.device
        self._ensure_state(name, batch_dim, device)

        diff = target - current  # (batch, 2)
        error = torch.norm(diff, dim=-1, keepdim=True)  # (batch, 1)

        # Dead-zone
        dead = error < 0.003
        safe_error = error.clamp(min=1e-6)

        # Derivative
        prev_e = self._prev_error[name]
        first = self._first_pass[name].unsqueeze(-1)  # (batch, 1)
        derivative = torch.where(first, torch.zeros_like(error), (error - prev_e) / self.dt)
        self._first_pass[name] = torch.zeros_like(self._first_pass[name])

        # Smith-predictor delay compensation
        effective_error = error + derivative * self.delay if self.delay > 0 else error

        # PID terms
        p_out = self.kp * effective_error if self.kp != 0 else torch.zeros_like(error)
        d_out = self.kd * derivative if self.kd != 0 else torch.zeros_like(error)

        i_out = torch.zeros_like(error)
        if self.ki != 0:
            self._integral[name] = self._integral[name] + effective_error * self.dt
            if self.integral_max is not None:
                self._integral[name] = self._integral[name].clamp(max=self.integral_max)
            if self.integral_min is not None:
                self._integral[name] = self._integral[name].clamp(min=self.integral_min)
            i_out = self.ki * self._integral[name]

        output_mag = p_out + i_out + d_out  # (batch, 1)
        self._prev_error[name] = error.detach()

        # Project back to 2D
        direction = diff / safe_error  # (batch, 2)
        vel = torch.where(dead, torch.zeros_like(diff), output_mag * direction)

        # Speed limiting
        speed = torch.norm(vel, dim=-1, keepdim=True)
        scale = torch.where(
            speed > self.max_velocity,
            self.max_velocity / speed.clamp(min=1e-8),
            torch.ones_like(speed),
        )
        vel = vel * scale

        # Acceleration limiting
        if self.max_acceleration > 0:
            allowed = self.max_acceleration * self.dt
            delta = vel - self._prev_output[name]
            delta_norm = torch.norm(delta, dim=-1, keepdim=True)
            accel_scale = torch.where(
                delta_norm > allowed,
                allowed / delta_norm.clamp(min=1e-8),
                torch.ones_like(delta_norm),
            )
            vel = self._prev_output[name] + delta * accel_scale
        self._prev_output[name] = vel.detach()

        return vel

    def reset(self, name: str, batch_index: Optional[int] = None):
        """Reset controller state for *name*.  ``None`` resets all envs."""
        if name not in self._prev_error:
            return
        if batch_index is None:
            self._prev_error[name].zero_()
            self._integral[name].zero_()
            self._first_pass[name].fill_(True)
            self._prev_output[name].zero_()
        else:
            self._prev_error[name][batch_index] = 0.0
            self._integral[name][batch_index] = 0.0
            self._first_pass[name][batch_index] = True
            self._prev_output[name][batch_index] = 0.0


class BatchedOrientationPID:
    """Batched angular PID controller (mirrors ``PID``).

    State is tracked per agent name in ``(batch_dim, 1)`` tensors.
    """

    def __init__(self, mode: Mode = Mode.VMAS):
        cfg = get_pid_configs(mode).orientation
        self.kp = cfg.kp
        self.kd = cfg.kd
        self.ki = cfg.ki
        self.dt = cfg.dt
        self.max_output = cfg.max_output
        self.min_output = cfg.min_output
        self.max_acceleration = cfg.max_acceleration
        self.integral_min = cfg.integral_min
        self.integral_max = cfg.integral_max
        self.delay = SENDING_DELAY / 1000.0

        self._prev_error: dict[str, Tensor] = {}
        self._integral: dict[str, Tensor] = {}
        self._first_pass: dict[str, Tensor] = {}
        self._prev_output: dict[str, Tensor] = {}

    def _ensure_state(self, name: str, batch_dim: int, device: torch.device):
        if name not in self._prev_error:
            self._prev_error[name] = torch.zeros(batch_dim, 1, device=device)
            self._integral[name] = torch.zeros(batch_dim, 1, device=device)
            self._first_pass[name] = torch.ones(batch_dim, dtype=torch.bool, device=device)
            self._prev_output[name] = torch.zeros(batch_dim, 1, device=device)

    def calculate(
        self,
        current: Tensor,
        target: Tensor,
        name: str,
    ) -> Tensor:
        """Compute angular velocity.  Returns ``(batch_dim, 1)``."""
        batch_dim = current.shape[0]
        device = current.device
        self._ensure_state(name, batch_dim, device)

        error = _angle_wrap(target - current)  # (batch, 1)

        # Dead-zone
        dead = error.abs() < 0.001

        # Derivative
        prev_e = self._prev_error[name]
        first = self._first_pass[name].unsqueeze(-1)
        derivative = torch.where(first, torch.zeros_like(error), _angle_wrap(error - prev_e) / self.dt)
        self._first_pass[name] = torch.zeros_like(self._first_pass[name])

        # Smith-predictor
        effective_error = _angle_wrap(target - (current + derivative * self.delay)) if self.delay > 0 else error

        # PID
        p_out = self.kp * effective_error if self.kp != 0 else torch.zeros_like(error)
        d_out = self.kd * derivative if self.kd != 0 else torch.zeros_like(error)

        i_out = torch.zeros_like(error)
        if self.ki != 0:
            self._integral[name] = self._integral[name] + effective_error * self.dt
            if self.integral_max is not None:
                self._integral[name] = self._integral[name].clamp(max=self.integral_max)
            if self.integral_min is not None:
                self._integral[name] = self._integral[name].clamp(min=self.integral_min)
            i_out = self.ki * self._integral[name]

        output = p_out + i_out + d_out
        self._prev_error[name] = error.detach()

        # Clamp output
        output = torch.where(dead, torch.zeros_like(output), output)
        if self.max_output is not None:
            output = output.clamp(max=self.max_output)
        if self.min_output is not None:
            output = output.clamp(min=self.min_output)

        # Acceleration limiting
        if self.max_acceleration > 0:
            allowed = self.max_acceleration * self.dt
            delta = output - self._prev_output[name]
            output = self._prev_output[name] + delta.clamp(-allowed, allowed)
        self._prev_output[name] = output.detach()

        return output

    def reset(self, name: str, batch_index: Optional[int] = None):
        if name not in self._prev_error:
            return
        if batch_index is None:
            self._prev_error[name].zero_()
            self._integral[name].zero_()
            self._first_pass[name].fill_(True)
            self._prev_output[name].zero_()
        else:
            self._prev_error[name][batch_index] = 0.0
            self._integral[name][batch_index] = 0.0
            self._first_pass[name][batch_index] = True
            self._prev_output[name][batch_index] = 0.0
