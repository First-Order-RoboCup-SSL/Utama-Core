"""
Fast Omnidirectional MPC for SSL Robots
FINAL POLISHED VERSION
- Velocity-Based Safety Margins (Anti-Crash)
- Soft Constraints (Robustness)
- Anti-Oscillation Logic (Smooth Arrival)
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cvxpy as cp
import numpy as np

from utama_core.config.physical_constants import ROBOT_RADIUS


@dataclass
class OmniMPCConfig:
    """Configuration for omnidirectional MPC"""

    T: int = 5
    DT: float = 0.05

    # Cost weights
    Q_pos: float = 80.0
    Q_vel: float = 200.0
    R_accel: float = 0.01
    R_jerk: float = 0.001
    Q_obstacle: float = 0.0
    Q_slack: float = 5000000.0

    # Limits
    max_vel: float = 2.0
    max_accel: float = 3.0
    robot_radius: float = ROBOT_RADIUS

    # Safety
    obstacle_buffer_ratio: float = 1.25
    safety_vel_coeff: float = 0.15

    max_solve_time: float = 0.010
    verbose: bool = False


class OmnidirectionalMPC:
    def __init__(self, config: OmniMPCConfig = None):
        self.config = config or OmniMPCConfig()
        self._build_dynamics()
        self.prev_states = None
        self.prev_controls = None
        print("[OmniMPC] ANTI-OSCILLATION SOLVER LOADED")

    def _build_dynamics(self):
        dt = self.config.DT
        self.A = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.B = np.array([[0, 0], [0, 0], [dt, 0], [0, dt]])

    def solve(self, current_state, goal_pos, obstacles=None):
        start_time = time.time()
        T = self.config.T
        dt = self.config.DT

        X = cp.Variable((4, T + 1))
        U = cp.Variable((2, T))

        # Slack variable
        max_constraints = 50
        Slacks = cp.Variable(max_constraints, nonneg=True)
        slack_index = 0

        cost = 0
        constraints = [X[:, 0] == current_state]

        # Check if we are already very close to goal (Arrival Mode)
        dist_to_goal_now = np.linalg.norm(np.array(goal_pos) - current_state[0:2])
        is_arriving = dist_to_goal_now < 0.15  # 15cm arrival zone

        # --- DYNAMIC OBSTACLE PREDICTION ---
        hyperplanes_per_step = []
        if obstacles:
            rx, ry, rvx, rvy = current_state
            current_speed = np.hypot(rvx, rvy)
            for k in range(T):
                current_step_planes = []
                time_k = k * dt
                for obs in obstacles:
                    ox_start, oy_start, ovx, ovy, rad = obs
                    ox_k = ox_start + ovx * time_k
                    oy_k = oy_start + ovy * time_k
                    dx = rx - ox_k
                    dy = ry - oy_k
                    dist = np.hypot(dx, dy)

                    # Safety Calculation
                    base_margin = self.config.robot_radius * self.config.obstacle_buffer_ratio
                    vel_margin = current_speed * self.config.safety_vel_coeff
                    safety_dist = base_margin + vel_margin + rad

                    if dist > 0.001:
                        nx = dx / dist
                        ny = dy / dist
                        bx = ox_k + nx * safety_dist
                        by = oy_k + ny * safety_dist
                        current_step_planes.append((nx, ny, bx, by))
                hyperplanes_per_step.append(current_step_planes)

        for k in range(T):
            # Dynamics & Limits
            constraints += [X[:, k + 1] == self.A @ X[:, k] + self.B @ U[:, k]]
            constraints += [cp.norm(X[2:4, k], 2) <= self.config.max_vel]
            constraints += [cp.norm(U[:, k], 2) <= self.config.max_accel]

            # Soft Obstacle Constraints
            if k < len(hyperplanes_per_step):
                for nx, ny, bx, by in hyperplanes_per_step[k]:
                    if slack_index < max_constraints:
                        constraints += [nx * (X[0, k + 1] - bx) + ny * (X[1, k + 1] - by) >= -Slacks[slack_index]]
                        slack_index += 1

            # --- ANTI-OSCILLATION LOGIC ---
            curr_dist = np.linalg.norm(np.array(goal_pos) - current_state[0:2])

            # 1. Target Velocity Logic
            # If we are arriving, FORCE target velocity to 0
            if is_arriving or curr_dist < 0.10:
                ref_vel = np.zeros(2)
            elif curr_dist > 0.05:
                dir_vec = (np.array(goal_pos) - current_state[0:2]) / curr_dist
                ref_vel = dir_vec * self.config.max_vel
            else:
                ref_vel = np.zeros(2)

            cost += self.config.Q_pos * cp.sum_squares(X[0:2, k] - goal_pos)
            cost += self.config.Q_vel * cp.sum_squares(X[2:4, k] - ref_vel)

            # 2. Damping Logic
            # If arriving, increase control cost to prevent twitching
            if is_arriving:
                # High damping (1.0) makes it very expensive to accelerate
                cost += 1.0 * cp.sum_squares(U[:, k])
            else:
                # Low damping (0.01) allows aggressive movement
                cost += self.config.R_accel * cp.sum_squares(U[:, k])

        cost += self.config.Q_pos * cp.sum_squares(X[0:2, T] - goal_pos)
        cost += self.config.Q_slack * cp.sum(Slacks)

        problem = cp.Problem(cp.Minimize(cost), constraints)

        if self.prev_states is not None:
            X.value = self.prev_states
            U.value = self.prev_controls

        try:
            problem.solve(solver=cp.CLARABEL, verbose=False, time_limit=self.config.max_solve_time)

            if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                self.prev_states = X.value
                self.prev_controls = U.value
                return U.value.T, X.value.T, {"success": True, "solve_time": time.time() - start_time}
            else:
                if X.value is not None:
                    return U.value.T, X.value.T, {"success": True, "solve_time": time.time() - start_time}
                return None, None, {"success": False, "solve_time": time.time() - start_time}
        except Exception as e:
            print(f"Error occured: {e}")
            return None, None, {"success": False, "solve_time": time.time() - start_time}

    def get_control_velocities(self, current_state, goal_pos, obstacles=None):
        controls, trajectory, info = self.solve(current_state, goal_pos, obstacles)

        if controls is None:
            info["fallback"] = True
            return 0.0, 0.0, info

        # --- INTELLIGENT LOOKAHEAD ---
        # If we are arriving, look closer (don't overshoot).
        # If we are travelling, look further (accelerate hard).

        dist = np.linalg.norm(goal_pos - current_state[0:2])

        if dist < 0.20:
            # Precision mode: Use immediate next step
            idx = 1
        else:
            # Turbo mode: Use future step
            idx = min(3, self.config.T)

        vx = trajectory[idx, 2]
        vy = trajectory[idx, 3]

        return vx, vy, info
