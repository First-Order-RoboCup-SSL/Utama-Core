"""
Fast Omnidirectional MPC for SSL Robots

CVXPY-based implementation for numerical stability.
Uses linear dynamics (no iterative linearization needed).
Perfect for holonomic/omnidirectional robots.
"""

import numpy as np
import cvxpy as cp
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import time


@dataclass
class OmniMPCConfig:
    """Configuration for omnidirectional MPC"""
    # Prediction horizon
    T: int = 5          # Reduced from 8 -> 5 (Crucial for 60Hz performance)
    DT: float = 0.05    # Reduced from 0.1 -> 0.05 (Faster updates)

    # Cost weights 
    Q_pos: float = 80.0   
    Q_vel: float = 200.0  # High velocity cost forces acceleration
    R_accel: float = 0.01 # Cheap acceleration
    R_jerk: float = 0.001 
    Q_obstacle: float = 50.0 

    # Robot limits 
    max_vel: float = 4.0    
    max_accel: float = 15.0 # Fake high acceleration to stop internal limiting
    robot_radius: float = 0.09  

    # Safety margins
    safety_base: float = 0.15  
    safety_vel_coeff: float = 0.05 

    # Solver
    max_solve_time: float = 0.015 
    verbose: bool = False

"""
@dataclass
class OmniMPCConfig:
    ###Configuration for omnidirectional MPC
    # Prediction horizon (shorter for speed)
    T: int = 8  # 8 steps
    DT: float = 0.1  # 100ms timestep = 0.8s total lookahead

    # Cost weights (BALANCED for reliable convergence)
    Q_pos: float = 100.0  # Position tracking
    Q_vel: float = 50.0  # Velocity tracking (prefer moving fast)
    R_accel: float = 0.01  # Acceleration cost
    R_jerk: float = 0.001  # Smoothness
    Q_obstacle: float = 50.0  # Obstacle avoidance (will add later)

    # Robot limits (aggressive for SSL)
    max_vel: float = 4.0  # m/s
    max_accel: float = 6.0  # m/s^2 (SSL robots can do 3-4 m/s^2, use 6 for responsiveness)
    robot_radius: float = 0.09  # m

    # Safety margins
    safety_base: float = 0.15  # Base safety radius
    safety_vel_coeff: float = 0.05  # Velocity-dependent component

    # Solver
    max_solve_time: float = 0.015  # 15ms max for 60Hz
    verbose: bool = False  # Disable for speed
"""

class OmnidirectionalMPC:
    """
    CVXPY-based MPC for omnidirectional robots with linear dynamics.

    State: [x, y, vx, vy] (no theta needed for translation control)
    Controls: [ax, ay]

    Linear dynamics:
        x_{k+1} = x_k + dt * vx_k
        y_{k+1} = y_k + dt * vy_k
        vx_{k+1} = vx_k + dt * ax_k
        vy_{k+1} = vy_k + dt * ay_k

    Matrix form: x_{k+1} = A @ x_k + B @ u_k
    """

    def __init__(self, config: OmniMPCConfig = None):
        self.config = config or OmniMPCConfig()

        # Build dynamics matrices
        self._build_dynamics()

        # Warm start
        self.prev_states = None
        self.prev_controls = None

        # Stats
        self.solve_time = 0.0
        self.success = True

        print(f"[OmniMPC] CVXPY solver initialized: T={self.config.T}, "
              f"dt={self.config.DT}s, horizon={self.config.T * self.config.DT:.2f}s")

    def _build_dynamics(self):
        """Build linear dynamics matrices A and B"""
        dt = self.config.DT

        # State transition matrix: x_{k+1} = A @ x_k + B @ u_k
        # State: [x, y, vx, vy]
        # Control: [ax, ay]

        self.A = np.array([
            [1, 0, dt, 0],   # x_next = x + dt * vx
            [0, 1, 0, dt],   # y_next = y + dt * vy
            [0, 0, 1, 0],    # vx_next = vx + dt * ax
            [0, 0, 0, 1]     # vy_next = vy + dt * ay
        ])

        self.B = np.array([
            [0, 0],          # x doesn't directly depend on control
            [0, 0],          # y doesn't directly depend on control
            [dt, 0],         # vx += dt * ax
            [0, dt]          # vy += dt * ay
        ])

    def solve(
        self,
        current_state: np.ndarray,  # [x, y, vx, vy]
        goal_pos: Tuple[float, float],
        obstacles: List[Tuple[float, float, float, float, float]] = None
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict]:
        """
        Solve MPC problem using CVXPY.

        Args:
            current_state: [x, y, vx, vy]
            goal_pos: (goal_x, goal_y)
            obstacles: List of (x, y, vx, vy, radius)

        Returns:
            controls: [T, 2] array of [ax, ay] or None
            trajectory: [T+1, 4] array of [x, y, vx, vy] or None
            info: Dict with solve stats
        """
        start_time = time.time()

        T = self.config.T
        dt = self.config.DT

        # Decision variables
        X = cp.Variable((4, T+1))  # States: [x, y, vx, vy] at each timestep
        U = cp.Variable((2, T))    # Controls: [ax, ay] at each timestep

        # Cost function
        cost = 0

        # Initial condition constraint
        constraints = [X[:, 0] == current_state]

        for k in range(T):
            # Dynamics constraints (linear!)
            constraints += [X[:, k+1] == self.A @ X[:, k] + self.B @ U[:, k]]

            # Velocity limits (total speed)
            constraints += [
                cp.norm(X[2:4, k], 2) <= self.config.max_vel,  # Speed limit
            ]

            # Acceleration limits (total acceleration magnitude)
            constraints += [
                cp.norm(U[:, k], 2) <= self.config.max_accel,  # Total accel limit
            ]

            # Position tracking cost (drive towards goal)
            pos_error = cp.sum_squares(X[0:2, k] - goal_pos)
            cost += self.config.Q_pos * pos_error

            # Velocity cost (prefer moving, not stopping)
            # Want to move fast towards goal
            goal_dir = np.array(goal_pos) - current_state[0:2]
            goal_dist = np.linalg.norm(goal_dir)
            if goal_dist > 0.1:
                desired_speed = min(self.config.max_vel * 0.9, goal_dist / (T * dt))
                desired_vel = (goal_dir / goal_dist) * desired_speed

                # Debug output on first iteration
                if k == 0 and start_time is not None:
                    print(f"[MPC Debug] goal_dist={goal_dist:.2f}m, desired_speed={desired_speed:.2f}m/s, "
                          f"desired_vel=({desired_vel[0]:.2f}, {desired_vel[1]:.2f})")

                vel_error = cp.sum_squares(X[2:4, k] - desired_vel)
                cost += self.config.Q_vel * vel_error

            # Control effort (penalize large accelerations)
            cost += self.config.R_accel * cp.sum_squares(U[:, k])

            # Control smoothness (penalize jerk)
            if k > 0:
                jerk = cp.sum_squares(U[:, k] - U[:, k-1])
                cost += self.config.R_jerk * jerk

            # TODO: Add obstacle avoidance with DCP-compliant formulation
            # For now, skip obstacles to get basic MPC working
            # Will add back using linearization or SOC constraints
            pass

        # Terminal cost (prefer being at goal at end)
        terminal_pos_error = cp.sum_squares(X[0:2, T] - goal_pos)
        cost += self.config.Q_pos * terminal_pos_error

        # Only penalize terminal velocity if very close to goal (within 0.5m)
        # This allows robot to move fast when far from goal
        # Note: can't use conditionals in CVXPY, so we skip this for now
        # The position cost will naturally slow the robot as it approaches

        # Create problem
        problem = cp.Problem(cp.Minimize(cost), constraints)

        # Warm start if available
        if self.prev_states is not None and self.prev_controls is not None:
            try:
                # Shift previous solution
                X.value = self.prev_states
                U.value = self.prev_controls
            except:
                pass

        # Solve
        try:
            problem.solve(
                solver=cp.CLARABEL,
                verbose=self.config.verbose,
                max_iter=50,
                time_limit=self.config.max_solve_time
            )

            if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                # Extract solution
                states = X.value
                controls = U.value

                # Debug: print trajectory velocities
                print(f"[MPC Debug] Planned velocities over horizon:")
                for k in range(min(3, T+1)):  # Print first 3 timesteps
                    vx, vy = states[2, k], states[3, k]
                    speed = np.hypot(vx, vy)
                    print(f"  t={k*self.config.DT:.2f}s: vx={vx:.3f}, vy={vy:.3f}, speed={speed:.3f} m/s")

                # Store for warm start
                self.prev_states = states
                self.prev_controls = controls

                self.solve_time = time.time() - start_time
                self.success = True

                return controls.T, states.T, {
                    'success': True,
                    'solve_time': self.solve_time,
                    'message': f'{problem.status}',
                    'cost': problem.value
                }
            else:
                # Solver didn't find optimal solution
                self.solve_time = time.time() - start_time
                self.success = False

                return None, None, {
                    'success': False,
                    'solve_time': self.solve_time,
                    'message': f'Solver status: {problem.status}'
                }

        except Exception as e:
            self.solve_time = time.time() - start_time
            self.success = False
            print(f"[OmniMPC] Solve failed: {e}")

            return None, None, {
                'success': False,
                'solve_time': self.solve_time,
                'message': str(e)
            }

    def get_control_velocities(
        self,
        current_state: np.ndarray,
        goal_pos: Tuple[float, float],
        obstacles: List[Tuple[float, float, float, float, float]] = None
    ) -> Tuple[float, float, Dict]:
        """
        Get velocity commands for current timestep.

        Returns:
            vx_cmd: X velocity command
            vy_cmd: Y velocity command
            info: Solve stats
        """
        controls, trajectory, info = self.solve(current_state, goal_pos, obstacles)

        if controls is None or trajectory is None:
            # Fallback: simple proportional control
            dx = goal_pos[0] - current_state[0]
            dy = goal_pos[1] - current_state[1]
            dist = np.hypot(dx, dy)

            if dist > 0.01:
                # Proportional control with high gain
                vx_cmd = np.clip(3.0 * dx, -self.config.max_vel, self.config.max_vel)
                vy_cmd = np.clip(3.0 * dy, -self.config.max_vel, self.config.max_vel)
            else:
                vx_cmd, vy_cmd = 0.0, 0.0

            info['fallback'] = True
            return vx_cmd, vy_cmd, info

        # Use first velocity from planned trajectory
        vx_cmd = trajectory[1, 2]  # Next velocity (not current)
        vy_cmd = trajectory[1, 3]

        info['fallback'] = False
        return vx_cmd, vy_cmd, info

    def reset(self):
        """Reset warm start"""
        self.prev_states = None
        self.prev_controls = None
