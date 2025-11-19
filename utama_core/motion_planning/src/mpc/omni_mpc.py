"""
Fast Omnidirectional MPC for SSL Robots

Uses linear dynamics (no need for iterative linearization) for faster solving.
Perfect for holonomic/omnidirectional robots that can move in any direction.
"""

import numpy as np
import casadi as ca
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import time


@dataclass
class OmniMPCConfig:
    """Configuration for omnidirectional MPC"""
    # Prediction horizon
    T: int = 15  # Shorter horizon for speed
    DT: float = 0.05  # 50ms timesteps

    # Cost weights
    Q_pos: float = 10.0  # Position tracking weight
    Q_vel: float = 0.1  # Velocity tracking weight
    R_accel: float = 0.01  # Acceleration cost
    R_smooth: float = 0.1  # Smoothness cost
    Q_obstacle: float = 100.0  # Obstacle avoidance weight (INCREASED from 50!)

    # Robot limits
    max_vel: float = 4.0  # m/s
    max_accel: float = 2.0  # m/s^2
    robot_radius: float = 0.09  # m

    # Safety margins (INCREASED to prevent collisions)
    safety_base: float = 0.30  # Base safety radius (3.3 * robot_radius) - LARGER!
    safety_vel_coeff: float = 0.20  # Velocity-dependent component - MORE aggressive!

    # Solver
    max_solve_time: float = 0.01  # 10ms for 60Hz with margin
    verbose: bool = False


class OmnidirectionalMPC:
    """
    Fast MPC for omnidirectional robots with linear dynamics.

    State: [x, y, vx, vy] (no theta needed for translation control)
    Controls: [ax, ay]

    Dynamics (linear!):
        x_{k+1} = x_k + dt * vx_k
        y_{k+1} = y_k + dt * vy_k
        vx_{k+1} = vx_k + dt * ax_k
        vy_{k+1} = vy_k + dt * ay_k
    """

    def __init__(self, config: OmniMPCConfig = None):
        self.config = config or OmniMPCConfig()

        # Build solver once (reuse for speed)
        self.solver = None
        self._build_solver()

        # Warm start
        self.prev_solution = None

        # Stats
        self.solve_time = 0.0
        self.success = True

    def _build_solver(self):
        """Build CasADi solver once for reuse"""
        T = self.config.T
        dt = self.config.DT

        # Decision variables
        X = ca.MX.sym('X', 4, T+1)  # States: [x, y, vx, vy]
        U = ca.MX.sym('U', 2, T)    # Controls: [ax, ay]

        # Parameters (things that change each solve)
        x0 = ca.MX.sym('x0', 4)  # Initial state
        goal = ca.MX.sym('goal', 2)  # Goal position [x, y]
        n_obs = ca.MX.sym('n_obs')  # Number of obstacles
        obs_pos = ca.MX.sym('obs_pos', 2, 10)  # Max 10 obstacles [x, y]
        obs_vel = ca.MX.sym('obs_vel', 2, 10)  # Obstacle velocities
        obs_rad = ca.MX.sym('obs_rad', 10)     # Obstacle radii

        # Cost function
        cost = 0

        # Dynamics constraints (LINEAR!)
        constraints = [X[:, 0] - x0]  # Initial condition
        lbg = [0, 0, 0, 0]
        ubg = [0, 0, 0, 0]

        for k in range(T):
            # Linear dynamics
            x_next = X[0, k] + dt * X[2, k]
            y_next = X[1, k] + dt * X[3, k]
            vx_next = X[2, k] + dt * U[0, k]
            vy_next = X[3, k] + dt * U[1, k]

            dynamics = ca.vertcat(x_next, y_next, vx_next, vy_next) - X[:, k+1]
            constraints.append(dynamics)
            lbg.extend([0, 0, 0, 0])
            ubg.extend([0, 0, 0, 0])

            # Position tracking cost
            pos_error = ca.sumsqr(X[0:2, k] - goal)
            cost += self.config.Q_pos * pos_error

            # Velocity cost (prefer moderate speed)
            vel_error = ca.sumsqr(X[2:4, k])
            cost += self.config.Q_vel * vel_error

            # Control effort
            cost += self.config.R_accel * ca.sumsqr(U[:, k])

            # Control smoothness
            if k > 0:
                du = U[:, k] - U[:, k-1]
                cost += self.config.R_smooth * ca.sumsqr(du)

            # Obstacle avoidance (soft constraints using barrier)
            for i in range(10):  # Max 10 obstacles
                # Skip if i >= n_obs (handled by weighting)
                active = ca.fmin(1.0, ca.fmax(0.0, n_obs - i))

                # Predict obstacle position
                obs_x = obs_pos[0, i] + obs_vel[0, i] * k * dt
                obs_y = obs_pos[1, i] + obs_vel[1, i] * k * dt

                # Distance to obstacle
                dx = X[0, k] - obs_x
                dy = X[1, k] - obs_y
                dist = ca.sqrt(dx**2 + dy**2 + 1e-6)

                # Velocity-dependent safety radius
                robot_speed = ca.sqrt(X[2, k]**2 + X[3, k]**2)
                safety_radius = (self.config.safety_base +
                               self.config.safety_vel_coeff * robot_speed +
                               obs_rad[i])

                # Barrier: penalize if dist < safety_radius
                barrier = ca.exp(-5.0 * (dist - safety_radius))
                cost += active * self.config.Q_obstacle * barrier

        # Terminal cost
        terminal_pos = ca.sumsqr(X[0:2, T] - goal)
        cost += 2.0 * self.config.Q_pos * terminal_pos

        # Variable bounds
        lbx = []
        ubx = []
        for k in range(T+1):
            lbx.extend([-np.inf, -np.inf,  # x, y unbounded
                       -self.config.max_vel, -self.config.max_vel])  # vx, vy bounded
            ubx.extend([np.inf, np.inf,
                       self.config.max_vel, self.config.max_vel])
        for k in range(T):
            lbx.extend([-self.config.max_accel, -self.config.max_accel])
            ubx.extend([self.config.max_accel, self.config.max_accel])

        # Pack decision variables and parameters
        opt_vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
        params = ca.vertcat(x0, goal, n_obs,
                           ca.reshape(obs_pos, -1, 1),
                           ca.reshape(obs_vel, -1, 1),
                           obs_rad)

        # Create NLP
        nlp = {
            'x': opt_vars,
            'f': cost,
            'g': ca.vertcat(*constraints),
            'p': params
        }

        # Solver options
        opts = {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.max_iter': 50,
            'ipopt.max_cpu_time': self.config.max_solve_time,
            'ipopt.warm_start_init_point': 'yes',
        }

        self.solver = ca.nlpsol('mpc_solver', 'ipopt', nlp, opts)
        self.lbg = lbg
        self.ubg = ubg
        self.lbx = lbx
        self.ubx = ubx

        print(f"[OmniMPC] Solver built: T={T}, dt={dt}s, horizon={T*dt:.2f}s")

    def solve(
        self,
        current_state: np.ndarray,  # [x, y, vx, vy]
        goal_pos: Tuple[float, float],
        obstacles: List[Tuple[float, float, float, float, float]] = None
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict]:
        """
        Solve MPC problem.

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

        # Prepare obstacle data (pad to max 10)
        obstacles = obstacles or []
        n_obs = min(len(obstacles), 10)

        obs_pos = np.zeros((2, 10))
        obs_vel = np.zeros((2, 10))
        obs_rad = np.zeros(10)

        for i, obs in enumerate(obstacles[:10]):
            obs_pos[0, i] = obs[0]  # x
            obs_pos[1, i] = obs[1]  # y
            obs_vel[0, i] = obs[2]  # vx
            obs_vel[1, i] = obs[3]  # vy
            obs_rad[i] = obs[4]     # radius

        # Pack parameters
        params = np.concatenate([
            current_state,
            goal_pos,
            [n_obs],
            obs_pos.flatten(),
            obs_vel.flatten(),
            obs_rad
        ])

        # Initial guess (use previous solution if available)
        if self.prev_solution is not None:
            # Shift previous solution
            x0_guess = self.prev_solution
        else:
            # Simple guess: go straight to goal
            x0_guess = np.zeros(4 * (self.config.T + 1) + 2 * self.config.T)
            for k in range(self.config.T + 1):
                x0_guess[4*k:4*k+2] = current_state[:2]  # Stay at current pos
                x0_guess[4*k+2:4*k+4] = [0, 0]  # Zero velocity

        # Solve
        try:
            sol = self.solver(
                x0=x0_guess,
                lbx=self.lbx,
                ubx=self.ubx,
                lbg=self.lbg,
                ubg=self.ubg,
                p=params
            )

            # Extract solution
            sol_x = sol['x'].full().flatten()

            n_states = 4 * (self.config.T + 1)
            states_flat = sol_x[:n_states]
            controls_flat = sol_x[n_states:]

            trajectory = states_flat.reshape((self.config.T + 1, 4))
            controls = controls_flat.reshape((self.config.T, 2))

            # Store for warm start
            self.prev_solution = sol_x

            self.solve_time = time.time() - start_time
            self.success = True

            return controls, trajectory, {
                'success': True,
                'solve_time': self.solve_time,
                'message': 'Success'
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
                vx_cmd = np.clip(2.0 * dx, -self.config.max_vel, self.config.max_vel)
                vy_cmd = np.clip(2.0 * dy, -self.config.max_vel, self.config.max_vel)
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
        self.prev_solution = None
