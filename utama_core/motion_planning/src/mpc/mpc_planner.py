"""
Model Predictive Control (MPC) Planner for SSL Robots

Implements iterative linear MPC with obstacle avoidance for differential drive robots.
Uses CasADi for optimization and C++ code generation.

Author: Adapted from iterative linear MPC approach
"""

import numpy as np
import casadi as ca
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import time

from .mpc_config import MPCConfig


@dataclass
class RobotState:
    """Robot state representation"""
    x: float  # X position [m]
    y: float  # Y position [m]
    theta: float  # Heading angle [rad]
    v: float  # Linear velocity [m/s]
    omega: float  # Angular velocity [rad/s]

    def to_array(self) -> np.ndarray:
        """Convert to numpy array"""
        return np.array([self.x, self.y, self.theta, self.v, self.omega])

    @staticmethod
    def from_array(arr: np.ndarray) -> 'RobotState':
        """Create from numpy array"""
        return RobotState(x=arr[0], y=arr[1], theta=arr[2], v=arr[3], omega=arr[4])


@dataclass
class Obstacle:
    """Dynamic obstacle representation"""
    x: float  # X position [m]
    y: float  # Y position [m]
    vx: float = 0.0  # X velocity [m/s]
    vy: float = 0.0  # Y velocity [m/s]
    radius: float = 0.09  # Obstacle radius [m]
    predicted_positions: Optional[List[Tuple[float, float]]] = None  # Future positions

    def position_at_time(self, t: float) -> Tuple[float, float]:
        """Predict position at future time t assuming constant velocity"""
        if self.predicted_positions is not None and len(self.predicted_positions) > int(t / 0.02):
            # Use shared prediction if available
            idx = min(int(t / 0.02), len(self.predicted_positions) - 1)
            return self.predicted_positions[idx]
        else:
            # Linear extrapolation
            return (self.x + self.vx * t, self.y + self.vy * t)


class MPCPlanner:
    """
    Model Predictive Control planner for local navigation with obstacle avoidance.

    Uses iterative linearization around operational trajectory and CasADi optimization.
    """

    def __init__(self, config: MPCConfig = None):
        """
        Initialize MPC planner.

        Args:
            config: MPC configuration parameters
        """
        self.config = config or MPCConfig()

        # Solver statistics
        self.solve_time = 0.0
        self.iterations = 0
        self.success = True

        # Warm start: store previous solution
        self.prev_controls: Optional[np.ndarray] = None
        self.prev_trajectory: Optional[np.ndarray] = None

    def _get_linearized_dynamics(
        self,
        theta_bar: float,
        v_bar: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get linearized dynamics matrices around operational point.

        Continuous dynamics:
            dx/dt = v * cos(theta)
            dy/dt = v * sin(theta)
            dtheta/dt = omega
            dv/dt = a
            domega/dt = alpha

        Linearized discrete dynamics:
            x[k+1] = A @ x[k] + B @ u[k] + C

        Args:
            theta_bar: Operating point heading [rad]
            v_bar: Operating point velocity [m/s]

        Returns:
            A: State transition matrix (5x5)
            B: Control input matrix (5x2)
            C: Constant offset vector (5,)
        """
        dt = self.config.DT
        cos_theta = np.cos(theta_bar)
        sin_theta = np.sin(theta_bar)

        # State transition matrix A
        A = np.eye(5)
        A[0, 2] = -dt * v_bar * sin_theta  # dx/dtheta
        A[0, 3] = dt * cos_theta  # dx/dv
        A[1, 2] = dt * v_bar * cos_theta  # dy/dtheta
        A[1, 3] = dt * sin_theta  # dy/dv
        A[2, 4] = dt  # dtheta/domega

        # Control input matrix B
        B = np.zeros((5, 2))
        B[3, 0] = dt  # dv/da
        B[4, 1] = dt  # domega/dalpha

        # Constant offset C (linearization residual)
        C = np.zeros(5)
        C[0] = dt * v_bar * sin_theta * theta_bar
        C[1] = -dt * v_bar * cos_theta * theta_bar

        return A, B, C

    def _simulate_step(self, state: RobotState, u: np.ndarray) -> RobotState:
        """
        Simulate one step of nonlinear dynamics.

        Args:
            state: Current robot state
            u: Control input [linear_accel, angular_accel]

        Returns:
            Next state after dt
        """
        dt = self.config.DT
        a, alpha = u[0], u[1]

        # Simple Euler integration (could use RK4 for better accuracy)
        x_next = state.x + state.v * np.cos(state.theta) * dt
        y_next = state.y + state.v * np.sin(state.theta) * dt
        theta_next = state.theta + state.omega * dt
        v_next = state.v + a * dt
        omega_next = state.omega + alpha * dt

        # Apply velocity limits
        v_next = np.clip(v_next, self.config.min_velocity, self.config.max_velocity)
        omega_next = np.clip(omega_next, -self.config.max_angular_velocity,
                            self.config.max_angular_velocity)

        return RobotState(x_next, y_next, theta_next, v_next, omega_next)

    def _predict_trajectory(
        self,
        x0: np.ndarray,
        controls: np.ndarray
    ) -> np.ndarray:
        """
        Predict trajectory using nonlinear dynamics.

        Args:
            x0: Initial state [5,]
            controls: Control sequence [T, 2]

        Returns:
            Trajectory [T+1, 5]
        """
        trajectory = np.zeros((self.config.T + 1, self.config.NX))
        trajectory[0] = x0

        state = RobotState.from_array(x0)
        for t in range(self.config.T):
            state = self._simulate_step(state, controls[t])
            trajectory[t + 1] = state.to_array()

        return trajectory

    def _solve_linear_mpc(
        self,
        x0: np.ndarray,
        xbar: np.ndarray,
        goal: Tuple[float, float],
        obstacles: List[Obstacle],
        prev_u: Optional[np.ndarray] = None
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], bool]:
        """
        Solve linearized MPC problem using CasADi.

        Args:
            x0: Initial state [5,]
            xbar: Operating trajectory [T+1, 5]
            goal: Goal position (x, y)
            obstacles: List of obstacles to avoid
            prev_u: Previous control sequence for warm start

        Returns:
            controls: Optimal control sequence [T, 2] or None
            trajectory: Predicted trajectory [T+1, 5] or None
            success: Whether optimization succeeded
        """
        # Create optimization variables
        x = ca.MX.sym('x', self.config.NX, self.config.T + 1)  # States
        u = ca.MX.sym('u', self.config.NU, self.config.T)  # Controls

        # Cost function
        cost = 0

        # Initial condition constraint
        constraints = [x[:, 0] - x0]
        lbg = [np.zeros(self.config.NX)]
        ubg = [np.zeros(self.config.NX)]

        for t in range(self.config.T):
            # Get linearization around operational point
            A, B, C = self._get_linearized_dynamics(xbar[t, 2], xbar[t, 3])

            # Dynamics constraint: x[t+1] = A @ x[t] + B @ u[t] + C
            constraints.append(x[:, t + 1] - (A @ x[:, t] + B @ u[:, t] + C))
            lbg.append(np.zeros(self.config.NX))
            ubg.append(np.zeros(self.config.NX))

            # Tracking cost - position only (not orientation)
            pos_error = ca.vertcat(x[0, t] - goal[0], x[1, t] - goal[1])
            cost += ca.mtimes([pos_error.T, self.config.Q_pos, pos_error])

            # Velocity cost (prefer forward motion)
            vel_error = ca.vertcat(
                x[2, t],  # theta (don't penalize much)
                x[3, t] - self.config.max_velocity * 0.5,  # prefer medium speed
                x[4, t]   # omega (prefer low)
            )
            cost += ca.mtimes([vel_error.T, self.config.Q_vel, vel_error])

            # Control effort cost
            cost += ca.mtimes([u[:, t].T, self.config.R, u[:, t]])

            # Control smoothness cost
            if t > 0:
                du = u[:, t] - u[:, t - 1]
                cost += ca.mtimes([du.T, self.config.Rd, du])

            # Obstacle avoidance using barrier function
            for obs in obstacles:
                # Predict obstacle position at this timestep
                obs_x, obs_y = obs.position_at_time(t * self.config.DT)

                # Distance to obstacle
                dx = x[0, t] - obs_x
                dy = x[1, t] - obs_y
                dist_sq = dx**2 + dy**2

                # Velocity-dependent safety radius
                safety_radius = self.config.get_velocity_dependent_safety_radius(xbar[t, 3])
                safety_radius_total = safety_radius + obs.radius

                # Barrier cost: penalize when too close
                # Using exponential barrier: exp(-gain * (dist - radius))
                barrier = ca.exp(-self.config.obstacle_barrier_gain *
                                (ca.sqrt(dist_sq) - safety_radius_total))
                cost += self.config.Q_obstacle * barrier

            # Velocity constraints (soft, handled via bounds below)
            # Angular velocity constraints (soft, handled via bounds below)

        # Terminal cost
        terminal_error = ca.vertcat(
            x[0, self.config.T] - goal[0],
            x[1, self.config.T] - goal[1],
            x[2, self.config.T],
            x[3, self.config.T],
            x[4, self.config.T]
        )
        cost += ca.mtimes([terminal_error.T, self.config.Q_terminal, terminal_error])

        # Combine all constraints
        g = ca.vertcat(*constraints)
        lbg = np.concatenate(lbg)
        ubg = np.concatenate(ubg)

        # Decision variables
        opt_variables = ca.vertcat(ca.reshape(x, -1, 1), ca.reshape(u, -1, 1))

        # Bounds on states and controls
        lbx = []
        ubx = []

        # State bounds
        for t in range(self.config.T + 1):
            lbx.extend([-np.inf, -np.inf, -np.inf,  # x, y, theta unbounded
                       self.config.min_velocity, -self.config.max_angular_velocity])
            ubx.extend([np.inf, np.inf, np.inf,  # x, y, theta unbounded
                       self.config.max_velocity, self.config.max_angular_velocity])

        # Control bounds
        for t in range(self.config.T):
            lbx.extend([-self.config.max_acceleration, -self.config.max_angular_acceleration])
            ubx.extend([self.config.max_acceleration, self.config.max_angular_acceleration])

        # Create NLP
        nlp = {'x': opt_variables, 'f': cost, 'g': g}

        # Solver options
        opts = {
            'ipopt.print_level': 0 if not self.config.solver_verbose else 5,
            'print_time': 0,
            'ipopt.max_iter': 100,
            'ipopt.max_cpu_time': self.config.solver_max_time,
            'ipopt.warm_start_init_point': 'yes' if prev_u is not None else 'no',
        }

        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        # Initial guess (warm start if available)
        x0_guess = np.zeros((self.config.NX, self.config.T + 1))
        u0_guess = np.zeros((self.config.NU, self.config.T))

        if prev_u is not None and self.prev_trajectory is not None:
            # Shift previous solution
            u0_guess[:, :-1] = prev_u[:, 1:]
            u0_guess[:, -1] = prev_u[:, -1]
            x0_guess[:, :-1] = self.prev_trajectory[:, 1:].T
            x0_guess[:, -1] = self.prev_trajectory[:, -1]
        else:
            # Simple guess: stay at current state
            for t in range(self.config.T + 1):
                x0_guess[:, t] = xbar[t]

        initial_guess = np.concatenate([x0_guess.flatten(), u0_guess.flatten()])

        # Solve
        try:
            sol = solver(x0=initial_guess, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)

            # Extract solution
            sol_x = sol['x'].full().flatten()

            # Parse states and controls
            n_states = self.config.NX * (self.config.T + 1)
            states_flat = sol_x[:n_states]
            controls_flat = sol_x[n_states:]

            trajectory = states_flat.reshape((self.config.T + 1, self.config.NX))
            controls = controls_flat.reshape((self.config.T, self.config.NU))

            # Check solver status
            success = solver.stats()['success']

            return controls, trajectory, success

        except Exception as e:
            print(f"MPC solver failed: {e}")
            return None, None, False

    def plan(
        self,
        current_state: RobotState,
        goal: Tuple[float, float],
        obstacles: List[Obstacle],
        target_velocity: Optional[float] = None
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict]:
        """
        Plan optimal trajectory using iterative linear MPC.

        Args:
            current_state: Current robot state
            goal: Goal position (x, y)
            obstacles: List of obstacles to avoid
            target_velocity: Desired velocity (default: max_velocity * 0.7)

        Returns:
            controls: Optimal control sequence [T, 2] or None if failed
            trajectory: Predicted trajectory [T+1, 5] or None if failed
            info: Dictionary with planning statistics
        """
        start_time = time.time()

        x0 = current_state.to_array()

        # Initialize controls (use previous if available, else zero)
        if self.prev_controls is not None:
            # Shift and reuse previous controls
            u = np.zeros((self.config.T, self.config.NU))
            u[:-1] = self.prev_controls[1:]
            u[-1] = self.prev_controls[-1]
        else:
            u = np.zeros((self.config.T, self.config.NU))

        # Iterative linearization
        for iteration in range(self.config.max_iter):
            # Predict trajectory with current controls
            xbar = self._predict_trajectory(x0, u)

            # Solve linearized MPC
            u_new, traj_new, success = self._solve_linear_mpc(
                x0, xbar, goal, obstacles, self.prev_controls
            )

            if not success or u_new is None:
                # Solver failed
                self.solve_time = time.time() - start_time
                self.iterations = iteration + 1
                self.success = False
                return None, None, {
                    'success': False,
                    'solve_time': self.solve_time,
                    'iterations': self.iterations,
                    'message': 'Solver failed'
                }

            # Check convergence
            du = np.sum(np.abs(u_new - u))
            u = u_new

            if du < self.config.du_threshold:
                # Converged
                break

        # Store for warm start
        self.prev_controls = u
        self.prev_trajectory = self._predict_trajectory(x0, u)

        # Statistics
        self.solve_time = time.time() - start_time
        self.iterations = iteration + 1
        self.success = True

        return u, self.prev_trajectory, {
            'success': True,
            'solve_time': self.solve_time,
            'iterations': self.iterations,
            'message': 'Success'
        }

    def get_control_command(
        self,
        current_state: RobotState,
        goal: Tuple[float, float],
        obstacles: List[Obstacle]
    ) -> Tuple[float, float, Dict]:
        """
        Get immediate control command for current timestep.

        Args:
            current_state: Current robot state
            goal: Goal position (x, y)
            obstacles: List of obstacles

        Returns:
            linear_accel: Linear acceleration command [m/s²]
            angular_accel: Angular acceleration command [rad/s²]
            info: Planning statistics
        """
        controls, trajectory, info = self.plan(current_state, goal, obstacles)

        if controls is None:
            # Planning failed, return zero controls
            return 0.0, 0.0, info

        # Return first control in sequence
        return controls[0, 0], controls[0, 1], info

    def reset(self):
        """Reset warm start and statistics"""
        self.prev_controls = None
        self.prev_trajectory = None
        self.solve_time = 0.0
        self.iterations = 0
        self.success = True
